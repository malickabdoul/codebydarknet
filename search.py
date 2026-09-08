"""
search.py
----------
Interroge la base vectorielle construite par build_index.py.
Fonctionne en français, en anglais, ou en mooré.

Usage :
    python3 search.py "eau"
    python3 search.py "how are you"
    python3 search.py "ko"
    python3 search.py "eau" --domaine Water
    python3 search.py "koom" --backend entrees

Backends :
    entrees     n-grammes de caracteres sur les EN-TETES seuls (mots moore)
    tfidf       l'index d'origine : en-tete + definitions melanges
    semantique  embeddings pre-entraines sur les definitions fr/en
    hybride     fusion RRF de tfidf et semantique
    auto        defaut : route selon la langue de la requete

Voir la section 9 du README et python3 benchmark.py pour les chiffres.
"""

import os
import pickle
import re
import sys
import unicodedata

import faiss
import numpy as np
import pandas as pd
from scipy.sparse import hstack

import embeddings

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_PATH = os.path.join(BASE_DIR, "vectors.index")
VECTORIZERS_PATH = os.path.join(BASE_DIR, "vectorizers.pkl")
META_PATH = os.path.join(BASE_DIR, "corpus_meta.pkl")

# Les trois fichiers pèsent ~70 Mo au total. Les relire à chaque appel coûtait
# plus d'une demi-seconde par recherche : on les garde en mémoire pour la durée
# du processus.
_CACHE = None
_DOMAINES = None


def load_all():
    global _CACHE
    if _CACHE is None:
        index = faiss.read_index(INDEX_PATH)
        with open(VECTORIZERS_PATH, "rb") as f:
            vec = pickle.load(f)
        meta = pd.read_pickle(META_PATH)
        _CACHE = (index, vec, meta)
    return _CACHE


def _clean(v):
    return "" if v is None or pd.isna(v) else str(v)


def _liste(valeur):
    """« wãbre, yãbre1, warpusi » -> ['wãbre', 'yãbre', 'warpusi'].

    Les renvois du corpus portent un chiffre quand le mot a plusieurs sens
    (« ma1 » = premier sens de « ma »). Ce chiffre ne fait pas partie du mot :
    sur les 3 545 renvois de synonymes, 2 800 correspondent à une entrée tels
    quels, et 3 429 une fois le chiffre retiré.
    """
    texte = _clean(valeur)
    if not texte:
        return []
    out = []
    for morceau in texte.split(","):
        mot = re.sub(r"\d+$", "", morceau.strip()).strip()
        if mot and mot not in out:
            out.append(mot)
    return out


def marquer_tronquee(definition):
    """Signale une définition coupée à la source.

    57 définitions françaises et 17 anglaises s'arrêtent net au milieu d'une
    parenthèse : « têtu (lit », « tomber (ex », « build, construct (e.g ». Le
    scraper a pris le point de l'abréviation (« litt. », « ex. », « e.g. »)
    pour une fin de phrase et a jeté la suite. Le texte manquant n'est pas
    dans le CSV : impossible de le reconstituer.

    On ajoute donc « […] » pour que ça se lise comme une donnée incomplète et
    non comme un bug d'affichage.
    """
    if definition.count("(") > definition.count(")"):
        return definition.rstrip(". ") + " […]"
    return definition


_ETIQUETTE = re.compile(r"^\s*(synonymes?|antonymes?)\s*:\s*", re.IGNORECASE)


def renvois(valeur_syn, valeur_ant):
    """Démêle les colonnes « synonymes » et « antonymes ».

    Le corpus source les mélange : 28 champs « synonymes » contiennent en
    réalité « ...; antonyme: X », et 29 champs « antonymes » contiennent
    « ...; synonyme: X ». Sans démêlage, ces renvois atterrissent dans la
    mauvaise colonne — « ba » (père) affichait « saamba; antonyme: ma » comme
    un seul synonyme.

    Renvoie (synonymes, antonymes).
    """
    resultat = {"syn": [], "ant": []}
    for valeur, defaut in ((valeur_syn, "syn"), (valeur_ant, "ant")):
        for bloc in _clean(valeur).split(";"):
            cible = defaut
            etiquette = _ETIQUETTE.match(bloc)
            if etiquette:
                cible = "ant" if etiquette.group(1).lower().startswith("ant") else "syn"
                bloc = bloc[etiquette.end():]
            for mot in _liste(bloc):
                if mot not in resultat[cible]:
                    resultat[cible].append(mot)
    return resultat["syn"], resultat["ant"]


def nettoyer_mot_moore(mot):
    """Répare les en-têtes du corpus source où du français/anglais est collé.

    Sept entrées mélangent la traduction et le mot mooré :

        'Come! y'                               -> 'y'
        'why? bõeedga'                          -> 'bõeedga'
        'Ne y waoongo ! Bienvenues ! Welcome! waoongo'  -> 'waoongo'
        'Ges neere! Regarde bien ! Watch well!' -> 'Ges neere'

    Le mot mooré est le morceau qui suit la dernière ponctuation — sauf quand
    la chaîne se termine par « ! » ou « ? », auquel cas il n'y a pas de mot
    final et c'est le premier morceau qui est le mooré (dernier cas ci-dessus).
    """
    mot = str(mot).strip()
    if "!" not in mot and "?" not in mot:
        return mot
    morceaux = [m.strip() for m in re.split(r"[!?]", mot)]
    morceaux = [m for m in morceaux if m]
    if not morceaux:
        return mot
    return morceaux[0] if mot[-1] in "!?" else morceaux[-1]


# --- domaines -------------------------------------------------------------
# La colonne « domaine » contient soit un domaine simple ("Weather"), soit
# plusieurs domaines collés ("Agriculture, Water"). Un split sur ", " serait
# faux : certains domaines contiennent eux-mêmes une virgule ("Bush, shrub",
# "Grass, herb, vine"). On construit donc le vocabulaire des domaines atomiques
# à partir du corpus, puis on découpe par correspondance la plus longue.

def _build_vocabulaire(valeurs):
    vocab = {v for v in valeurs if ", " not in v}
    # Traiter les valeurs les plus courtes d'abord : "Bush, shrub" doit être
    # reconnu comme atome avant qu'on essaie de découper "Food, Bush, shrub".
    for v in sorted(valeurs, key=lambda s: (s.count(", "), s)):
        reste = v
        while reste:
            atome = _plus_long_prefixe(reste, vocab)
            if atome is None:
                vocab.add(reste)
                break
            reste = reste[len(atome):].removeprefix(", ")
    return vocab


def _plus_long_prefixe(texte, vocab):
    candidats = [a for a in vocab if texte == a or texte.startswith(a + ", ")]
    return max(candidats, key=len) if candidats else None


def split_domaines(valeur):
    """"Agriculture, Water" -> ["Agriculture", "Water"]."""
    if not valeur:
        return []
    vocab = _vocabulaire_domaines()
    out, reste = [], valeur
    while reste:
        atome = _plus_long_prefixe(reste, vocab)
        if atome is None:
            out.append(reste)
            break
        out.append(atome)
        reste = reste[len(atome):].removeprefix(", ")
    return out


def _vocabulaire_domaines():
    global _DOMAINES
    if _DOMAINES is None:
        _, _, meta = load_all()
        valeurs = {str(v) for v in meta["domaine"].dropna().unique() if str(v).strip()}
        _DOMAINES = _build_vocabulaire(valeurs)
    return _DOMAINES


def list_domaines():
    """Liste triée des domaines atomiques réellement présents dans le corpus."""
    return sorted(_vocabulaire_domaines(), key=str.casefold)


# --- recherche ------------------------------------------------------------

def embed_query(query, vec):
    X_char = vec["char_vectorizer"].transform([query])
    X_word = vec["word_vectorizer"].transform([query])
    X = hstack([X_char * 0.5, X_word * 1.0]).tocsr()
    dense = vec["svd"].transform(X).astype(np.float32)
    norm = np.linalg.norm(dense, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    return dense / norm


# --- les backends ---------------------------------------------------------
# entrees   : n-grammes de caractères sur les EN-TÊTES seuls. Pour retrouver
#             un mot mooré, y compris mal orthographié.
# tfidf     : l'index d'origine (build_index.py) : en-tête ET définitions
#             mélangés, réduits par SVD.
# semantique: embeddings multilingues pré-entraînés sur les définitions fr/en.
#             Comprend le sens ; ne connaît pas le mooré.
# hybride   : fusion RRF de tfidf et semantique.
# auto      : route selon la langue de la requête. C'est le défaut.
#
# Mesuré sur le corpus (python3 benchmark.py) :
#
#                    expression depuis        entrée mooré avec
#                    sa définition fr         une faute de frappe
#                    top-1     top-5          top-1     top-5
#   entrees           0,7 %     2,9 %        59,1 %    77,0 %
#   tfidf            52,3 %    77,8 %         6,4 %    16,6 %
#   semantique       73,9 %    90,8 %         0,0 %     0,0 %
#   hybride          77,5 %    94,8 %         5,1 %    13,2 %
#   auto             77,5 %    94,8 %        57,4 %    74,9 %
#
# Chaque backend est excellent d'un côté et nul de l'autre : le sémantique
# tombe à 0 % en mooré (le modèle ne connaît pas la langue), et l'index des
# en-têtes à 0,7 % en français (il ne regarde pas les définitions). Aucun ne
# peut servir de défaut. Seul le routage obtient les deux colonnes.
#
# L'écart de « auto » à « entrees » en mooré (57,4 contre 59,1) est le coût
# des requêtes mal aiguillées : un mot déformé ressemble parfois assez à du
# français pour partir du mauvais côté.
BACKENDS = ("tfidf", "entrees", "semantique", "hybride", "auto")

_INDEX_ST = None
_VOCAB_LANGUE = None

# Caractères propres à l'orthographe mooré : leur présence tranche seule.
_CARACTERES_MOORE = set("ãẽĩõũɛɩʋŋɲ")


def _index_st():
    global _INDEX_ST
    if _INDEX_ST is None:
        _INDEX_ST = faiss.read_index(embeddings.INDEX_ST_PATH)
    return _INDEX_ST


def _mots(texte):
    return re.findall(r"[^\W\d_]+", str(texte).lower(), flags=re.UNICODE)


def _vocabulaires_langue():
    """Les mots du corpus, séparés en « côté mooré » et « côté fr/en »."""
    global _VOCAB_LANGUE
    if _VOCAB_LANGUE is None:
        _, _, meta = load_all()
        moore, autre = set(), set()
        for col in ("mot_moore", "variante", "pluriel"):
            if col in meta.columns:
                for v in meta[col].dropna():
                    moore.update(_mots(v))
        for col in ("definition_francais", "definition_english"):
            for v in meta[col].dropna():
                autre.update(_mots(v))
        _VOCAB_LANGUE = (moore, autre)
    return _VOCAB_LANGUE


def detecter_langue(query):
    """« moore » ou « fr_en », d'après le vocabulaire du corpus lui-même.

    Pas de bibliothèque de détection : sur des requêtes d'un ou deux mots
    elles sont peu fiables, et le corpus donne directement la réponse.
    """
    if _CARACTERES_MOORE & set(str(query).lower()):
        return "moore"
    mots = _mots(query)
    if not mots:
        return "fr_en"
    moore, autre = _vocabulaires_langue()
    n_moore = sum(1 for m in mots if m in moore)
    n_autre = sum(1 for m in mots if m in autre)
    if n_moore == 0 and n_autre == 0:
        # Mot inconnu des deux côtés : souvent une entrée mooré mal
        # orthographiée. Le modèle d'embeddings ne peut rien pour elle — il ne
        # connaît pas la langue — alors que les n-grammes de caractères, si.
        return "moore"
    # À égalité on penche vers fr/en : un mot présent des deux côtés est plus
    # souvent un mot français court qu'une entrée mooré.
    return "moore" if n_moore > n_autre else "fr_en"


def backend_par_defaut():
    """Toujours « auto » : il s'adapte à ce qui est installé.

    « entrees » ne demande aucune dépendance supplémentaire, et c'est lui qui
    apporte le plus gros gain (77,0 % contre 16,6 % en top-5 sur une faute de
    frappe). Retomber sur « tfidf » quand le modèle est absent priverait de ce
    gain exactement les installations minimales, qui sont le cas courant.
    """
    return "auto"


def resoudre_backend(backend, query):
    """Remplace « auto » par le backend adapté à la requête ET à l'installé."""
    if backend != "auto":
        return backend
    if detecter_langue(query) == "moore":
        return "entrees"
    return "hybride" if embeddings.index_disponible() else "tfidf"


def _rangs_tfidf(query, n):
    index, vec, _ = load_all()
    q = embed_query(query, vec).astype(np.float32)
    scores, idxs = index.search(q, min(n, index.ntotal))
    return [(int(i), float(s)) for s, i in zip(scores[0], idxs[0]) if i != -1]


def _rangs_semantique(query, n):
    index = _index_st()
    q = embeddings.encoder([query])
    scores, idxs = index.search(q, min(n, index.ntotal))
    return [(int(i), float(s)) for s, i in zip(scores[0], idxs[0]) if i != -1]


_INDEX_ENTREES = None


def _index_entrees():
    """Index n-grammes de caractères sur les EN-TÊTES seuls.

    build_index.py indexe un document mélangé (mot mooré + définitions fr/en
    + variante + domaine). Pour chercher par le sens c'est ce qu'on veut, mais
    pour retrouver un mot mooré mal orthographié c'est fatal : dans
    « maande | gombo | okra », l'en-tête ne pèse qu'un tiers du document, et
    une requête comme « maade » n'a presque rien à quoi s'accrocher.

    Mesuré sur 235 fautes de frappe (python3 benchmark.py) :

        document mélangé   top-1  6,4 %   top-5 16,6 %
        en-têtes seuls     top-1 59,1 %   top-5 77,0 %

    Construit à la volée (~1 s) : pas de fichier supplémentaire à versionner.
    """
    global _INDEX_ENTREES
    if _INDEX_ENTREES is None:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.preprocessing import normalize

        _, _, meta = load_all()
        entetes = [nettoyer_mot_moore(_clean(meta.iloc[i]["mot_moore"]))
                   for i in range(len(meta))]
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
        matrice = normalize(vec.fit_transform(entetes))
        _INDEX_ENTREES = (vec, matrice, normalize)
    return _INDEX_ENTREES


def _rangs_entrees(query, n):
    vec, matrice, normalize = _index_entrees()
    v = normalize(vec.transform([str(query)]))
    scores = (matrice @ v.T).toarray().ravel()
    n = max(1, min(n, scores.shape[0]))
    if n < scores.shape[0]:
        candidats = np.argpartition(-scores, n - 1)[:n]
    else:
        candidats = np.arange(scores.shape[0])
    candidats = candidats[np.argsort(-scores[candidats])]
    return [(int(i), float(scores[i])) for i in candidats if scores[i] > 0]


def _fusion_rrf(listes, k_rrf=60):
    """Reciprocal Rank Fusion : fusionne des classements, pas des scores.

    Les deux index ne produisent pas des scores comparables — un cosinus
    TF-IDF/LSA et un cosinus d'embeddings ne vivent pas sur la même échelle,
    et les normaliser demanderait de les calibrer. La RRF ne regarde que le
    RANG : un document vaut 1/(k + rang) dans chaque liste, et on somme. C'est
    la méthode standard en recherche hybride, et elle n'a rien à régler.
    """
    cumul = {}
    for liste in listes:
        for rang, (idx, _) in enumerate(liste, start=1):
            cumul[idx] = cumul.get(idx, 0.0) + 1.0 / (k_rrf + rang)
    return sorted(cumul.items(), key=lambda kv: kv[1], reverse=True)


_ENTREES_EXACTES = None


def _table_entrees():
    """En-tête normalisé -> lignes du corpus, pour la correspondance exacte."""
    global _ENTREES_EXACTES
    if _ENTREES_EXACTES is None:
        _, _, meta = load_all()
        table = {}
        for pos in range(len(meta)):
            mot = nettoyer_mot_moore(_clean(meta.iloc[pos]["mot_moore"]))
            if not mot:
                continue
            cle = mot.lower().strip()
            table.setdefault(cle, []).append(pos)
            # Variante sans diacritiques : l'utilisateur n'a pas toujours ã ou ʋ
            # sur son clavier (le clavier virtuel de la page existe pour ça,
            # mais on ne peut pas l'exiger).
            sans = "".join(c for c in unicodedata.normalize("NFD", cle)
                           if unicodedata.category(c) != "Mn")
            if sans != cle:
                table.setdefault(sans, []).append(pos)
        _ENTREES_EXACTES = table
    return _ENTREES_EXACTES


def positions_exactes(query):
    """Lignes dont l'en-tête EST la requête.

    Ni le TF-IDF ni les embeddings ne garantissent qu'une correspondance
    exacte sorte en tête : les vecteurs sont compressés par SVD, et une entrée
    courte comme « koom » se fait dépasser par « rʋʋd-koom » ou
    « koom gʋls moodo », dont le document est plus riche. Or chercher un mot
    du dictionnaire et ne pas l'obtenir en premier est le défaut le plus
    visible qu'un moteur de dictionnaire puisse avoir.
    """
    cle = str(query).lower().strip()
    table = _table_entrees()
    if cle in table:
        return table[cle]
    sans = "".join(c for c in unicodedata.normalize("NFD", cle)
                   if unicodedata.category(c) != "Mn")
    return table.get(sans, [])


def classement(query, n, backend):
    """[(ligne du corpus, score), ...] pour un backend, sans mise en forme.

    Utile quand on a besoin des positions et des scores bruts plutôt que des
    résultats formatés — par exemple pour croiser deux backends (cf. les
    tournures idiomatiques dans translate.py).
    """
    if backend == "tfidf":
        return _rangs_tfidf(query, n)
    if backend == "entrees":
        return _rangs_entrees(query, n)
    if backend == "semantique":
        return _rangs_semantique(query, n)
    if backend == "hybride":
        return _fusion_rrf([_rangs_tfidf(query, n), _rangs_semantique(query, n)])
    if backend == "auto":
        return classement(query, n, resoudre_backend("auto", query))
    raise ValueError(f"backend inconnu : {backend} (attendu : {BACKENDS})")


def search(query, k=5, domaine=None, backend=None):
    index, _, meta = load_all()
    # FAISS lève une AssertionError sur k <= 0, et ne peut pas rendre plus de
    # voisins qu'il n'y a de vecteurs.
    k = max(1, min(int(k), index.ntotal))

    backend = backend or "tfidf"
    if backend not in BACKENDS:
        raise ValueError(f"backend inconnu : {backend} (attendu : {BACKENDS})")
    if backend in ("semantique", "hybride") and not embeddings.index_disponible():
        raise RuntimeError(
            "index sémantique absent — lancez : python3 build_st_index.py")
    effectif = resoudre_backend(backend, query)
    langue = detecter_langue(query)

    # Avec un filtre par domaine il faut balayer tout l'index : filtrer
    # seulement les k premiers résultats afficherait souvent 0 carte alors que
    # le corpus contient des entrées pertinentes.
    fetch_k = index.ntotal if domaine else k

    if effectif == "tfidf":
        classement_ = _rangs_tfidf(query, fetch_k)
    elif effectif == "entrees":
        classement_ = _rangs_entrees(query, fetch_k)
    elif effectif == "semantique":
        classement_ = _rangs_semantique(query, fetch_k)
    else:
        # Chaque moteur doit proposer assez de candidats pour que la fusion
        # ait de quoi travailler, sans balayer tout l'index pour rien.
        profondeur = index.ntotal if domaine else max(fetch_k * 5, 50)
        classement_ = _fusion_rrf([_rangs_tfidf(query, profondeur),
                                   _rangs_semantique(query, profondeur)])

    # Les correspondances exactes passent devant, quel que soit le backend.
    exacts = positions_exactes(query)
    if exacts:
        deja = set(exacts)
        classement_ = ([(p, 1.0) for p in exacts]
                       + [(i, sc) for i, sc in classement_ if i not in deja])

    results = []
    for idx, score in classement_:
        row = meta.iloc[idx]
        dom = _clean(row.get("domaine", ""))
        if domaine and domaine not in split_domaines(dom):
            continue
        mot = nettoyer_mot_moore(_clean(row["mot_moore"]))
        syn, ant = renvois(row.get("synonymes", ""), row.get("antonymes", ""))
        results.append({
            "mot_moore": mot,
            "definition_francais": marquer_tronquee(_clean(row["definition_francais"])),
            "definition_english": marquer_tronquee(_clean(row["definition_english"])),
            "categorie": _clean(row["categorie_grammaticale"]),
            "prononciation": _clean(row.get("prononciation", "")),
            "domaine": dom,
            # Colonnes présentes dans le corpus mais qui n'étaient pas
            # exploitées : 2 600 entrées ont des synonymes, 3 209 un pluriel.
            "synonymes": syn,
            "antonymes": ant,
            "variantes": _liste(row.get("variante", "")),
            "pluriel": _clean(row.get("pluriel", "")),
            # 947 mots ont plusieurs sens ; on signale combien, pour que
            # l'interface propose de les afficher.
            "autres_sens": max(0, len(_table_entrees().get(mot.lower(), [])) - 1),
            "score": float(score),
            "backend": effectif,
            "langue_detectee": langue,
        })
        if len(results) >= k:
            break
    return results


if __name__ == "__main__":
    args = sys.argv[1:]
    domaine = None
    if "--domaine" in args:
        i = args.index("--domaine")
        domaine = args[i + 1] if i + 1 < len(args) else None
        del args[i:i + 2]

    backend = None
    if "--backend" in args:
        i = args.index("--backend")
        backend = args[i + 1] if i + 1 < len(args) else None
        del args[i:i + 2]

    query = " ".join(args) or "eau"
    backend = backend or backend_par_defaut()
    filtre = f" [domaine : {domaine}]" if domaine else ""
    filtre += f" [backend : {backend}]"
    print(f"Requête : « {query} »{filtre}\n")

    resultats = search(query, domaine=domaine, backend=backend)
    if not resultats:
        print("  (aucun résultat)")
    for r in resultats:
        print(f"  {r['score']:.3f}  {r['mot_moore']:20s} "
              f"({r['categorie']})  fr: {r['definition_francais'][:60]}")
