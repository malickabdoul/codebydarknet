"""
translate.py
-------------
Glossage français -> mooré au niveau de la PHRASE, par-dessus la base
vectorielle.

Pourquoi ce module existe
-------------------------
search.py encode toute la requête en un seul vecteur et rend l'entrée la plus
proche. Sur un mot isolé c'est ce qu'on veut ; sur une phrase c'est absurde :
« je veux boire de l'eau » renvoyait « tas de résidus de minerai de fer ».

Ici on fait l'inverse : on découpe la phrase, et on cherche pour chaque
morceau. Le point clé est de chercher les EXPRESSIONS d'abord (le corpus en
contient 504, et 5 753 gloses françaises font plus d'un mot) : « n'importe
comment » doit sortir « a bal », pas « n'importe » + « comment ».

Ce que ce module ne fait PAS
----------------------------
Ce n'est pas de la traduction. Le résultat est une GLOSE : le meilleur
équivalent mooré de chaque segment, dans l'ordre du français. Le mooré a son
propre ordre des mots, ses postpositions et ses marques d'aspect, qu'un
dictionnaire ne permet pas de reconstruire. La glose est un support pour un
locuteur, pas une phrase à publier telle quelle.
"""

import re
import unicodedata

import embeddings
from search import classement as recherche_classement
from search import load_all, nettoyer_mot_moore

# Longueur maxi d'une expression cherchée d'un bloc (en mots).
MAX_NGRAM = 6

# Mots-outils français dont l'équivalent mooré, s'il existe, est forcément un
# mot grammatical. Sans ce garde-fou le corpus renvoie des homographes :
# « pas » -> kẽndre (le nom « enjambée »), « est » -> yaanga (le point
# cardinal Est).
CATEGORIES_GRAMMATICALES = {
    "conjonction", "postposition", "auxiliaire", "pronom",
    "interrogatif", "adverbe", "indéfinie", "numéral", "interj",
}
MOTS_OUTILS = {
    "et", "ou", "dans", "sur", "avec", "pour", "que", "qui", "quoi", "dont",
    "ne", "pas", "plus", "est", "sont", "sous", "vers", "chez", "sans",
    "mais", "donc", "car", "si", "quand", "comme",
}

# Articles et prépositions sans équivalent isolé en mooré (le mooré n'a pas
# d'articles, et les rapports sont rendus par postposition ou par simple
# juxtaposition). Les chercher produit des faux amis : « la » -> ka, qui est
# la négation.
IGNORES = {
    "le", "la", "les", "l", "un", "une", "des", "du", "de", "d",
    "au", "aux", "a", "en", "y", "ce", "cet", "cette", "ces",
}

# Mots qui, en tête d'un fragment de définition, signalent une suite de phrase
# plutôt qu'une glose autonome (cf. _glose_valide).
_DEBUTS_DE_FRAGMENT = IGNORES | {
    "à", "ou", "et", "pour", "avec", "dans", "sur", "par", "sans", "sous",
    "comme", "qui", "que", "dont", "où", "vers", "chez", "selon", "entre",
    "son", "sa", "ses", "leur", "leurs", "mon", "ma", "mes", "notre", "votre",
}

# Formes irrégulières fréquentes : le découpage par suffixe ne peut pas les
# retrouver (veux -> vouloir).
IRREGULIERS = {
    "suis": "etre", "es": "etre", "est": "etre", "sommes": "etre",
    "etes": "etre", "sont": "etre", "etait": "etre", "etaient": "etre",
    "ai": "avoir", "as": "avoir", "avons": "avoir", "avez": "avoir",
    "ont": "avoir", "avait": "avoir",
    "vais": "aller", "vas": "aller", "va": "aller", "allons": "aller",
    "allez": "aller", "vont": "aller",
    "veux": "vouloir", "veut": "vouloir", "voulons": "vouloir",
    "voulez": "vouloir", "veulent": "vouloir",
    "peux": "pouvoir", "peut": "pouvoir", "pouvons": "pouvoir",
    "pouvez": "pouvoir", "peuvent": "pouvoir",
    "fais": "faire", "fait": "faire", "faisons": "faire", "faites": "faire",
    "font": "faire",
    "viens": "venir", "vient": "venir", "venons": "venir", "venez": "venir",
    "viennent": "venir",
    "dis": "dire", "dit": "dire", "disons": "dire", "dites": "dire",
    "disent": "dire",
    "vois": "voir", "voit": "voir", "voyons": "voir", "voyez": "voir",
    "voient": "voir",
    "sais": "savoir", "sait": "savoir", "savons": "savoir",
    "savez": "savoir", "savent": "savoir",
    "bois": "boire", "boit": "boire", "buvons": "boire", "buvez": "boire",
    "boivent": "boire",
    "prends": "prendre", "prend": "prendre", "prenons": "prendre",
    "mets": "mettre", "met": "mettre",
}

# Élisions : « l'eau » doit devenir « l » + « eau ».
_ELISION = re.compile(r"\b(qu|[cdjlmnst])['’]", re.IGNORECASE)

# Catégories des tournures toutes faites. Une glose mot-à-mot ne peut pas les
# produire : « tu vas bien » se dit « laafɩ bala » (littéralement « ça va »),
# pas « f kɩbe neere » (« tu » + « aller » + « bien »).
CATEGORIES_IDIOMATIQUES = {"expression", "interj", "exclamation"}

# Seuil de proximité en dessous duquel une suggestion n'est plus significative.
# Il dépend du backend : un cosinus TF-IDF/LSA et un cosinus d'embeddings ne se
# distribuent pas pareil, et la fusion RRF ne produit pas un cosinus du tout
# (ses scores valent ~1/60 par liste). Un seuil unique n'aurait aucun sens.
SEUILS_IDIOME = {
    "tfidf": 0.45,        # « laafɩ bala » pour « tu vas bien » sort à 0,64
    "semantique": 0.55,   # bruit mesuré entre 0,03 et 0,35 ; bonne réponse 0,65
}
PROFONDEUR_IDIOMES = 200


def backend_idiomes():
    """Backend utilisé pour les tournures idiomatiques.

    Chercher une tournure à partir d'une phrase française est une tâche
    purement sémantique : on prend les embeddings dès qu'ils sont là. Pas la
    fusion hybride, dont les scores RRF ne sont pas interprétables comme une
    proximité de sens — or c'est ce score qu'on affiche à l'utilisateur.
    """
    return "semantique" if embeddings.index_disponible() else "tfidf"

_CACHE = None


def normaliser_accents(texte):
    """Minuscules, ponctuation neutralisée, élisions coupées, accents GARDÉS.

    Les accents distinguent des sens différents : « marché » (raaga, le lieu)
    et « marche » (kẽnde, l'action). On les conserve pour tenter d'abord une
    correspondance exacte.
    """
    texte = _ELISION.sub(r"\1 ", str(texte)).lower()
    # Les ligatures ne sont pas dans la plage latin-1 : sans ce remplacement
    # elles sont effacées comme de la ponctuation, et « œil » devient « il ».
    texte = texte.replace("œ", "oe").replace("æ", "ae")
    texte = re.sub(r"[^0-9a-zà-öø-ÿ' ]", " ", texte)
    return re.sub(r"\s+", " ", texte).strip()


def sans_accents(texte):
    texte = unicodedata.normalize("NFD", str(texte))
    return "".join(c for c in texte if unicodedata.category(c) != "Mn")


def normaliser(texte):
    """Comme normaliser_accents, mais accents retirés (recherche tolérante)."""
    return sans_accents(normaliser_accents(texte))


def _index_gloses():
    """Index inverse français -> entrées du corpus.

    Une définition comme « manger, consommer » donne deux gloses distinctes :
    on veut retrouver l'entrée aussi bien par « manger » que par « consommer ».
    """
    global _CACHE
    if _CACHE is not None:
        return _CACHE

    _, _, meta = load_all()
    gloses, gloses_acc = {}, {}
    for pos in range(len(meta)):
        fr = meta.iloc[pos]["definition_francais"]
        if fr is None or str(fr) == "nan":
            continue
        for rang, morceau in enumerate(str(fr).split(",")):
            cle_acc = normaliser_accents(morceau)
            if not cle_acc or not _glose_valide(cle_acc, rang):
                continue
            gloses_acc.setdefault(cle_acc, []).append(pos)
            gloses.setdefault(sans_accents(cle_acc), []).append(pos)
    _CACHE = (gloses, gloses_acc, meta)
    return _CACHE


def _glose_valide(cle, rang):
    """Un fragment après virgule est-il une vraie glose, ou un bout de phrase ?

    Découper les définitions sur les virgules donne les synonymes (« manger,
    consommer » -> deux gloses utiles), mais découpe aussi les définitions
    descriptives en fragments qui n'en sont pas : « dispositif pour sécher ou
    griller la viande, le poisson » produisait la glose « le poisson », et la
    phrase « il a mangé le poisson » y tombait.

    Le premier fragment est toujours une glose. Les suivants ne le sont que
    s'ils ne commencent pas par un mot de liaison — un fragment qui démarre
    par « le », « pour » ou « ou » continue la phrase précédente.
    """
    return rang == 0 or cle.split()[0] not in _DEBUTS_DE_FRAGMENT


def _lemmes_candidats(token):
    """Formes à essayer pour un token non trouvé tel quel."""
    essais = [token]
    if token in IRREGULIERS:
        essais.append(IRREGULIERS[token])
    n = len(token)
    # pluriel / féminin
    if n > 3 and token[-1] in "sx":
        essais.append(token[:-1])
    if n > 4 and token.endswith("es"):
        essais.append(token[:-2])
    # 1er groupe : mange / mangez / mangé -> manger
    for suf in ("erions", "eraient", "eront", "erez", "erons", "eras", "erai",
                "aient", "ait", "ais", "ent", "ons", "ez", "es", "as", "e", "a"):
        if token.endswith(suf) and n - len(suf) >= 3:
            essais.append(token[: n - len(suf)] + "er")
    # 2e groupe : finis -> finir
    for suf in ("issent", "issons", "issez", "it", "is"):
        if token.endswith(suf) and n - len(suf) >= 3:
            essais.append(token[: n - len(suf)] + "ir")

    uniques = []
    for forme in essais:
        if forme and forme not in uniques:
            uniques.append(forme)
    return uniques


def _candidats(positions, meta, limite, cle, filtrer_grammatical=False,
               categorie_attendue=None):
    """Candidats pour une glose, les plus fiables d'abord.

    Deux critères, dans cet ordre :

    1. La catégorie attendue. Quand on est arrivé ici en lemmatisant une forme
       conjuguée (« vais » -> « aller »), on cherche un verbe : sans ça le
       corpus rend kẽnde, qui est le NOM « marche, aller, voyage ».
    2. Une définition qui vaut exactement la glose cherchée, plutôt qu'une
       entrée où la glose n'est qu'un sens parmi d'autres : pour « boire »,
       yũ (« boire ») passe devant fõbge (« boire, vider d'un trait »).

    Ce sont des préférences, pas des filtres : les autres candidats restent
    proposés en second. Départager les synonymes restants demande un locuteur.
    """
    retenus = []
    for rang, pos in enumerate(positions):
        row = meta.iloc[pos]
        cat = str(row["categorie_grammaticale"] or "")
        if filtrer_grammatical and cat.lower() not in CATEGORIES_GRAMMATICALES:
            continue
        definition = str(row["definition_francais"])
        retenus.append((
            0 if categorie_attendue and cat.lower() == categorie_attendue else 1,
            0 if normaliser(definition) == cle else 1,
            rang,
            {
                "mot_moore": nettoyer_mot_moore(row["mot_moore"]),
                "definition_francais": definition,
                "categorie": cat,
            },
        ))
    retenus.sort(key=lambda x: x[:3])
    return [c for *_, c in retenus[:limite]]


def segmenter(phrase, max_candidats=3):
    """Découpe la phrase en cherchant les plus longues expressions d'abord."""
    gloses, gloses_acc, meta = _index_gloses()
    # Deux vues des mêmes tokens : avec accents (correspondance stricte, qui
    # sépare « marché » de « marche ») et sans (rattrapage tolérant).
    tokens_acc = normaliser_accents(phrase).split()
    tokens = [sans_accents(t) for t in tokens_acc]
    segments = []
    i = 0

    while i < len(tokens):
        # 1. la plus longue suite de mots qui est une glose connue
        trouve = False
        for n in range(min(MAX_NGRAM, len(tokens) - i), 1, -1):
            span_acc = " ".join(tokens_acc[i:i + n])
            span = " ".join(tokens[i:i + n])
            if span_acc in gloses_acc:
                positions, cle = gloses_acc[span_acc], span_acc
            elif span in gloses:
                positions, cle = gloses[span], span
            else:
                continue
            segments.append({
                "source": span_acc,
                "type": "expression",
                "n_mots": n,
                "candidats": _candidats(positions, meta, max_candidats, cle),
                "confiance": "exacte",
            })
            i += n
            trouve = True
            break
        if trouve:
            continue

        # 2. mot seul
        tok, tok_acc = tokens[i], tokens_acc[i]
        i += 1

        if tok in IGNORES:
            segments.append({
                "source": tok_acc, "type": "ignore", "n_mots": 1,
                "candidats": [], "confiance": "aucune",
                "note": "article ou préposition : pas d'équivalent isolé en mooré",
            })
            continue

        grammatical = tok in MOTS_OUTILS
        # La forme accentuée telle qu'écrite d'abord, puis les lemmes.
        essais = [(tok_acc, gloses_acc, "exacte")]
        essais += [(f, gloses, "exacte" if f == tok else "lemmatisé")
                   for f in _lemmes_candidats(tok)]

        for forme, index, confiance in essais:
            if forme not in index:
                continue
            # Si on a dû lemmatiser vers un infinitif, c'est un verbe qu'on veut.
            attendue = ("verbe" if confiance == "lemmatisé"
                        and forme.endswith(("er", "ir", "re")) else None)
            cands = _candidats(index[forme], meta, max_candidats, forme,
                               filtrer_grammatical=grammatical,
                               categorie_attendue=attendue)
            if not cands:
                continue
            segment = {
                "source": tok_acc, "type": "mot", "n_mots": 1,
                "candidats": cands, "confiance": confiance,
            }
            if confiance == "lemmatisé":
                segment["lemme"] = forme
            segments.append(segment)
            break
        else:
            segments.append({
                "source": tok_acc, "type": "inconnu", "n_mots": 1,
                "candidats": [], "confiance": "aucune",
                "note": "absent du corpus",
            })

    return segments


def suggestions_idiomatiques(phrase, k=3, backend=None, seuil=None):
    """Tournures toutes faites proches du SENS de la phrase entière.

    La segmentation ne peut pas trouver ça : « tu vas bien » se dit
    « laafɩ bala » (« ça va »), et aucun mot ne coïncide.

    Chaque index seul échoue d'une manière différente, mesurée sur le corpus :

    - **TF-IDF** classe bien (« laafɩ bala » au rang 1) mais propose du bruit
      sur les phrases descriptives : « la maison de mon père » lui inspire
      « Zezi kũum teegre » à 0,72.
    - **Embeddings** ne proposent rien sur ces mêmes phrases — leur cosinus
      pour ce bruit tombe entre 0,03 et 0,35 — mais ils classent mal : les
      définitions d'un ou deux mots (« woo » = « d'accord ») obtiennent des
      scores élevés un peu partout, et « laafɩ bala » ne sort qu'au rang 3.

    D'où la combinaison : **on ordonne par fusion RRF** (les deux index doivent
    s'accorder, ce qui remet « laafɩ bala » au rang 1) et **on filtre par le
    cosinus sémantique** (qui élimine le bruit lexical). Le score affiché est
    ce cosinus : contrairement au score RRF, il veut dire quelque chose.

    Calibré sur une poignée de phrases, faute de jeu de test annoté : à
    revoir si vous en constituez un.
    """
    if len(normaliser(phrase).split()) < 2:
        return []

    _, _, meta = load_all()
    semantique_ok = embeddings.index_disponible()
    backend = backend or ("hybride" if semantique_ok else "tfidf")

    if seuil is None:
        seuil = SEUILS_IDIOME["semantique" if semantique_ok else "tfidf"]

    ordre = recherche_classement(phrase, PROFONDEUR_IDIOMES, backend)
    # Le veto : la proximité de sens, indépendamment du classement.
    cosinus = (dict(recherche_classement(phrase, PROFONDEUR_IDIOMES, "semantique"))
               if semantique_ok else None)

    trouves = []
    for pos, score_ordre in ordre:
        row = meta.iloc[pos]
        if str(row["categorie_grammaticale"] or "").lower() not in CATEGORIES_IDIOMATIQUES:
            continue
        definition = str(row["definition_francais"])
        # « Category: » est un résidu du scraping de Webonary : la ligne a ses
        # colonnes décalées et son champ mot_moore ne contient pas du mooré.
        if "Category:" in definition:
            continue

        score = cosinus.get(pos) if cosinus is not None else score_ordre
        if score is None or score < seuil:
            continue

        trouves.append({
            "mot_moore": nettoyer_mot_moore(str(row["mot_moore"])),
            "definition_francais": definition,
            "categorie": str(row["categorie_grammaticale"] or ""),
            "score": round(float(score), 3),
            "backend": backend,
        })
        if len(trouves) >= k:
            break
    return trouves


def traduire(phrase, max_candidats=3):
    segments = segmenter(phrase, max_candidats=max_candidats)
    utiles = [s for s in segments if s["type"] != "ignore"]
    resolus = [s for s in utiles if s["candidats"]]

    glose = " ".join(s["candidats"][0]["mot_moore"]
                     for s in segments if s["candidats"])
    couverture = len(resolus) / len(utiles) if utiles else 0.0

    return {
        "phrase": phrase,
        "segments": segments,
        "glose": glose,
        "couverture": round(couverture, 3),
        "n_expressions": sum(1 for s in segments if s["type"] == "expression"),
        "non_resolus": [s["source"] for s in utiles if not s["candidats"]],
        "idiomes": suggestions_idiomatiques(phrase),
        "avertissement": (
            "Glose mot-à-mot alignée sur le français, pas une phrase mooré "
            "grammaticale : l'ordre des mots, les postpositions et les marques "
            "d'aspect du mooré ne sont pas reconstruits."
        ),
    }


if __name__ == "__main__":
    import sys

    MARQUES = {"expression": "EXPR", "mot": "mot ", "ignore": "----",
               "inconnu": "????"}

    phrase = " ".join(sys.argv[1:]) or "je veux boire de l'eau"
    res = traduire(phrase)
    print(f"Phrase : « {res['phrase']} »")
    print(f"Couverture : {res['couverture'] * 100:.0f} %"
          f"  ({res['n_expressions']} expression(s) reconnue(s))\n")
    for s in res["segments"]:
        tete = s["candidats"][0]["mot_moore"] if s["candidats"] else "—"
        lemme = f"   (via « {s['lemme']} »)" if s.get("lemme") else ""
        print(f"  [{MARQUES[s['type']]}] {s['source']:22s} -> {tete}{lemme}")
        for c in s["candidats"][1:]:
            print(f"         {'':22s}    ou {c['mot_moore']}")
    print(f"\nGlose : {res['glose']}")

    if res["idiomes"]:
        print("\nTournures idiomatiques proches (base vectorielle) :")
        for i in res["idiomes"]:
            print(f"  {i['score']:.3f}  {i['mot_moore']:20s} "
                  f"({i['categorie']})  {i['definition_francais'][:50]}")

    print(f"\n! {res['avertissement']}")
