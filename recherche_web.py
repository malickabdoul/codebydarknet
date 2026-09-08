"""
recherche_web.py
-----------------
Recherche d'informations sur le Web à partir d'une requête en mooré, en
français ou en anglais.

C'est ce que demande le cahier des charges (EF07, EF08, EF10, EF11, EF12) :
chercher sur le Web, récupérer les résultats, afficher pour chacun son titre
et un extrait, et fournir un lien vers la page.

Le rôle du corpus
-----------------
Le dictionnaire n'est pas la destination, c'est le **pont**. « koom » ne donne
rien sur un moteur francophone ; le corpus le traduit en « eau », et c'est ce
terme-là qu'on envoie au Web. Inversement, une requête française est traduite
en mooré pour interroger les sources en mooré. C'est ce qui rend le moteur
multilingue plutôt que littéral.

    requête mooré  ->  corpus  ->  terme français
                                        |
                                   sources Web
                                        |
        titre + extrait + lien  <-  reclassement sémantique

Les sources
-----------
Trois encyclopédies MediaWiki, sans clé d'API ni quota : Wikipédia en mooré,
Wikipédia en français et le Wiktionnaire. Elles renvoient exactement les trois
champs exigés par EF10 et EF11 (titre, extrait, lien).

L'architecture est enfichable : ajouter un moteur généraliste (Bing, Brave,
Google Custom Search) revient à ajouter une entrée dans SOURCES et une
fonction d'interrogation. Ces moteurs demandent une clé d'API, raison pour
laquelle ils ne sont pas activés par défaut.
"""

import html
import json
import re
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import embeddings
import search

# Identification honnête : les API MediaWiki demandent un User-Agent explicite.
USER_AGENT = "MooreSearch/1.0 (projet universitaire IBAM/UJKZ ; moteur de recherche mooré)"
DELAI = 12  # secondes

SOURCES = [
    {
        "id": "mos.wikipedia",
        "nom": "Wikipédia en mooré",
        "api": "https://mos.wikipedia.org/w/api.php",
        "page": "https://mos.wikipedia.org/wiki/",
        "langue": "moore",
    },
    {
        "id": "fr.wikipedia",
        "nom": "Wikipédia",
        "api": "https://fr.wikipedia.org/w/api.php",
        "page": "https://fr.wikipedia.org/wiki/",
        "langue": "fr",
    },
    {
        "id": "fr.wiktionary",
        "nom": "Wiktionnaire",
        "api": "https://fr.wiktionary.org/w/api.php",
        "page": "https://fr.wiktionary.org/wiki/",
        "langue": "fr",
    },
]

# Les réponses du Web sont lentes et se répètent beaucoup pendant une démo.
_CACHE = {}
_CACHE_MAX = 200


def _nettoyer(fragment):
    """MediaWiki entoure les mots trouvés de <span class="searchmatch">."""
    texte = re.sub(r"<[^>]+>", "", str(fragment))
    return re.sub(r"\s+", " ", html.unescape(texte)).strip()


def _interroger(source, terme, k):
    """Un appel à une API MediaWiki. Ne lève jamais : le Web est faillible."""
    if not terme:
        return []
    params = urllib.parse.urlencode({
        "action": "query",
        "list": "search",
        "srsearch": terme,
        "srlimit": k,
        "srprop": "snippet",
        "format": "json",
    })
    requete = urllib.request.Request(f"{source['api']}?{params}",
                                     headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(requete, timeout=DELAI) as reponse:
            donnees = json.loads(reponse.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError, TimeoutError):
        return []

    resultats = []
    for element in donnees.get("query", {}).get("search", []):
        titre = _nettoyer(element.get("title", ""))
        if not titre:
            continue
        resultats.append({
            "titre": titre,
            "extrait": _nettoyer(element.get("snippet", "")),
            "url": source["page"] + urllib.parse.quote(titre.replace(" ", "_")),
            "source": source["nom"],
            "source_id": source["id"],
            "terme_interroge": terme,
        })
    return resultats


# En dessous, la correspondance trouvée dans le corpus n'est pas fiable.
# Mesuré : un mot exact obtient 1,000, une faute de frappe plausible 0,551,
# et du charabia 0,256. Sans ce garde-fou, « zzzqxwv » était « traduit » en
# « presque totalité » et le Web était interrogé sur cette invention.
SEUIL_PONT = 0.45


def traduire_requete(requete):
    """Établit le pont : donne le terme mooré ET le terme français.

    Renvoie (terme_moore, terme_fr, explication).
    """
    requete = str(requete).strip()
    if not requete:
        return "", "", ""

    langue = search.detecter_langue(requete)

    if langue == "moore":
        # Le mot mooré sert tel quel côté mooré ; sa définition sert côté
        # français. C'est là que le corpus gagne son utilité.
        entrees = search.search(requete, k=1, backend="entrees")
        if (entrees and entrees[0]["definition_francais"]
                and entrees[0]["score"] >= SEUIL_PONT):
            # La première glose suffit : « eau fraîche, eau froide » -> « eau ».
            fr = entrees[0]["definition_francais"].split(",")[0].strip()
            fr = re.sub(r"\s*\([^)]*\)?\s*$", "", fr).strip()
            return requete, fr, f"« {requete} » traduit en « {fr} » par le corpus"
        return requete, requete, "terme absent du corpus : envoyé au Web tel quel"

    # Requête française ou anglaise : on cherche son équivalent mooré pour
    # interroger les sources en mooré.
    equivalents = search.search(requete, k=1, backend="hybride"
                                if embeddings.index_disponible() else "tfidf")
    if equivalents:
        moore = equivalents[0]["mot_moore"]
        return moore, requete, f"« {requete} » rapproché de « {moore} » par le corpus"
    return "", requete, ""


def _reclasser(requete_fr, resultats):
    """EF13 appliqué aux résultats Web : trier par proximité de sens.

    Les moteurs classent par correspondance de mots ; le modèle compare le
    SENS de la requête à celui de chaque extrait.

    Une précaution s'impose : le modèle ne connaît pas le mooré. Reclasser
    l'ensemble enterrait systématiquement les pages en mooré — sur « koom »,
    aucune ne subsistait dans les quinze premiers résultats. Dans un moteur
    destiné à des locuteurs du mooré, c'est un contresens. On ne reclasse donc
    que les pages en langue connue du modèle, et les pages en mooré gardent
    l'ordre de pertinence de leur propre source.
    """
    if not resultats or not embeddings.index_disponible() or not requete_fr:
        return resultats

    connues = [r for r in resultats if r["source_id"] != "mos.wikipedia"]
    moore = [r for r in resultats if r["source_id"] == "mos.wikipedia"]
    if not connues:
        return resultats

    try:
        textes = [f"{r['titre']}. {r['extrait']}" for r in connues]
        vecteurs = embeddings.encoder([requete_fr] + textes)
        scores = vecteurs[1:] @ vecteurs[0]
        for resultat, score in zip(connues, scores):
            resultat["score"] = round(float(score), 3)
            resultat["reclasse"] = True
        connues.sort(key=lambda r: r["score"], reverse=True)
    except Exception:
        return resultats

    return _entrelacer(moore, connues)


# Une page sur trois au maximum est réservée au mooré : assez pour que la
# langue du projet reste visible, pas au point d'évincer les résultats
# francophones quand le Wikipédia mooré (1 326 articles) n'a rien de pertinent.
PERIODE_MOORE = 3


def _entrelacer(moore, autres):
    """Intercale les pages en mooré parmi les autres, sans les noyer."""
    sortie, i, j = [], 0, 0
    while i < len(moore) or j < len(autres):
        if moore and len(sortie) % PERIODE_MOORE == 0 and i < len(moore):
            sortie.append(moore[i])
            i += 1
        elif j < len(autres):
            sortie.append(autres[j])
            j += 1
        elif i < len(moore):
            sortie.append(moore[i])
            i += 1
    return sortie


def rechercher_web(requete, k=8, reclasser=True):
    """Cherche sur le Web et renvoie titre, extrait et lien pour chaque page."""
    requete = str(requete).strip()
    if not requete:
        return {"requete": "", "resultats": [], "sources": [], "pont": ""}

    cle = (requete, k, bool(reclasser))
    if cle in _CACHE:
        return _CACHE[cle]

    moore, francais, explication = traduire_requete(requete)

    # Un appel par source, en parallèle : trois requêtes séquentielles
    # feraient attendre l'utilisateur pour rien.
    taches = []
    with ThreadPoolExecutor(max_workers=len(SOURCES)) as pool:
        for source in SOURCES:
            terme = moore if source["langue"] == "moore" else francais
            taches.append((source, pool.submit(_interroger, source, terme, k)))
        collecte = [(source, tache.result()) for source, tache in taches]

    resultats, vus, sources_utiles = [], set(), []
    for source, trouves in collecte:
        if trouves:
            sources_utiles.append(source["nom"])
        for resultat in trouves:
            if resultat["url"] in vus:
                continue
            vus.add(resultat["url"])
            resultats.append(resultat)

    if reclasser:
        resultats = _reclasser(francais, resultats)

    reponse = {
        "requete": requete,
        "terme_moore": moore,
        "terme_francais": francais,
        "pont": explication,
        "sources": sources_utiles,
        "reclasse_semantiquement": bool(resultats and resultats[0].get("reclasse")),
        "resultats": resultats[:k],
    }

    if len(_CACHE) >= _CACHE_MAX:
        _CACHE.clear()
    _CACHE[cle] = reponse
    return reponse


if __name__ == "__main__":
    import sys

    requete = " ".join(sys.argv[1:]) or "koom"
    reponse = rechercher_web(requete)
    print(f"Requête : « {reponse['requete']} »")
    if reponse["pont"]:
        print(f"Pont     : {reponse['pont']}")
    print(f"Sources  : {', '.join(reponse['sources']) or 'aucune'}")
    if reponse["reclasse_semantiquement"]:
        print("Résultats reclassés par proximité de sens.")
    print()
    for i, resultat in enumerate(reponse["resultats"], 1):
        score = f"  [{resultat['score']:.2f}]" if "score" in resultat else ""
        print(f"{i}. {resultat['titre']}{score}")
        print(f"   {resultat['extrait'][:100]}")
        print(f"   {resultat['url']}")
        print(f"   — {resultat['source']}")
        print()
