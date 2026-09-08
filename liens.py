"""
liens.py
---------
Liens de consultation externes pour un mot, affichés sous les résultats.

Toutes les URL ci-dessous ont été ouvertes dans un vrai navigateur avant
d'être retenues, avec un mot du corpus (« koom ») ET un mot qui n'existe
nulle part (« zzzqxwv »), pour vérifier qu'elles ne tombent pas en 404 sur un
terme absent :

    Wikipédia mooré     koom=200   zzzqxwv=200
    Wiktionnaire fr     koom=200   zzzqxwv=200
    Wikipédia fr        koom=200   zzzqxwv=200
    Glosbe mos->fr      koom=404   zzzqxwv=404   (la page s'affiche quand même)
    Webonary            koom=403   zzzqxwv=403   (blocage Cloudflare)

D'où le choix d'URL de **recherche** plutôt que de page directe : une URL de
page renvoie 404 dès que le mot est absent, ce qui arrive constamment avec un
corpus de 10 566 entrées face à des dictionnaires généralistes.

Les deux cas particuliers sont signalés par le champ « note » et affichés
comme tels dans l'interface : mieux vaut prévenir que laisser l'utilisateur
croire à un bug.
"""

from urllib.parse import quote_plus


def liens_externes(terme, definition_fr=""):
    """Ressources externes pour un mot mooré.

    `definition_fr` sert aux ressources francophones : chercher « koom » sur
    le Wiktionnaire donne peu, chercher « eau » donne l'article.
    """
    terme = str(terme).strip()
    if not terme:
        return []

    q = quote_plus(terme)
    fr = quote_plus(str(definition_fr).split(",")[0].strip()) if definition_fr else ""

    liens = [
        {
            "nom": "Wikipédia en mooré",
            "url": f"https://mos.wikipedia.org/w/index.php?search={q}",
            "description": "Articles rédigés en mooré (Wikipidiya).",
            "source": "mos.wikipedia.org",
        },
        {
            "nom": "Glosbe mooré → français",
            "url": f"https://glosbe.com/mos/fr/{q}",
            "description": "Dictionnaire mooré-français avec exemples traduits.",
            "source": "glosbe.com",
            "note": "Le serveur renvoie un code 404 même quand la page s'affiche.",
        },
        {
            "nom": "Wiktionnaire",
            "url": f"https://fr.wiktionary.org/w/index.php?search={q}",
            "description": "Définitions, étymologie et prononciation.",
            "source": "fr.wiktionary.org",
        },
        {
            "nom": "Webonary Moore",
            "url": f"https://www.webonary.org/moore/?s={q}&search_type=fulltext",
            "description": "Le dictionnaire dont ce corpus est tiré (SIL).",
            "source": "webonary.org",
            "note": "Protégé par Cloudflare : une vérification peut s'afficher.",
        },
    ]

    if fr:
        liens.append({
            "nom": f"Wikipédia en français « {str(definition_fr).split(',')[0].strip()} »",
            "url": f"https://fr.wikipedia.org/w/index.php?search={fr}",
            "description": "Le concept désigné, côté français.",
            "source": "fr.wikipedia.org",
        })

    liens.append({
        "nom": "Recherche web",
        "url": f"https://duckduckgo.com/?q={q}+moor%C3%A9",
        "description": "Tout le reste : pages, PDF, documents universitaires.",
        "source": "duckduckgo.com",
    })
    return liens
