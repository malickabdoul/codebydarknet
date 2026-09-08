"""
api.py
-------
API qui expose search.py en JSON, pour brancher la base vectorielle derrière
l'interface moteur_recherche_moore_live.html.

Lancement :
    pip install -r requirements.txt
    python3 api.py
    -> API disponible sur http://127.0.0.1:5000
    -> interface servie sur http://127.0.0.1:5000/

Endpoints :
    GET /api/search?q=eau&k=10[&domaine=Water][&backend=auto]
        -> [{"mot_moore": "...", "definition_francais": "...", ...}, ...]
    GET /api/domaines
        -> ["Agriculture", "Animal", ...]  (déduits du corpus)
    GET /api/web?q=koom&k=8
        -> {"pont": "...", "resultats": [{"titre", "extrait", "url", ...}]}
    GET /api/liens?q=koom[&fr=eau]
        -> [{"nom": "...", "url": "...", "description": "..."}, ...]
    GET /api/translate?q=je veux boire de l'eau
        -> glose segment par segment + tournures idiomatiques proches
    GET /api/health
        -> {"status": "ok", "entrees": 10566, "backends": [...]}
"""

import os

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

import embeddings
from liens import liens_externes
from recherche_web import rechercher_web
from search import BACKENDS, backend_par_defaut, list_domaines, load_all, search
from translate import traduire

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PAGE = "moteur_recherche_moore_live.html"

# k maximum accepté : borne le travail fait pour une requête HTTP.
MAX_K = 100

app = Flask(__name__, static_folder=None)

# L'interface peut être ouverte de deux façons : servie par cette API (même
# origine, rien à autoriser) ou en double-cliquant le fichier HTML, auquel cas
# le navigateur envoie « null » comme origine. On autorise ces cas-là
# seulement, au lieu de « toutes les origines » : sans ça, n'importe quel site
# visité pendant que l'API tourne pourrait l'interroger.
CORS(app, resources={r"/api/*": {"origins": [
    "null",
    "http://127.0.0.1:5000",
    "http://localhost:5000",
]}})


def _parse_k(raw):
    """Renvoie un k valide, ou None si la valeur est inexploitable."""
    try:
        k = int(raw)
    except (TypeError, ValueError):
        return None
    if k < 1:
        return None
    return min(k, MAX_K)


@app.route("/")
def page():
    return send_from_directory(BASE_DIR, PAGE)


@app.route("/api/search")
def api_search():
    q = request.args.get("q", "").strip()
    k = _parse_k(request.args.get("k", 10))
    if k is None:
        return jsonify({"error": f"paramètre k invalide (entier entre 1 et {MAX_K})"}), 400

    domaine = request.args.get("domaine", "").strip() or None
    if domaine and domaine not in list_domaines():
        return jsonify({"error": f"domaine inconnu : {domaine}"}), 400

    backend = request.args.get("backend", "").strip() or backend_par_defaut()
    if backend not in BACKENDS:
        return jsonify({"error": f"backend inconnu : {backend} "
                                 f"(attendu : {', '.join(BACKENDS)})"}), 400
    if backend != "tfidf" and not embeddings.index_disponible():
        return jsonify({"error": "index sémantique absent — lancez "
                                 "python3 build_st_index.py"}), 409

    if not q:
        return jsonify([])
    return jsonify(search(q, k=k, domaine=domaine, backend=backend))


@app.route("/api/translate")
def api_translate():
    """Glossage français -> mooré d'une phrase entière.

    Distinct de /api/search : celui-ci découpe la phrase et cherche segment par
    segment, en privilégiant les expressions. /api/search encode toute la
    requête en un seul vecteur, ce qui n'a de sens que pour un mot ou une
    expression isolée.
    """
    q = request.args.get("q", "").strip()
    if not q:
        return jsonify({"phrase": "", "segments": [], "glose": "",
                        "couverture": 0.0, "n_expressions": 0,
                        "non_resolus": []})
    if len(q) > 500:
        return jsonify({"error": "phrase trop longue (500 caractères maxi)"}), 400
    return jsonify(traduire(q))


@app.route("/api/web")
def api_web():
    """EF07-EF12 : cherche sur le Web et rend titre, extrait et lien.

    Le corpus sert de pont : une requête mooré est traduite avant d'être
    envoyée aux sources francophones, et inversement.
    """
    q = request.args.get("q", "").strip()
    if not q:
        return jsonify({"requete": "", "resultats": [], "sources": [], "pont": ""})
    if len(q) > 200:
        return jsonify({"error": "requête trop longue (200 caractères maxi)"}), 400

    k = _parse_k(request.args.get("k", 8))
    if k is None:
        return jsonify({"error": f"paramètre k invalide (entier entre 1 et {MAX_K})"}), 400

    return jsonify(rechercher_web(q, k=min(k, 20)))


@app.route("/api/liens")
def api_liens():
    """Ressources externes pour un mot : affichées sous les résultats."""
    q = request.args.get("q", "").strip()
    if not q:
        return jsonify([])
    if len(q) > 200:
        return jsonify({"error": "terme trop long (200 caractères maxi)"}), 400
    fr = request.args.get("fr", "").strip()
    return jsonify(liens_externes(q, definition_fr=fr))


@app.route("/api/domaines")
def api_domaines():
    return jsonify(list_domaines())


@app.route("/api/health")
def health():
    index, _, _ = load_all()
    return jsonify({
        "status": "ok",
        "entrees": index.ntotal,
        "backends": ([b for b in BACKENDS] if embeddings.index_disponible()
                     else [b for b in BACKENDS if b not in ("semantique", "hybride")]),
        "backend_defaut": backend_par_defaut(),
        "semantique_disponible": embeddings.index_disponible(),
    })


if __name__ == "__main__":
    # Chargement au démarrage plutôt qu'à la première requête : la page ne
    # reste pas bloquée plusieurs secondes sur la première recherche.
    print("Chargement de la base vectorielle...")
    index, _, _ = load_all()
    print(f"  -> {index.ntotal} entrées, {len(list_domaines())} domaines")
    print("  -> interface : http://127.0.0.1:5000/")
    # debug=True activerait le debugger Werkzeug (exécution de code à distance
    # si le port est joignable) : à garder désactivé.
    app.run(debug=False, port=5000)
