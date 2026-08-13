"""
api.py
-------
Petite API qui expose search.py en JSON, pour brancher la vraie base
vectorielle derrière la maquette moteur_recherche_moore_maquette.html
(qui, pour l'instant, ne fait tourner qu'un échantillon de 80 mots en local
dans le navigateur).

Lancement :
    pip install flask flask-cors
    python3 api.py
    -> API disponible sur http://127.0.0.1:5000

Endpoint :
    GET /api/search?q=eau&k=10
    -> [{"mot_moore": "...", "definition_francais": "...", ...}, ...]
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from search import search

app = Flask(__name__)
CORS(app)  # autorise la maquette HTML (fichier local ou autre origine) à appeler l'API


@app.route("/api/search")
def api_search():
    q = request.args.get("q", "").strip()
    k = int(request.args.get("k", 10))
    if not q:
        return jsonify([])
    results = search(q, k=k)
    return jsonify(results)


@app.route("/api/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
