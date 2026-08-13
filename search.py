"""
search.py
----------
Interroge la base vectorielle construite par build_index.py.
Fonctionne en français, en anglais, ou en mooré.

Usage :
    python3 search.py "eau"
    python3 search.py "how are you"
    python3 search.py "ko"
"""

import sys
import pickle
import numpy as np
import faiss
from scipy.sparse import hstack


def load_all():
    index = faiss.read_index("vectors.index")
    with open("vectorizers.pkl", "rb") as f:
        vec = pickle.load(f)
    import pandas as pd
    meta = pd.read_pickle("corpus_meta.pkl")
    return index, vec, meta


def embed_query(query, vec):
    X_char = vec["char_vectorizer"].transform([query])
    X_word = vec["word_vectorizer"].transform([query])
    X = hstack([X_char * 0.5, X_word * 1.0]).tocsr()
    dense = vec["svd"].transform(X).astype(np.float32)
    norm = np.linalg.norm(dense, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    return dense / norm


def search(query, k=5):
    index, vec, meta = load_all()
    q = embed_query(query, vec)
    scores, idxs = index.search(q.astype(np.float32), k)
    results = []
    for score, idx in zip(scores[0], idxs[0]):
        if idx == -1:
            continue
        row = meta.iloc[idx]

        def clean(v):
            return "" if v is None or (isinstance(v, float) and str(v) == "nan") else str(v)

        results.append({
            "mot_moore": clean(row["mot_moore"]),
            "definition_francais": clean(row["definition_francais"]),
            "definition_english": clean(row["definition_english"]),
            "categorie": clean(row["categorie_grammaticale"]),
            "prononciation": clean(row.get("prononciation", "")),
            "domaine": clean(row.get("domaine", "")),
            "score": float(score),
        })
    return results


if __name__ == "__main__":
    query = " ".join(sys.argv[1:]) or "eau"
    print(f"Requête : « {query} »\n")
    for r in search(query):
        print(f"  {r['score']:.3f}  {r['mot_moore']:20s} "
              f"({r['categorie']})  fr: {r['definition_francais'][:60]}")
