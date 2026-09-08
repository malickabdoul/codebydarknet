"""
build_st_index.py
------------------
Construit l'index sémantique (embeddings pré-entraînés), en plus de l'index
TF-IDF/LSA produit par build_index.py.

À lancer une fois :
    pip install sentence-transformers
    python3 build_st_index.py

Sortie : vectors_st.index (aligné ligne à ligne avec corpus_meta.pkl)

Ce qui est encodé
-----------------
Les définitions **française et anglaise**, pas le mot mooré. Le modèle ne
connaît pas le mooré (cf. embeddings.py) : l'encoder produirait du bruit. Les
entrées mooré restent couvertes par l'index TF-IDF, qui les gère mieux.

L'index TF-IDF n'est pas touché. Les deux coexistent et sont fusionnés au
moment de la recherche.
"""

import os
import sys

import faiss
import numpy as np
import pandas as pd

import embeddings

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
META_PATH = os.path.join(BASE_DIR, "corpus_meta.pkl")


def texte_semantique(row):
    """Le côté du corpus que le modèle sait lire."""
    parts = []
    for col in ("definition_francais", "definition_english"):
        v = row.get(col)
        if v is not None and str(v) != "nan" and str(v).strip():
            parts.append(str(v).strip())
    return " | ".join(parts)


def main():
    if not embeddings.disponible():
        print("sentence-transformers n'est pas installé.")
        print("  pip install sentence-transformers")
        return 1

    print("Chargement des métadonnées...")
    meta = pd.read_pickle(META_PATH)
    print(f"  -> {len(meta)} entrées")

    textes = [texte_semantique(meta.iloc[i]) for i in range(len(meta))]
    vides = sum(1 for t in textes if not t)
    print(f"  -> {len(textes) - vides} entrées avec définition, {vides} vides")

    print(f"Chargement du modèle {embeddings.MODELE}...")
    print("  (premier lancement : téléchargement d'environ 470 Mo)")

    print("Encodage du corpus...")
    vecteurs = embeddings.encoder(textes, batch_size=64, montrer_progres=True)
    print(f"  -> vecteurs : {vecteurs.shape}")

    if vecteurs.shape[1] != embeddings.DIMENSION:
        print(f"  ! dimension inattendue : {vecteurs.shape[1]} "
              f"au lieu de {embeddings.DIMENSION}")

    print("Construction de l'index FAISS...")
    index = faiss.IndexFlatIP(vecteurs.shape[1])
    index.add(vecteurs)

    faiss.write_index(index, embeddings.INDEX_ST_PATH)
    taille = os.path.getsize(embeddings.INDEX_ST_PATH) / (1024 * 1024)
    print(f"Terminé : {embeddings.INDEX_ST_PATH} ({taille:.1f} Mo, "
          f"{index.ntotal} vecteurs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
