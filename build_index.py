"""
build_index.py
----------------
Construit une base de données vectorielle (index FAISS) à partir du corpus
mooré-français-anglais (corpus_moore_webonary.csv).

Cette base sert de "backend" pour le moteur de recherche sémantique
(cf. maquette moteur_recherche_moore_maquette.html).

Choix technique :
- Pas d'accès réseau vers des modèles d'embeddings hébergés (Hugging Face,
  OpenAI, etc.) dans cet environnement, donc on démarre avec une
  vectorisation TF-IDF hybride (mots + n-grams de caractères) qui gère bien
  le mooré (mots courts, diacritiques : ã ẽ ĩ õ ũ ɛ ɩ ʋ ŋ ɲ).
  -> C'est un point de départ solide et 100% explicable au prof.
  -> Étape suivante possible : remplacer par de vrais embeddings
     (sentence-transformers multilingue) une fois l'accès réseau/GPU dispo,
     sans changer l'architecture (même API de recherche).

Sorties :
  - vectorizers.pkl       : les vectoriseurs TF-IDF + le SVD entraînés
  - vectors.index         : index FAISS (recherche par similarité cosinus)
  - corpus_meta.pkl       : métadonnées alignées avec les vecteurs (mot,
                            définitions, catégorie, etc.)
"""

import pandas as pd
import numpy as np
import pickle
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import hstack

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CORPUS_CSV = os.path.join(BASE_DIR, "data", "corpus_moore_webonary.csv")
# Sorties en chemin absolu : sinon elles atterrissent dans le répertoire
# courant et search.py ne les retrouve pas.
INDEX_PATH = os.path.join(BASE_DIR, "vectors.index")
VECTORIZERS_PATH = os.path.join(BASE_DIR, "vectorizers.pkl")
META_PATH = os.path.join(BASE_DIR, "corpus_meta.pkl")


def load_corpus() -> pd.DataFrame:
    df = pd.read_csv(CORPUS_CSV, sep=";")
    df = df.dropna(subset=["mot_moore"]).reset_index(drop=True)

    def build_doc(row):
        parts = [
            str(row["mot_moore"]),
            str(row.get("definition_francais", "") or ""),
            str(row.get("definition_english", "") or ""),
            str(row.get("variante", "") or ""),
            str(row.get("domaine", "") or ""),
        ]
        return " ".join(p for p in parts if p and p != "nan")

    df["doc_text"] = df.apply(build_doc, axis=1)
    return df


def build_vectors(df: pd.DataFrame):
    # Vectoriseur 1 : n-grams de caractères -> capte bien la morphologie
    # du mooré et les fautes/variantes orthographiques
    char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2)
    X_char = char_vec.fit_transform(df["doc_text"])

    # Vectoriseur 2 : mots -> capte le sens via les définitions fr/en
    word_vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2)
    X_word = word_vec.fit_transform(df["doc_text"])

    # Fusion des deux représentations (pondération simple)
    X = hstack([X_char * 0.5, X_word * 1.0]).tocsr().astype(np.float32)

    return X, {"char_vectorizer": char_vec, "word_vectorizer": word_vec}


def main():
    print("Chargement du corpus...")
    df = load_corpus()
    print(f"  -> {len(df)} entrées chargées")

    print("Vectorisation (TF-IDF hybride mots + caractères)...")
    X, vectorizers = build_vectors(df)
    print(f"  -> matrice sparse : {X.shape}")

    print("Réduction dimensionnelle (LSA / SVD -> 256 dimensions)...")
    n_components = 256
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    dense = svd.fit_transform(X).astype(np.float32)
    print(f"  -> vecteurs denses : {dense.shape}")

    print("Normalisation (pour similarité cosinus via produit scalaire)...")
    norms = np.linalg.norm(dense, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    dense = dense / norms

    print("Construction de l'index FAISS...")
    index = faiss.IndexFlatIP(dense.shape[1])
    index.add(dense.astype(np.float32))

    print("Sauvegarde...")
    faiss.write_index(index, INDEX_PATH)
    with open(VECTORIZERS_PATH, "wb") as f:
        pickle.dump({**vectorizers, "svd": svd}, f)
    # doc_text ne sert qu'à la vectorisation : inutile de l'embarquer dans
    # les métadonnées (le CSV source la reconstruit).
    df.drop(columns=["doc_text"]).to_pickle(META_PATH)

    print("Terminé. Fichiers générés : vectors.index, vectorizers.pkl, corpus_meta.pkl")


if __name__ == "__main__":
    main()
