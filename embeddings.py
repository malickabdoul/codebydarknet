"""
embeddings.py
--------------
Vecteurs sémantiques pré-entraînés, en complément du TF-IDF/LSA.

Pourquoi en COMPLÉMENT et pas en remplacement
---------------------------------------------
Le modèle multilingue couvre une cinquantaine de langues. Le mooré n'en fait
pas partie : c'est une langue gur très peu dotée, absente des corpus
d'entraînement. Le tokenizer la découpe en sous-mots qui ne veulent rien dire
pour le modèle.

Concrètement :

- côté **français / anglais** (les définitions), le modèle comprend le sens.
  C'est ce qui manquait : TF-IDF ne fait pas de sémantique, il fait de la
  co-occurrence, d'où « œufs de pou » à 0,58 pour « je veux boire de l'eau ».
- côté **mooré** (les entrées elles-mêmes), le TF-IDF en n-grammes de
  caractères est meilleur : il capte la morphologie et les diacritiques
  (ã ẽ ĩ õ ũ ɛ ɩ ʋ ŋ ɲ), et tolère les variantes orthographiques.

Remplacer l'un par l'autre améliorerait donc une moitié du moteur en cassant
l'autre. On garde les deux index et on les fusionne (cf. search.py).

Le modèle est optionnel : sans lui, tout le projet continue de tourner sur le
TF-IDF seul. C'est `disponible()` qui le dit.
"""

import os

MODELE = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DIMENSION = 384

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_ST_PATH = os.path.join(BASE_DIR, "vectors_st.index")

_MODELE = None


def disponible():
    """La bibliothèque est-elle installée ?

    Le reste du projet doit fonctionner sans : le dépôt fournit l'index
    TF-IDF déjà construit, et tout le monde n'a pas envie de télécharger
    torch et un modèle de 470 Mo pour faire tourner la démo.
    """
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        return False
    return True


def index_disponible():
    return disponible() and os.path.exists(INDEX_ST_PATH)


def modele():
    """Charge le modèle une seule fois (plusieurs secondes au premier appel)."""
    global _MODELE
    if _MODELE is None:
        from sentence_transformers import SentenceTransformer
        _MODELE = SentenceTransformer(MODELE)
    return _MODELE


def encoder(textes, batch_size=64, montrer_progres=False):
    """Encode une liste de textes en vecteurs normalisés (float32).

    Normalisés parce que l'index FAISS est un IndexFlatIP : le produit
    scalaire de deux vecteurs normalisés vaut leur similarité cosinus.
    """
    import numpy as np

    if isinstance(textes, str):
        textes = [textes]
    vecteurs = modele().encode(
        list(textes),
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=montrer_progres,
    )
    return vecteurs.astype(np.float32)
