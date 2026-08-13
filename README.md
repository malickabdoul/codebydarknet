# Moteur de recherche sémantique mooré — Base vectorielle (Groupe 9)

Projet de web sémantique : moteur de recherche mooré ↔ français ↔ anglais,
basé sur une base de données vectorielle (embeddings TF-IDF/LSA + index
FAISS) construite sur le corpus Webonary Moore (10 566 entrées).

---

## 1. Structure du repo

```
.
├── data/
│   └── corpus_moore_webonary.csv   # corpus source (mot mooré, définitions fr/en, etc.)
├── build_index.py                  # construit la base vectorielle à partir du corpus
├── search.py                       # interroge la base en ligne de commande
├── api.py                          # API Flask qui expose la recherche en JSON
├── moteur_recherche_moore_live.html # interface web (branchée sur l'API)
├── vectors.index                   # index FAISS déjà construit (fourni, prêt à l'emploi)
├── vectorizers.pkl                 # vectoriseurs TF-IDF + SVD entraînés (fourni)
├── corpus_meta.pkl                 # métadonnées alignées avec les vecteurs (fourni)
├── requirements.txt                # dépendances Python
└── README.md                       # ce fichier
```

La base vectorielle (`vectors.index`, `vectorizers.pkl`, `corpus_meta.pkl`)
est **déjà construite et incluse dans le repo** : vous n'êtes pas obligés de
la reconstruire pour tester. Mais si vous modifiez le corpus, relancez
`build_index.py` (voir étape 4).

---

## 2. Prérequis

- Python 3.9 ou plus récent
- `pip`
- Un navigateur web (pour l'interface)

Vérifier votre version de Python :
```bash
python3 --version
```

---

## 3. Installation

### 3.1. Cloner le repo (branche à utiliser : `feature/base-vectorielle`)
```bash
git clone <URL_DU_REPO>
cd <nom_du_dossier>
git checkout feature/base-vectorielle
```

### 3.2. Créer un environnement virtuel (recommandé, évite les conflits)
```bash
python3 -m venv .venv

# Activer l'environnement :
# Linux / Mac :
source .venv/bin/activate
# Windows (PowerShell) :
.venv\Scripts\Activate.ps1
# Windows (cmd) :
.venv\Scripts\activate.bat
```
Vous devez voir `(.venv)` apparaître au début de votre ligne de commande.

### 3.3. Installer les dépendances
```bash
pip install -r requirements.txt
```
Ça installe : `pandas`, `numpy`, `scikit-learn`, `faiss-cpu`, `flask`,
`flask-cors`.

---

## 4. (Optionnel) Reconstruire la base vectorielle

À faire seulement si vous modifiez `data/corpus_moore_webonary.csv`, sinon
passez directement à l'étape 5 (la base fournie dans le repo fonctionne
déjà).

```bash
python3 build_index.py
```

Ce que ça affiche si tout se passe bien :
```
Chargement du corpus...
  -> 10566 entrées chargées
Vectorisation (TF-IDF hybride mots + caractères)...
  -> matrice sparse : (10566, 56529)
R�duction dimensionnelle (LSA / SVD -> 256 dimensions)...
  -> vecteurs denses : (10566, 256)
Normalisation (pour similarité cosinus via produit scalaire)...
Construction de l'index FAISS...
Sauvegarde...
Terminé. Fichiers générés : vectors.index, vectorizers.pkl, corpus_meta.pkl
```
Ça prend environ 30 secondes à 1 minute selon la machine.

---

## 5. Tester en ligne de commande (rapide, sans interface)

```bash
python3 search.py "eau"
python3 search.py "manger"
python3 search.py "how are you"
python3 search.py "ko"          # marche aussi directement en mooré
```

R�sultat attendu (exemple) :
```
Requête : « eau »

  0.866  ko-maasga            (Nom)  fr: eau fraîche, eau froide
  0.861  sudga                (Nom)  fr: cascade, chute d'eau
  0.842  nesneedo             (Nom)  fr: eau croupie, noirâtre
  0.842  bãg-tẽoko            (Nom)  fr: flaque d'eau
  0.837  ko-raalem            (Nom)  fr: eau courante
```

Si ça affiche des résultats comme ça : **la base vectorielle fonctionne.**

---

## 6. Lancer la démo complète (interface web)

Il faut **deux choses en même temps** : l'API qui tourne en arrière-plan, et
la page HTML ouverte dans le navigateur.

### 6.1. Démarrer l'API
Dans un terminal (avec l'environnement virtuel activé) :
```bash
python3 api.py
```
Vous devez voir quelque chose comme :
```
 * Running on http://127.0.0.1:5000
```
**Laissez ce terminal ouvert** — tant qu'il tourne, l'API est disponible.

Vous pouvez vérifier que ça marche en ouvrant dans un navigateur :
```
http://127.0.0.1:5000/api/health
```
→ doit afficher `{"status":"ok"}`

### 6.2. Ouvrir l'interface
Double-cliquez sur `moteur_recherche_moore_live.html` (ou clic droit →
"Ouvrir avec" → votre navigateur). Aucun serveur web n'est nécessaire pour
la page elle-même, seule l'API doit tourner (étape 6.1).

### 6.3. Utilisation
- Tapez un mot en mooré, en français ou en anglais dans la barre de
  recherche → les résultats s'affichent automatiquement après un court
  délai (ou cliquez sur "Baoo").
- Les boutons avec les caractères spéciaux (ã ẽ ĩ õ ũ ɛ ɩ ʋ ŋ ɲ) insèrent
  ces lettres dans la recherche si elles ne sont pas sur votre clavier.
- Les puces sous la barre de recherche filtrent par domaine (ex : "Bird",
  "Body", "Weather"...).

---

## 7. Problèmes fréquents

| Symptôme | Cause probable | Solution |
|---|---|---|
| `ModuleNotFoundError: No module named 'faiss'` | dépendances pas installées | relancer `pip install -r requirements.txt` avec le venv activé |
| La page affiche "Impossible de joindre l'API" | `api.py` n'est pas lancé, ou lancé puis fermé | relancer `python3 api.py` dans un terminal et le laisser ouvert |
| `Address already in use` au lancement de l'API | un `api.py` tourne déjà | fermer l'ancien terminal, ou changer le port dans `api.py` (`app.run(port=5001)`) et dans `API_URL` du HTML |
| Résultats vides sur une recherche | mot trop rare / pas dans le corpus | essayer un synonyme ou un mot plus courant |

---

## 8. Notes techniques (pour la présentation au prof)

- **Vectorisation** : TF-IDF hybride (n-grammes de caractères + mots), pas
  de modèle pré-entraîné téléchargé (contrainte d'environnement), réduit à
  256 dimensions par SVD (Latent Semantic Analysis) → ce sont de vrais
  vecteurs sémantiques, pas juste du texte.
- **Base vectorielle** : index FAISS (`IndexFlatIP`), recherche par
  similarité cosinus.
- **Amélioration future envisageable** : remplacer le TF-IDF par un modèle
  d'embeddings multilingue pré-entraîné (ex.
  `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`) sans
  changer le reste de l'architecture (FAISS + API restent identiques).

---

## 9. Contribuer / pousser vos changements

```bash
git checkout feature/base-vectorielle
git pull
# ... vos modifications ...
git add .
git commit -m "description de ce que vous avez changé"
git push origin feature/base-vectorielle
```
Merci de ne pas pousser directement sur `main` — passez par une pull
request sur cette branche pour que le groupe puisse relire.
