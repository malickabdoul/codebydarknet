# Moteur de recherche Web multilingue pour le mooré (Groupe 9)

Projet de web sémantique : l'utilisateur formule une requête en **mooré**, en
**français** ou en **anglais**, et le moteur lui rend des **pages Web** avec
leur titre, un extrait et un lien.

Le corpus Webonary Moore (10 566 entrées), indexé dans une base vectorielle
(FAISS), sert de **pont de traduction** entre les langues : « koom » devient
« eau » avant d'interroger les sources francophones. Deux modes
complémentaires donnent accès au dictionnaire lui-même et au glossage de
phrases.

**Documents du projet**
- [`docs/manuel_utilisation.pdf`](docs/manuel_utilisation.pdf) — manuel d'utilisation (20 pages)
- [`docs/cahier_charges_technique.pdf`](docs/cahier_charges_technique.pdf) — cahier des charges technique (19 pages)

---

## 1. Structure du repo

```
.
├── data/
│   └── corpus_moore_webonary.csv   # corpus source (mot mooré, définitions fr/en, etc.)
├── test_moteur.py                  # tests de non-régression (57)
├── benchmark.py                    # compare les backends, chiffré
├── build_index.py                  # construit l'index TF-IDF/LSA à partir du corpus
├── build_st_index.py               # construit l'index sémantique (optionnel)
├── embeddings.py                   # modèle d'embeddings pré-entraîné (optionnel)
├── recherche_web.py                # recherche Web + pont de traduction
├── search.py                       # interroge la base en ligne de commande
├── liens.py                        # ressources web externes par mot
├── translate.py                    # glose une PHRASE française en mooré
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
Réduction dimensionnelle (LSA / SVD -> 256 dimensions)...
  -> vecteurs denses : (10566, 256)
Normalisation (pour similarité cosinus via produit scalaire)...
Construction de l'index FAISS...
Sauvegarde...
Terminé. Fichiers générés : vectors.index, vectorizers.pkl, corpus_meta.pkl
```
Ça prend environ 30 secondes à 1 minute selon la machine.

---

## 4 bis. Recherche Web : ce que fait le moteur

C'est la fonction principale, celle que décrit le cahier des charges (EF07 à
EF12). Le corpus n'est pas la destination, c'est le **pont** :

```
requête mooré  ->  corpus  ->  terme français
                                    |
                               sources Web
                                    |
    titre + extrait + lien  <-  reclassement sémantique
```

```bash
python3 recherche_web.py "koom"
```

```
Requête : « koom »
Pont     : « koom » traduit en « eau » par le corpus
Sources  : Wikipédia en mooré, Wikipédia, Wiktionnaire
Résultats reclassés par proximité de sens.

1. Eau  [0.65]
   précise : eau minérale, eau de Seltz, eau de source, eau de mer…
   https://fr.wikipedia.org/wiki/Eau
   — Wikipédia
```

### Les sources
Trois encyclopédies MediaWiki, **sans clé d'API ni quota** : Wikipédia en
mooré, Wikipédia en français et le Wiktionnaire. Elles rendent exactement les
trois champs exigés — titre, extrait, lien.

L'architecture est enfichable : ajouter un moteur généraliste (Bing, Brave,
Google Custom Search) revient à ajouter une entrée dans `SOURCES` et une
fonction d'interrogation. Ces moteurs demandent une clé, raison pour laquelle
ils ne sont pas activés par défaut.

### Deux précautions apprises en chemin

**Le pont refuse les approximations.** Un mot absent du corpus obtenait quand
même une « traduction » : `zzzqxwv` devenait « presque totalité », et le Web
était interrogé sur cette invention. Un seuil de 0,45 sur le score de
correspondance y met fin — mesuré : un mot exact obtient 1,000, une faute de
frappe plausible 0,551, du charabia 0,256.

**Le reclassement épargne le mooré.** Le modèle d'embeddings ne connaît pas la
langue : reclasser tous les résultats enterrait systématiquement les pages du
Wikipédia mooré — sur « koom », aucune ne subsistait dans les quinze
premières. Seules les pages en langue connue du modèle sont donc reclassées,
et les pages en mooré, gardées dans l'ordre de leur source, sont intercalées
une sur trois.

---

## 5. Tester en ligne de commande (rapide, sans interface)

```bash
python3 search.py "eau"
python3 search.py "manger"
python3 search.py "how are you"
python3 search.py "ko"          # marche aussi directement en mooré
python3 search.py "eau" --domaine Water   # restreint à un domaine
```

Résultat attendu (exemple) :
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

## 5 bis. Traduire une phrase (et pas seulement un mot)

`search.py` encode toute la requête en **un seul vecteur** et rend l'entrée la
plus proche. C'est ce qu'on veut pour un mot, mais absurde sur une phrase :
« je veux boire de l'eau » renvoyait « tas de résidus de minerai de fer ».

`translate.py` fait l'inverse : il découpe la phrase et cherche segment par
segment, **en essayant les expressions d'abord**.

```bash
python3 translate.py "je veux boire de l'eau"
python3 translate.py "n'importe comment"
```

Ce que ça donne :
```
Phrase : « je veux boire de l'eau »
Couverture : 100 %  (0 expression(s) reconnue(s))

  [mot ] je                     -> m
  [mot ] veux                   -> da   (via « vouloir »)
  [mot ] boire                  -> yũ
  [----] de                     -> —
  [----] l                      -> —
  [mot ] eau                    -> koom

Glose : m da yũ koom
```

### Ce que ça sait faire
- **Expressions d'abord** : « n'importe comment » sort `a bal` d'un bloc, au
  lieu de « n'importe » + « comment ». Le corpus contient 504 expressions et
  5 753 gloses françaises de plus d'un mot.
- **Formes conjuguées** : « veux » → `vouloir`, « mangé » → `manger`,
  « allons » → `aller` (table d'irréguliers + découpage par suffixe).
- **Accents porteurs de sens** : « marché » → `raaga` (le lieu) et
  « marche » → `kẽnde` (l'action) ne sont pas confondus.
- **Mots-outils** : « le », « la », « de » sont marqués non traduits plutôt
  que forcés. Sans ça « la » sortait `ka`, qui est la **négation**.

Mesure sur les 306 expressions multi-mots du corpus (vérité terrain = le
corpus lui-même) :

| | top-1 | top-3 |
|---|---|---|
| `search.py` (vecteur unique) | 52,3 % | 72,9 % |
| `translate.py` (segmentation) | **82,4 %** | **93,5 %** |

### Tournures idiomatiques
La segmentation cherche des correspondances littérales. Elle ne peut pas
deviner que « tu vas bien » se dit **`laafɩ bala`** (littéralement « ça va ») :
aucun mot ne coïncide. La base vectorielle, elle, place cette entrée à **0,64**
de la phrase.

`translate.py` interroge donc l'index FAISS sur la phrase entière, garde les
entrées de type expression / interj, et les propose **en plus** de la glose :

```
Glose : f kɩbe neere

Tournures idiomatiques proches (base vectorielle) :
  0.643  laafɩ bala   (interj)  ça va, ça va bien
```

C'est le seul endroit où la base vectorielle sert à autre chose qu'à chercher
un mot — et c'est ce qui justifie de l'avoir construite.

Attention : ces suggestions sont **bruitées sur les phrases descriptives**. Sur
« je veux boire de l'eau », la première proposition est « œufs de pou » à 0,58,
donc plus haut que des suggestions justes ailleurs. Les scores se chevauchent :
aucun seuil ne sépare le bon du mauvais. C'est pour ça qu'elles sont affichées
avec leur score et présentées comme **à valider par un locuteur**, jamais
substituées à la glose.

### Ce que ça ne sait PAS faire
Ce n'est **pas de la traduction**, c'est une **glose** : le meilleur équivalent
mooré de chaque segment, dans l'ordre du français. Le mooré a son propre ordre
des mots, ses postpositions et ses marques d'aspect, qu'un dictionnaire ne
permet pas de reconstruire. La sortie est un support pour un locuteur, pas une
phrase à publier telle quelle — et l'interface l'affiche explicitement.

Départager les synonymes restants (`kɩbe` / `kẽnge` pour « aller ») demande un
locuteur : le module propose les alternatives au lieu de trancher seul.

---

## 5 ter. Vérifier que tout marche

```bash
python3 test_moteur.py          # 57 tests, ~25 s
python3 test_moteur.py -v       # detail test par test
```

Aucune dépendance en plus : `unittest` est dans la bibliothèque standard. Les
tests qui exigent l'index sémantique se sautent d'eux-mêmes s'il n'est pas
construit — la suite passe donc sur une installation minimale.

Ce qui est couvert, ce sont les endroits où le corpus piège, chacun venant
d'un défaut réellement rencontré :

| bloc | ce qui est vérifié |
|---|---|
| `TestDomaines` | `Bush, shrub` reste un seul domaine ; les 130 valeurs se reconstruisent |
| `TestEntetesCorrompus` | les 7 en-têtes mélangés, dans les deux sens ; notations tonales intactes |
| `TestRenvois` | synonymes et antonymes démêlés ; chiffre de sens retiré |
| `TestCorrespondanceExacte` | `koom` sort en premier ; `bag-teoko` retrouve `bãg-tẽoko` |
| `TestRoutage` | `auto` choisit le bon index ; backend inconnu lève |
| `TestBornesRecherche` | `k` négatif, nul, démesuré |
| `TestNormalisationFrancaise` | élision `l'eau`, ligature `œil` (qui devenait `il`) |
| `TestGloses` | `le poisson` n'est pas indexé comme glose autonome |
| `TestSegmentation` | expression d'un bloc, lemmatisation, `la` non traduit, `marché`/`marche` |
| `TestApi` | codes 400, filtre de domaine composé, CORS restreint |

### Ces tests détectent-ils vraiment quelque chose ?
Une suite verte ne prouve rien par elle-même. Cinq régressions ont été
introduites volontairement pour vérifier qu'un test échoue bien à chaque fois :
retour au découpage naïf des domaines, arrêt du démêlage synonymes/antonymes,
`la` de nouveau traduit, fragments de définition réindexés, en-tête corrompu
mal réparé. **5 sur 5 ont été détectées.**

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
| Un résultat affiche du français ou de l'anglais comme mot mooré | en-tête corrompu dans le CSV source | 7 cas connus, réparés à l'affichage par `nettoyer_mot_moore()` |
| `{"error": "paramètre k invalide"}` | `k` non entier, ≤ 0, ou absent | passer un entier entre 1 et 100 |
| L'API répond mais sans le champ `entrees` | un ancien `api.py` tourne encore sur le port 5000 | fermer tous les terminaux Python, puis relancer |

---

## 7 bis. Défauts connus du corpus source

Trouvés en travaillant sur les données, non corrigés dans le CSV :

- **7 en-têtes mélangent le mot mooré et sa traduction** — `'Come! y'`,
  `'Ges neere! Regarde bien ! Watch well!'`. Réparés à l'affichage par
  `nettoyer_mot_moore()` dans `search.py`, mais le CSV reste à nettoyer.
- **Ligne 1035 du CSV : colonnes décalées.** Le champ `mot_moore` contient
  `past tense marker,` et la définition française contient du mooré, du
  français, de l'anglais et un résidu de scraping `Category: Grammar`. Trop
  ambigu pour être réparé automatiquement — à corriger à la main.
- **Doublons** : plusieurs entrées identiques (deux `neere` définis « bien »).
- **74 définitions coupées net** (57 fr, 17 en) : « têtu (lit », « tomber (ex »,
  « build, construct (e.g ». Le scraper a pris le point de l'abréviation
  (`litt.`, `ex.`, `e.g.`) pour une fin de phrase. Le texte manquant n'est pas
  dans le CSV, donc irrécupérable ; `marquer_tronquee()` ajoute « […] » pour
  que ça se lise comme une donnée incomplète et non comme un bug d'affichage.
- **Colonnes synonymes/antonymes mélangées** : 28 champs « synonymes »
  contiennent en réalité `...; antonyme: X`, et 29 champs « antonymes »
  contiennent `...; synonyme: X`. `renvois()` dans `search.py` les redistribue
  (57 entrées concernées) — sans ça, « ba » affichait « saamba; antonyme: ma »
  comme un seul synonyme.
- **Renvois suffixés d'un chiffre** : `ma1` désigne le premier sens de `ma`.
  Sur 3 545 renvois, 2 800 correspondent à une entrée tels quels et 3 429 une
  fois le chiffre retiré — d'où le nettoyage avant résolution.
- **Un domaine vaut `7`** (42 entrées), ce qui donne une puce « 7 » dans
  l'interface. C'est la valeur réelle du corpus : filtrer ces entrées les
  rendrait invisibles, donc elles sont laissées telles quelles.

---

## 8. Notes techniques (pour la présentation au prof)

- **Vectorisation** : TF-IDF hybride (n-grammes de caractères + mots), pas
  de modèle pré-entraîné téléchargé (contrainte d'environnement), réduit à
  256 dimensions par SVD (Latent Semantic Analysis) → ce sont de vrais
  vecteurs sémantiques, pas juste du texte.
- **Base vectorielle** : index FAISS (`IndexFlatIP`), recherche par
  similarité cosinus.
- **Recherche hybride** : deux index coexistent et sont fusionnés. Voir la
  section 9 ci-dessous — c'est le point le plus intéressant à présenter.

---

## 9. Recherche : cinq backends, et pourquoi aucun ne suffit

### Le point de départ
TF-IDF ne fait pas de sémantique, il fait de la co-occurrence de caractères et
de mots. D'où des suggestions absurdes : « œufs de pou » proposé à 0,58 pour
« je veux boire de l'eau », parce que les chaînes se ressemblent, pas les sens.

La réponse habituelle est de remplacer le TF-IDF par des embeddings
pré-entraînés. **Les mesures disent que ce serait une erreur.**

### Ce que dit le benchmark

`python3 benchmark.py`, vérité terrain = le corpus lui-même.

| backend | expression depuis sa définition fr | | entrée mooré avec une faute de frappe | |
|---|---|---|---|---|
| | top-1 | top-5 | top-1 | top-5 |
| `entrees` | 0,7 % | 2,9 % | **59,1 %** | **77,0 %** |
| `tfidf` | 52,3 % | 77,8 % | 6,4 % | 16,6 % |
| `semantique` | 73,9 % | 90,8 % | 0,0 % | 0,0 % |
| `hybride` | **77,5 %** | **94,8 %** | 5,1 % | 13,2 % |
| **`auto`** | **77,5 %** | **94,8 %** | 57,4 % | 74,9 % |

Chaque backend est excellent d'un côté et nul de l'autre. Le sémantique tombe
à **0 %** en mooré : `paraphrase-multilingual-MiniLM-L12-v2` couvre une
cinquantaine de langues, et le mooré — langue gur très peu dotée — n'en fait
pas partie. Le tokenizer le découpe en sous-mots qui ne signifient rien pour
le modèle. Symétriquement, l'index des en-têtes fait 0,7 % en français : il ne
regarde pas les définitions.

Aucun ne peut servir de défaut. Seul le routage obtient les deux colonnes.

### Les cinq backends

| backend | ce qu'il indexe | pour quoi |
|---|---|---|
| `entrees` | les en-têtes mooré seuls, n-grammes de caractères | retrouver un mot mooré, même mal écrit |
| `tfidf` | en-tête **et** définitions mélangés (l'index d'origine) | recherche générale, sans rien installer |
| `semantique` | définitions fr/en, embeddings pré-entraînés | sens d'une requête française |
| `hybride` | fusion RRF de `tfidf` et `semantique` | requête française |
| `auto` | route selon la langue détectée | **défaut** |

```bash
python3 search.py "koom"            # auto -> entrees
python3 search.py "eau"             # auto -> hybride
python3 search.py "eau" --backend tfidf
```

### Trois détails qui comptent

**Pourquoi un index séparé pour les en-têtes.** `build_index.py` indexe un
document mélangé : pour l'entrée `maande`, c'est `maande | gombo | okra`.
L'en-tête ne pèse qu'un tiers, donc une requête mal orthographiée comme
`maade` n'a presque rien à quoi s'accrocher — 16,6 % en top-5. En n'indexant
que les en-têtes : 77,0 %. Cet index est construit à la volée (~1 s), il n'y a
pas de fichier de plus à versionner.

**Pourquoi fusionner par les rangs.** Un cosinus TF-IDF/LSA et un cosinus
d'embeddings ne vivent pas sur la même échelle ; les additionner n'a pas de
sens et les normaliser demanderait une calibration arbitraire. La *Reciprocal
Rank Fusion* ne regarde que le rang :

```
score(entrée) = Σ  1 / (60 + rang dans la liste i)
                i
```

Aucun paramètre à régler. À noter : la fusion n'aide pas partout — sur les
fautes de frappe elle fait **moins bien** (13,2 %) que l'index des en-têtes
seul (77,0 %), parce que le document mélangé n'y apporte que du bruit.

**Les correspondances exactes passent devant.** Ni le TF-IDF ni les embeddings
ne garantissent qu'une entrée cherchée mot pour mot sorte en tête : les
vecteurs sont compressés par SVD, et `koom` se faisait dépasser par
`rʋʋd-koom`. Une recherche exacte est maintenant traitée avant les vecteurs,
et tolère l'absence de diacritiques (`bag-teoko` retrouve `bãg-tẽoko`).

### Choisir un index à la main
Le sélecteur « Index » de l'interface permet de forcer un backend, ce qui est
utile pour montrer les différences en soutenance. Mais forcer un index
inadapté donne du bruit : « Sémantique seul » sur un mot mooré renvoie
*savane*, *savoir*, *Sahel* pour `saamba` — c'est le 0,0 % du tableau, en
direct.

L'interface le détecte et l'affiche, avec un bouton de retour en « Auto » :

> Requête détectée comme **mooré** — mais l'index sémantique ne connaît pas le
> mooré. Au-delà des correspondances exactes, ces résultats ne veulent pas dire
> grand-chose.

Sur la même requête `saamba` :

| index | résultats |
|---|---|
| Sémantique seul | saamba, sãamba, **weo-faoogo, mi, tẽn-koɛɛnga** |
| Auto | saamba, sãamba, soaamba, karen-saamba, yamba |

### Installation (optionnelle)
Le dépôt fonctionne **sans** : `entrees` et `tfidf` ne demandent rien de plus,
et `auto` s'y limite automatiquement. Personne n'est obligé de télécharger
1 Go pour faire tourner la démo.

```bash
pip install sentence-transformers   # tire torch, ~550 Mo
python3 build_st_index.py           # modèle ~470 Mo, puis encodage (~3 min)
```

Ça produit `vectors_st.index` (15,5 Mo). `GET /api/health` indique quels
backends sont disponibles, et l'interface n'affiche le sélecteur que s'il y a
plus d'un choix.

### Limite de ces chiffres
La vérité terrain vient du corpus, et les index sont construits sur ce même
corpus : ces mesures évaluent la **récupération**, pas la généralisation à des
formulations inédites. Un vrai jeu de test demanderait des phrases annotées
par un locuteur.

---

## 9 bis. Liens web externes

Sous les résultats, la section « Sur le web » propose des ressources pour le
mot trouvé. Elles **ne font pas partie du corpus** et l'interface le dit.

Chaque URL a été ouverte dans un vrai navigateur avant d'être retenue, avec un
mot du corpus (`koom`) **et** un mot qui n'existe nulle part (`zzzqxwv`), pour
vérifier qu'elle ne tombe pas en 404 sur un terme absent :

| ressource | `koom` | `zzzqxwv` | retenue |
|---|---|---|---|
| Wikipédia mooré | 200 | 200 | oui |
| Wiktionnaire fr | 200 | 200 | oui |
| Wikipédia fr | 200 | 200 | oui |
| Glosbe mooré→fr | 404 | 404 | oui, **signalée** |
| Webonary Moore | 403 | 403 | oui, **signalée** |

D'où l'usage d'URL de **recherche** plutôt que de page directe : une URL de
page renvoie 404 dès que le mot manque, ce qui arrive constamment quand on
confronte 10 566 entrées à des dictionnaires généralistes.

Les deux cas problématiques sont affichés avec un avertissement plutôt que
masqués : Glosbe renvoie un code 404 alors que la page s'affiche correctement,
et Webonary — la source même du corpus — est protégé par Cloudflare, qui
bloque tout accès automatisé. Un navigateur ordinaire passe généralement.

Les liens s'ouvrent dans un nouvel onglet avec `rel="noopener noreferrer"`.

---

## 10. Contribuer / pousser vos changements

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
