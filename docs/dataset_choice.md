# Choix et description du dataset — Partie supervisée

Projet DR CVRIE · Epitech B-SPE-311
Dataset retenu : **BloodMNIST** (collection MedMNIST v2)

---

## 1. Présentation du dataset

BloodMNIST est un sous-ensemble de la collection MedMNIST v2, un recueil de 12 datasets
biomédicaux 2D standardisés au format 28×28, conçu comme un banc d'essai léger pour
l'évaluation d'algorithmes de classification.

Les données d'origine proviennent du laboratoire d'hématologie de l'hôpital Clínic de
Barcelone. Il s'agit de clichés de cellules sanguines individuelles acquis au microscope
sur frottis coloré.

**Point déterminant pour l'interprétation de nos résultats :** les prélèvements proviennent
de patients exempts d'infection, de maladie hématologique ou oncologique, et de tout
traitement pharmacologique au moment du prélèvement. Toutes les cellules du dataset sont
donc morphologiquement normales.

### Composition

| Classe | Type cellulaire | Images (train) | Proportion |
|--------|-----------------|----------------|------------|
| 0 | Basophile | 852 | 7,1 % |
| 1 | Éosinophile | 2 181 | 18,2 % |
| 2 | Érythroblaste | 1 085 | 9,1 % |
| 3 | Granulocytes immatures | 2 026 | 16,9 % |
| 4 | Lymphocyte | 849 | 7,1 % |
| 5 | Monocyte | 993 | 8,3 % |
| 6 | Neutrophile | 2 330 | 19,5 % |
| 7 | Plaquette | 1 643 | 13,7 % |
| | **Total train** | **11 959** | **100 %** |

Le dataset complet compte 17 092 images, réparties selon un ratio 7:1:2 entre
entraînement, validation et test — soit 11 959 images d'entraînement et 3 421 de test.

### Format technique

| Caractéristique | Valeur |
|-----------------|--------|
| Dimensions | 28 × 28 pixels |
| Canaux | 3 (RVB) |
| Type | `uint8`, valeurs de 0 à 255 |
| Features par image après aplatissement | 2 352 |
| Format des labels | tableau `(n, 1)`, entiers 0–7 |
| Fichiers | 6 fichiers `.npy` (images et labels × train/val/test) |

Les images sources, initialement en 360×363 pixels, ont été recadrées au centre en
200×200 puis réduites en 28×28 par les auteurs du dataset.

---

## 2. Justification du choix

### 2.1 Compatibilité avec les contraintes du projet

Le sujet limite les outils autorisés à Scikit-learn, Pandas, Matplotlib, Numpy et
Jupyterlab. L'absence de TensorFlow, PyTorch et Keras exclut les réseaux de neurones
convolutifs, qui constituent pourtant l'approche standard en classification d'images.

Nous sommes donc contraints d'aplatir chaque image en un vecteur de pixels et de recourir
à des algorithmes classiques (régression logistique, SVM, Random Forest, k plus proches
voisins), éventuellement précédés d'une réduction de dimension.

Cette contrainte impose une résolution faible. En 28×28×3, une image représente 2 352
features, ce qui reste traitable sur une machine sans GPU. À titre de comparaison, un
dataset en 224×224 couleur produirait plus de 150 000 features par image, rendant
l'entraînement d'un SVM impraticable dans le temps imparti.

**BloodMNIST satisfait cette contrainte par construction**, puisque la collection MedMNIST
a précisément été conçue pour permettre l'évaluation d'algorithmes légers.

### 2.2 Conformité aux critères imposés par le sujet

| Critère du sujet | Vérification |
|------------------|--------------|
| Contient uniquement images et labels | Oui — six tableaux numpy, aucune métadonnée patient |
| Représentatif d'un problème de classification | Oui — 8 classes discrètes, multi-classes |
| Groupe à l'aise avec la thématique | Oui — imagerie cellulaire neutre, sans contenu pathologique éprouvant |
| Justification distincte des autres groupes | Voir 2.5 |

### 2.3 Un déséquilibre pédagogiquement pertinent

Le sujet consacre une section entière au problème des données déséquilibrées, en
rappelant qu'un jeu composé à 90 % de cas positifs rend un classifieur trivial précis
à 90 %.

BloodMNIST présente un déséquilibre **modéré** : la classe majoritaire (neutrophile,
19,5 %) est 2,7 fois plus représentée que les classes minoritaires (basophile et
lymphocyte, 7,1 % chacune).

Ce rapport est un bon compromis. Il est suffisamment marqué pour que l'accuracy globale
devienne trompeuse — un modèle peut négliger entièrement les classes minoritaires sans
que son score s'effondre — ce qui nous oblige à :

- privilégier le **F1-score macro**, qui pondère chaque classe également quelle que soit
  sa taille ;
- tester le paramètre `class_weight='balanced'` sur les modèles qui le supportent ;
- analyser la matrice de confusion classe par classe plutôt qu'un score agrégé.

Il reste néanmoins assez faible pour ne pas rendre l'apprentissage des classes
minoritaires impossible, contrairement à un dataset où une classe représenterait 2 %
des données.

### 2.4 Une matrice de confusion exploitable

Le passage à 8 classes, plutôt qu'un problème binaire, enrichit considérablement
l'analyse des erreurs. Un problème binaire ne produit que quatre cases ; une matrice
8×8 permet d'identifier des schémas de confusion spécifiques et de les confronter à la
réalité biologique.

Nous anticipons notamment une confusion entre les **granulocytes immatures** (classe 3)
et les **neutrophiles** matures (classe 6) : ces deux populations appartiennent au même
continuum de maturation cellulaire, et leur distinction repose sur des différences
morphologiques fines. Une telle confusion, si elle se confirme, sera un résultat
interprétable plutôt qu'un simple échec.

Cette capacité à commenter les erreurs du modèle répond directement à l'exigence du sujet
d'une « analyse en profondeur de toutes vos décisions et observations ».

### 2.5 Différenciation

Le sujet interdit explicitement de choisir le même dataset qu'un autre groupe sans
justification. Les datasets de radiographie thoracique (Chest X-Ray Pneumonia et
équivalents) constituent le choix par défaut le plus fréquent en imagerie médicale.

BloodMNIST relève d'une modalité différente — microscopie optique plutôt que radiologie —
et d'un problème multi-classes plutôt que binaire. Ce choix est donc distinctif, et les
arguments ci-dessus le motivent indépendamment de cette considération.

---

## 3. Observations issues de l'exploration

### 3.1 Homogénéité colorimétrique

L'inspection visuelle d'échantillons de chaque classe révèle une colorimétrie quasi
identique d'une classe à l'autre : noyau violet foncé, cytoplasme clair, fond crème
constellé de globules rouges pâles. Cette uniformité s'explique par le protocole de
coloration standardisé appliqué aux frottis.

Les classes se distinguent en revanche par des critères **morphologiques** : forme du
noyau (rond, réniforme, lobé), taille relative de la cellule, et rapport
noyau/cytoplasme.

**Hypothèse de travail :** les trois canaux couleur apportent peu d'information
discriminante, puisque la teinte ne varie pas significativement entre classes. Une
conversion en niveaux de gris ramènerait le nombre de features de 2 352 à 784, soit une
division par trois du coût de calcul.

Cette hypothèse sera testée expérimentalement dans la section de pré-traitement, par
comparaison des performances d'un même modèle entraîné sur les deux représentations.
Le résultat de ce test, quel qu'il soit, sera documenté.

### 3.2 Limite structurelle de résolution

La réduction de 200×200 à 28×28 correspond à une perte d'environ 98 % des pixels
d'origine : chaque pixel final résulte de la moyenne d'une cinquantaine de pixels
sources.

Cette compression fait disparaître les détails de texture et les contours fins,
notamment la lobulation du noyau qui constitue un critère de distinction majeur entre
plusieurs types de granulocytes.

**Conséquence méthodologique :** une partie de l'information discriminante n'est
physiquement plus présente dans les données. Cette limite plafonne la performance
atteignable indépendamment de la qualité du modèle. Nous en tiendrons compte dans
l'interprétation de nos résultats, en distinguant les erreurs imputables au modèle de
celles imputables aux données.

Nous visons en conséquence une performance réaliste, de l'ordre de 70 à 85 % de F1-score
macro, plutôt qu'un score maximal obtenu au prix d'un surajustement.

---

## 4. Positionnement médical et usage responsable

Le sujet insiste sur la nécessité de « tout faire dans les règles » et d'adresser la
question de l'usage responsable de l'IA. Trois précisions s'imposent.

**Notre modèle ne pose aucun diagnostic.** Il effectue une tâche de reconnaissance
morphologique : identifier le type d'une cellule isolée. Aucune image du dataset ne
présente de cellule pathologique.

**Le lien avec la pratique clinique est indirect.** En hématologie, l'analyse d'un frottis
consiste à établir la formule leucocytaire, c'est-à-dire la proportion de chaque type
cellulaire. Ce sont les écarts de ces proportions par rapport aux valeurs de référence
qui orientent le diagnostic — une proportion anormale de granulocytes immatures peut
par exemple constituer un signal d'alerte. Cette interprétation relève du praticien et
se situe hors du périmètre de notre modèle.

**Le modèle n'a aucune connaissance biologique.** Il apprend des corrélations
statistiques entre positions de pixels et étiquettes. Il est donc sensible à toute
variation de protocole d'acquisition — éclairage, coloration, matériel — et ses
prédictions ne sauraient être transposées à des données produites dans d'autres
conditions sans réévaluation.

---

## 5. Source et licence

**Référence :**
Yang, J., Shi, R., Wei, D., Liu, Z., Zhao, L., Ke, B., Pfister, H., Ni, B.
*MedMNIST v2 — A large-scale lightweight benchmark for 2D and 3D biomedical image
classification.* Scientific Data, 2023.
https://doi.org/10.1038/s41597-022-01721-8

**Licences :** données sous CC BY 4.0 (attribution obligatoire), code de MedMNIST sous Apache 2.0.
Source : dépôt officiel `github.com/MedMNIST/MedMNIST`, lu le 2026-10-15 ; BloodMNIST n'y figure pas parmi les exceptions.
Attribution : citer le papier MedMNIST v2 ci-dessus. MedMNIST demande aussi de citer le papier du jeu source de BloodMNIST (référence sur medmnist.com).

**Emplacement dans le dépôt :** `data/bloodmnist/` (six fichiers `.npy`)

---

## 6. Points à vérifier avant la soutenance

- [ ] Confirmer la correspondance numéro ↔ nom de classe sur la fiche officielle du
      dataset (l'ordre alphabétique anglais est la convention, mais doit être vérifié)
- [ ] Compléter la section 3.1 avec le résultat chiffré du test couleur / niveaux de gris
- [ ] Compléter la section 3.2 avec la performance effectivement atteinte
- [ ] Vérifier la distribution des classes sur le jeu de test (cohérence avec le train)
