# DR CVRIE

Projet Epitech (Bachelor 2, IA — B-SPE-311) : entraîner **CVRIE V1**, une IA médicale capable de :

1. **Analyser des témoignages de patients** et les regrouper par pathologie/urgence (apprentissage non supervisé).
2. **Analyser des images médicales** pour formuler un diagnostic (apprentissage supervisé).

**Outils autorisés** : `scikit-learn`, `pandas`, `matplotlib`, `numpy`, `jupyterlab`. Toute approche utilisant d'autres bibliothèques est isolée dans le dossier `bonus/`.

## Structure du projet

```
cvrie_project/
├── data/
│   ├── Student_Dataset.csv        # témoignages patients (partie non supervisée)
│   └── bloodmnist/                 # images BloodMNIST, 6 fichiers .npy (partie supervisée)
├── docs/
│   └── dataset_choice.md           # justification du choix de BloodMNIST
├── notebooks/
│   ├── unsupervised_learning.ipynb # clustering des témoignages
│   └── supervised_learning.ipynb   # classification de cellules sanguines
├── src/
│   ├── main.py                     # point d'entrée en ligne de commande
│   ├── unsupervised_pipeline.py    # version script (norme 84) du clustering
│   └── supervised_pipeline.py      # version script (norme 84) de la classification
├── bonus/                          # pistes utilisant des outils hors sujet (nltk, sentence-transformers)
│   ├── unsupervised_learning_bonus.ipynb
│   └── requirements-bonus.txt
└── requirements.txt
```

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Pour exécuter en plus le notebook bonus (embeddings de phrases, hors outils autorisés) :

```bash
pip install -r bonus/requirements-bonus.txt
```

## Utilisation

### Notebooks (analyse complète : pré-traitement, fonction de perte, comparaison de modèles, conclusions)

```bash
jupyter lab
```

Ouvrir `notebooks/unsupervised_learning.ipynb` ou `notebooks/supervised_learning.ipynb`.

### Scripts en ligne de commande

Les deux pipelines retenus dans les notebooks sont aussi exécutables directement. Dans les deux cas : erreurs écrites sur la sortie d'erreur, code de sortie **84** en cas d'échec, **0** sinon.

```bash
# Clustering des témoignages (TF-IDF + LSA + K-Means)
python3 src/main.py [chemin_dataset.csv] [chemin_sortie.csv]
# sans argument : lit data/Student_Dataset.csv, écrit data/clusters_output.csv

# Classification des cellules sanguines (PCA + SVM RBF)
python3 src/supervised_pipeline.py [--model {svm,random-forest,logreg,knn,dummy}] [--confusion fichier.png] [--variance 0.95] [--data dossier]
```

---

## Partie non supervisée — tri des témoignages patients

**Dataset** : `data/Student_Dataset.csv`, 1011 témoignages en anglais, texte libre, non labellisé. Un chargement naïf (`pandas.read_csv`) n'en récupère que 397 : le nombre de colonnes varie d'une ligne à l'autre (parfois `id, texte`, parfois `id, couleur, texte`) — corrigé par un parsing manuel ligne à ligne. La colonne couleur, toujours `0x000000` dans les données fournies, est volontairement neutre (vérifié empiriquement) : les vraies catégories ne sont révélées qu'en soutenance pour vérifier la cohérence des clusters.

**Pipeline retenu** : nettoyage du texte → TF-IDF (unigrammes + bigrammes, vocabulaire élargi à 2000 termes) → réduction de dimension par LSA (`TruncatedSVD`, 10 composantes, **normalisée** — distance cosinus) → K-Means (k=10).

**Démarche de comparaison** (toutes les approches restent dans les outils autorisés) :

| Approche | Score de silhouette |
| --- | --- |
| TF-IDF brut + K-Means (référence, k=5) | 0,014 |
| + clustering hiérarchique (cosinus, k=5) | ≈ 0,015 — pas d'amélioration |
| + DBSCAN (cosinus) | inexploitable (bruit dominant) |
| + stemming maison (sans dépendance externe, k=5) | 0,023 |
| + LSA 5 composantes (k=5) | 0,41 — mais se dégrade vite si on augmente k sur cette même représentation |
| **+ vocabulaire élargi + LSA 10 composantes normalisée (k=10)** | **0,38, avec deux fois plus de clusters** |

La réduction de dimension par LSA reste le levier le plus efficace, mais demander directement plus de clusters à cette représentation la dégrade (0,41 → 0,20 à k=30). Élargir le vocabulaire et normaliser l'espace LSA (distance cosinus) permet de doubler le nombre de clusters (k=10) en gardant une séparation forte — confirmé meilleur que le clustering hiérarchique, Gaussian Mixture et Birch à k=10 égal. La fonction de perte de K-Means (l'**inertie**, somme des distances au carré aux centroïdes) est expliquée et complétée par le score de silhouette, qui seul permet de choisir un nombre de clusters pertinent.

**Précision clinique gagnée avec 10 clusters** : les thèmes génériques à 5 clusters se scindent en sous-catégories distinctes — par exemple "peau" devient deux clusters séparés (réactions allergiques vs brûlures/lésions), et "respiratoire" se sépare en rhume léger vs détresse respiratoire.

**Validation externe** : 50 témoignages étiquetés à la main (jamais transmis au modèle) comparés aux 10 clusters trouvés — Adjusted Rand Index 0,53, Normalized Mutual Information 0,68. Le tableau croisé confirme la séparation plus fine : la catégorie manuelle "peau" se répartit exactement entre les deux clusters allergies/brûlures identifiés.

**Limite assumée** : ces clusters explorent des *familles de symptômes*, ils ne posent aucun diagnostic — un même symptôme peut relever de pathologies très différentes.

Détails complets, code et interprétations : `notebooks/unsupervised_learning.ipynb`.

## Partie supervisée — classification de cellules sanguines

**Dataset** : [BloodMNIST](https://doi.org/10.1038/s41597-022-01721-8) (collection MedMNIST v2), 17 092 images 28×28 RGB de cellules sanguines individuelles, 8 classes morphologiques (basophile, éosinophile, érythroblaste, granulocytes immatures, lymphocyte, monocyte, neutrophile, plaquette). Choix justifié en détail dans `docs/dataset_choice.md` : léger (compatible avec l'absence de réseaux de neurones), déséquilibre modéré et pédagogique (rapport ≈ 2,7 entre classes), problème multi-classes distinctif.

**Pipeline retenu** : normalisation des pixels → aplatissement (2 352 features) → PCA (95 % de variance conservée → 176 composantes) → SVM à noyau RBF (`class_weight='balanced'`).

**Résultats** : F1 macro **0,844** sur le jeu de test (accuracy 0,861), contre 0,848 en validation — écart minime confirmant l'absence de surapprentissage. Comparé à un Random Forest (0,845, plus lent), une régression logistique (0,808) et un KNN (0,740). La fonction de perte retenue (log loss / entropie croisée pour la régression logistique de référence) est justifiée et comparée à la hinge loss du SVM et au critère de Gini du Random Forest.

**Observations notables** :

- L'hypothèse "les niveaux de gris suffisent" (couleur peu discriminante) est testée puis **réfutée** : -14,4 points de F1 macro en niveaux de gris.
- La confusion dominante n'est pas celle anticipée (granulocytes immatures / neutrophiles) mais entre granulocytes immatures et monocytes — cohérente avec une explication morphologique réelle (continuum de maturation cellulaire).
- Une section dédiée (« Essais infructueux ») documente les pistes qui n'ont pas fonctionné et pourquoi.

**Limite assumée** : le modèle identifie un type cellulaire, il ne pose aucun diagnostic — le dataset ne contient que des cellules de patients sains, et l'aplatissement en pixels plafonne mécaniquement la performance atteignable sans réseau convolutif.

Détails complets, code et interprétations : `notebooks/supervised_learning.ipynb`.

## Bonus

Le dossier `bonus/` explore une piste hors du cadre des outils autorisés par le sujet : des **embeddings de phrases** (`sentence-transformers`, modèle `all-MiniLM-L6-v2`) à la place du TF-IDF pour le clustering des témoignages, avec une lemmatisation `nltk` en complément. Résultat : silhouette 0,087 (contre 0,018 avec lemmatisation seule) — une nette amélioration par rapport au TF-IDF brut, mais qui reste **en dessous** du résultat obtenu dans le notebook principal avec TF-IDF + LSA (0,38, avec deux fois plus de clusters), en restant pourtant dans les outils autorisés. Comparaison et résultats détaillés dans `bonus/unsupervised_learning_bonus.ipynb`.

## Avertissement responsable

Aucun des deux modèles de ce projet ne pose de diagnostic médical. Ils effectuent des tâches ciblées — regroupement de témoignages par similarité, reconnaissance morphologique de cellules isolées — sur des données qui ne contiennent aucune pathologie confirmée. Toute extrapolation vers un usage clinique réel nécessiterait une validation indépendante hors du cadre de ce projet.
