# DR CVRIE

Projet Epitech (Bachelor 2, IA) : entraîner CVRIE V1, une IA médicale capable de :

1. **Analyser des témoignages de patients** et les regrouper par pathologie/urgence (apprentissage
   non supervisé).
2. **Analyser des images médicales** pour formuler un diagnostic (apprentissage supervisé).

**Outils autorisés** : `scikit-learn`, `pandas`, `matplotlib`, `numpy`, `jupyterlab`. Toute
approche utilisant d'autres bibliothèques est isolée dans le dossier `bonus/`.

## Structure du projet

```
cvrie_project/
├── data/
│   ├── Student_Dataset.csv     # témoignages patients (partie non supervisée)
│   └── bloodmnist/              # images BloodMNIST (partie supervisée)
├── notebooks/
│   ├── unsupervised_learning.ipynb   # clustering des témoignages
│   └── supervised_learning.ipynb     # classification d'images médicales
├── src/
│   ├── main.py                  # point d'entrée en ligne de commande
│   ├── unsupervised_pipeline.py # version script (norme 84) du clustering
│   └── supervised_pipeline.py   # version script (norme 84) de la classification
├── bonus/                       # pistes utilisant des outils hors sujet (nltk, sentence-transformers, ...)
└── requirements.txt
```

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Utilisation

### Notebooks (analyse complète, prétraitement, comparaison de modèles, conclusions)

```bash
jupyter lab
```

Ouvrir `notebooks/unsupervised_learning.ipynb` ou `notebooks/supervised_learning.ipynb`.

### Scripts en ligne de commande

Le clustering des témoignages peut aussi être exécuté directement :

```bash
python3 src/main.py [chemin_dataset.csv] [chemin_sortie.csv]
```

Sans argument, il utilise `data/Student_Dataset.csv` et écrit le résultat dans
`data/clusters_output.csv`. En cas d'erreur, le message est écrit sur la sortie d'erreur et le
programme se termine avec le code 84 (0 sinon), conformément aux consignes du sujet.

## Partie non supervisée

Dataset de témoignages de patients (non labellisé). Pipeline : nettoyage du CSV, prétraitement NLP
(TF-IDF), réduction de dimension (LSA), clustering K-Means, comparaison avec d'autres algorithmes
(clustering hiérarchique, DBSCAN), et validation externe sur un échantillon étiqueté à la main.
Détails complets dans `notebooks/unsupervised_learning.ipynb`.

## Partie supervisée

Dataset d'images médicales BloodMNIST (classification de cellules sanguines). Pipeline : analyse
exploratoire, prétraitement, comparaison de modèles, évaluation finale. Détails complets dans
`notebooks/supervised_learning.ipynb`.

## Bonus

Le dossier `bonus/` contient des pistes explorées avec des outils hors du cadre autorisé par le
sujet (par exemple des embeddings de phrases via `sentence-transformers`), documentées séparément
avec leurs propres résultats et leur propre fichier de dépendances.
