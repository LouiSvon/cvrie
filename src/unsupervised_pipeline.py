#!/usr/bin/env python3
"""Pipeline de clustering non supervise des temoignages de symptomes.

Usage:
    python3 src/unsupervised_pipeline.py [dataset.csv] [output.csv]

Reprend la configuration retenue dans notebooks/unsupervised_learning.ipynb :
TF-IDF (unigrammes + bigrammes) -> LSA (TruncatedSVD, 5 composantes) -> K-Means (k=5).
"""

import csv
import re
import sys
from pathlib import Path

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score

N_CLUSTERS = 5
N_LSA_COMPONENTS = 5
DEFAULT_DATASET = Path(__file__).resolve().parent.parent / "data" / "Student_Dataset.csv"


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"fichier introuvable : {path}")

    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for line in reader:
            if not line:
                continue
            # Cas a 3 elements : ID, Couleur, Temoignage
            if len(line) == 3:
                rows.append({"id": line[0], "color": line[1], "text": line[2]})
            # Cas a 2 elements : ID, Temoignage (sans couleur)
            elif len(line) == 2:
                rows.append({"id": line[0], "color": "#000000", "text": line[1]})

    data = pd.DataFrame(rows)
    if data.empty:
        raise ValueError(f"aucune donnee exploitable dans {path}")

    data["text"] = data["text"].fillna("")
    data = data[data["text"].str.strip() != ""]
    if data.empty:
        raise ValueError(f"tous les temoignages de {path} sont vides")

    return data.reset_index(drop=True)


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def run_pipeline(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data["cleaned_text"] = data["text"].apply(clean_text)

    vectorizer = TfidfVectorizer(stop_words="english", max_features=500, ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(data["cleaned_text"])

    svd = TruncatedSVD(n_components=N_LSA_COMPONENTS, random_state=42)
    reduced_matrix = svd.fit_transform(tfidf_matrix)

    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42)
    data["cluster"] = kmeans.fit_predict(reduced_matrix)

    score = silhouette_score(reduced_matrix, data["cluster"])
    print(f"Temoignages traites : {len(data)}")
    print(f"Inertie : {kmeans.inertia_:.2f}")
    print(f"Score de Silhouette : {score:.4f}")

    # Mots-clefs dominants par cluster : reprojection des centres LSA vers l'espace TF-IDF
    words = vectorizer.get_feature_names_out()
    centers_tfidf = kmeans.cluster_centers_ @ svd.components_
    for i in range(N_CLUSTERS):
        top_idx = centers_tfidf[i].argsort()[-5:][::-1]
        top_words = ", ".join(words[j] for j in top_idx)
        size = int((data["cluster"] == i).sum())
        print(f"Cluster {i} ({size} temoignages) : {top_words}")

    return data


def main() -> int:
    dataset_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DATASET
    output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else dataset_path.parent / "clusters_output.csv"

    try:
        data = load_dataset(dataset_path)
        result = run_pipeline(data)
        result[["id", "color", "text", "cluster"]].to_csv(output_path, index=False)
        print(f"Resultats ecrits dans {output_path}")
    except Exception as error:
        print(f"Erreur : {error}", file=sys.stderr)
        return 84

    return 0


if __name__ == "__main__":
    sys.exit(main())
