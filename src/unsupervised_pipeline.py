import csv
import re
import sys
from pathlib import Path

import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

N_CLUSTERS = 15
N_NMF_COMPONENTS = 12
MAX_FEATURES = 2000
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

    vectorizer = TfidfVectorizer(
        stop_words="english", max_features=MAX_FEATURES, ngram_range=(1, 2), sublinear_tf=True
    )
    tfidf_matrix = vectorizer.fit_transform(data["cleaned_text"])

    # NMF (factorisation non negative) : sur ce jeu de donnees, separe mieux que la LSA
    nmf = NMF(n_components=N_NMF_COMPONENTS, random_state=42, max_iter=500)
    reduced_matrix_raw = nmf.fit_transform(tfidf_matrix)
    reduced_matrix = normalize(reduced_matrix_raw)  # distance cosinus plutot qu'euclidienne

    # Clustering hierarchique (cosinus) : meilleure validation externe que K-Means a ce niveau de finesse
    clustering = AgglomerativeClustering(n_clusters=N_CLUSTERS, metric="cosine", linkage="average")
    data["cluster"] = clustering.fit_predict(reduced_matrix)

    score = silhouette_score(reduced_matrix, data["cluster"])
    print(f"Temoignages traites : {len(data)}")
    print(f"Score de Silhouette : {score:.4f}")

    # Mots-clefs dominants par cluster : AgglomerativeClustering n'a pas de centres, on moyenne
    # les temoignages de chaque cluster (espace NMF) puis on reprojette vers le TF-IDF
    words = vectorizer.get_feature_names_out()
    for i in range(N_CLUSTERS):
        mask = (data["cluster"] == i).to_numpy()
        centre = reduced_matrix_raw[mask].mean(axis=0)
        centre_tfidf = centre @ nmf.components_
        top_idx = centre_tfidf.argsort()[-5:][::-1]
        top_words = ", ".join(words[j] for j in top_idx)
        print(f"Cluster {i} ({int(mask.sum())} temoignages) : {top_words}")

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
