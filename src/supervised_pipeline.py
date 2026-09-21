#!/usr/bin/env python3
"""CVRIE V1 — partie supervisée.

Classification de cellules sanguines (BloodMNIST, MedMNIST v2) en huit types
morphologiques. Ce script reprend la chaîne retenue dans
`notebooks/supervised_learning.ipynb` :

    normalisation -> aplatissement -> PCA (95 % de variance) -> SVM noyau RBF

Usage :
    python3 src/supervised_pipeline.py
    python3 src/supervised_pipeline.py --model random-forest
    python3 src/supervised_pipeline.py --confusion matrice.png
    python3 src/supervised_pipeline.py --data chemin/vers/bloodmnist

Sortie : code 0 si tout s'est bien passé, 84 en cas d'erreur (message sur stderr).
Outils : numpy, scikit-learn, matplotlib uniquement.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

RANDOM_STATE = 42
EXIT_ERROR = 84

# Correspondance officielle des labels BloodMNIST (documentation MedMNIST v2).
CLASSES = {
    0: "basophile",
    1: "eosinophile",
    2: "erythroblaste",
    3: "granulocytes immatures",
    4: "lymphocyte",
    5: "monocyte",
    6: "neutrophile",
    7: "plaquette",
}
NOMS = [CLASSES[i] for i in range(len(CLASSES))]

SPLITS = ("train", "val", "test")


class PipelineError(Exception):
    """Erreur métier : message affiché sur stderr, sortie en code 84."""


# --------------------------------------------------------------------------
# Chargement
# --------------------------------------------------------------------------
def racine_projet():
    """Racine du dépôt, déduite de l'emplacement de ce script (src/)."""
    return Path(__file__).resolve().parent.parent


def charger(dossier):
    """Charge les six fichiers .npy du dataset et valide leur cohérence."""
    dossier = Path(dossier)
    if not dossier.is_dir():
        raise PipelineError(
            f"dossier de donnees introuvable : {dossier}\n"
            "Placez les fichiers .npy de BloodMNIST dans data/bloodmnist/ "
            "ou indiquez le chemin avec --data."
        )

    donnees = {}
    for split in SPLITS:
        for genre in ("images", "labels"):
            chemin = dossier / f"{split}_{genre}.npy"
            if not chemin.is_file():
                raise PipelineError(f"fichier manquant : {chemin}")
            try:
                donnees[f"{split}_{genre}"] = np.load(chemin)
            except (ValueError, OSError) as err:
                raise PipelineError(f"lecture impossible de {chemin} : {err}") from err

    for split in SPLITS:
        images = donnees[f"{split}_images"]
        labels = donnees[f"{split}_labels"].ravel()
        if images.ndim != 4 or images.shape[1:] != (28, 28, 3):
            raise PipelineError(
                f"forme inattendue pour {split}_images : {images.shape}, "
                "attendu (N, 28, 28, 3)"
            )
        if len(images) != len(labels):
            raise PipelineError(
                f"{split} : {len(images)} images pour {len(labels)} labels"
            )
        donnees[f"{split}_labels"] = labels

    labels_vus = np.unique(donnees["train_labels"])
    inconnus = set(labels_vus) - set(CLASSES)
    if inconnus:
        raise PipelineError(f"labels hors des huit classes attendues : {sorted(inconnus)}")

    return donnees


def pretraiter(images):
    """Normalise les pixels dans [0, 1] et aplatit chaque image en vecteur.

    28 x 28 x 3 -> 2 352 features. L'aplatissement detruit le voisinage entre
    pixels : c'est la limite assumee d'une approche sans reseau convolutif.
    """
    return (images.astype(np.float64) / 255.0).reshape(len(images), -1)


# --------------------------------------------------------------------------
# Modèles
# --------------------------------------------------------------------------
def construire_modele(nom):
    """Renvoie (modele, utilise_pca) pour le nom demande."""
    modeles = {
        "svm": (
            SVC(kernel="rbf", class_weight="balanced", random_state=RANDOM_STATE),
            True,
        ),
        "random-forest": (
            RandomForestClassifier(
                n_estimators=300,
                class_weight="balanced",
                n_jobs=-1,
                random_state=RANDOM_STATE,
            ),
            False,
        ),
        "logreg": (
            LogisticRegression(
                max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
            ),
            False,
        ),
        "knn": (KNeighborsClassifier(n_neighbors=5, n_jobs=-1), True),
        "dummy": (DummyClassifier(strategy="most_frequent"), False),
    }
    if nom not in modeles:
        raise PipelineError(
            f"modele inconnu : {nom} (disponibles : {', '.join(sorted(modeles))})"
        )
    return modeles[nom]


def tracer_confusion(y_vrai, y_pred, chemin, titre):
    """Enregistre la matrice de confusion normalisee par ligne."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = confusion_matrix(y_vrai, y_pred, normalize="true")
    fig, ax = plt.subplots(figsize=(8.5, 7))
    image = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(NOMS)), NOMS, rotation=40, ha="right")
    ax.set_yticks(range(len(NOMS)), NOMS)
    ax.set_xlabel("classe predite")
    ax.set_ylabel("classe reelle")
    for i in range(len(NOMS)):
        for j in range(len(NOMS)):
            if cm[i, j] >= 0.01:
                ax.text(
                    j,
                    i,
                    f"{cm[i, j]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if cm[i, j] > 0.5 else "black",
                )
    fig.colorbar(image, label="proportion de la classe reelle")
    ax.set_title(titre)
    fig.tight_layout()
    try:
        fig.savefig(chemin, dpi=150)
    except OSError as err:
        raise PipelineError(f"ecriture impossible de {chemin} : {err}") from err
    finally:
        plt.close(fig)


# --------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------
def executer(dossier_donnees, nom_modele, chemin_confusion=None, variance=0.95):
    donnees = charger(dossier_donnees)

    y_train = donnees["train_labels"]
    y_val = donnees["val_labels"]
    y_test = donnees["test_labels"]

    print("Jeu de donnees")
    for split in SPLITS:
        print(f"  {split:5} : {len(donnees[f'{split}_images']):6d} images")

    comptes = np.bincount(y_train, minlength=len(CLASSES))
    print("\nRepartition des classes (entrainement)")
    for indice, compte in enumerate(comptes):
        part = 100.0 * compte / len(y_train)
        print(f"  {CLASSES[indice]:<24} {compte:5d}  ({part:4.1f} %)")
    print(f"  rapport majoritaire / minoritaire : {comptes.max() / comptes.min():.1f}")

    X_train = pretraiter(donnees["train_images"])
    X_val = pretraiter(donnees["val_images"])
    X_test = pretraiter(donnees["test_images"])
    print(f"\nPre-traitement : {X_train.shape[1]} features par image")

    modele, utilise_pca = construire_modele(nom_modele)

    if utilise_pca:
        debut = time.time()
        pca = PCA(n_components=variance, random_state=RANDOM_STATE)
        X_train = pca.fit_transform(X_train)
        X_val = pca.transform(X_val)
        X_test = pca.transform(X_test)
        print(
            f"PCA : {pca.n_features_in_} -> {pca.n_components_} composantes "
            f"({pca.explained_variance_ratio_.sum():.1%} de variance, "
            f"{time.time() - debut:.0f} s)"
        )

    print(f"\nEntrainement du modele '{nom_modele}' ...")
    debut = time.time()
    modele.fit(X_train, y_train)
    duree = time.time() - debut
    print(f"  termine en {duree:.0f} s")

    pred_val = modele.predict(X_val)
    print(
        f"\nValidation : accuracy {accuracy_score(y_val, pred_val):.3f} "
        f"| F1 macro {f1_score(y_val, pred_val, average='macro'):.3f}"
    )

    pred_test = modele.predict(X_test)
    acc = accuracy_score(y_test, pred_test)
    f1 = f1_score(y_test, pred_test, average="macro")
    print(f"Test       : accuracy {acc:.3f} | F1 macro {f1:.3f}\n")
    print(classification_report(y_test, pred_test, target_names=NOMS, digits=3))

    if chemin_confusion:
        tracer_confusion(
            y_test, pred_test, chemin_confusion, f"Matrice de confusion — {nom_modele} (test)"
        )
        print(f"Matrice de confusion enregistree : {chemin_confusion}")

    print(
        "\nCe modele identifie un type cellulaire morphologique. Il ne pose aucun "
        "diagnostic :\nle dataset ne contient que des cellules de patients sains, et "
        "ses auteurs precisent\nqu'il n'est pas destine a un usage clinique."
    )
    return acc, f1


def analyser_arguments(argv):
    parseur = argparse.ArgumentParser(
        prog="supervised_pipeline.py",
        description="CVRIE V1 — classification supervisee de cellules sanguines.",
    )
    parseur.add_argument(
        "--data",
        default=str(racine_projet() / "data" / "bloodmnist"),
        help="dossier contenant les six fichiers .npy (defaut : data/bloodmnist)",
    )
    parseur.add_argument(
        "--model",
        default="svm",
        choices=["svm", "random-forest", "logreg", "knn", "dummy"],
        help="modele a entrainer (defaut : svm, le meilleur en validation)",
    )
    parseur.add_argument(
        "--confusion",
        metavar="FICHIER.png",
        help="enregistre la matrice de confusion du jeu de test dans ce fichier",
    )
    parseur.add_argument(
        "--variance",
        type=float,
        default=0.95,
        help="variance conservee par la PCA, entre 0 et 1 (defaut : 0.95)",
    )
    arguments = parseur.parse_args(argv)
    if not 0.0 < arguments.variance <= 1.0:
        parseur.error("--variance doit etre strictement entre 0 et 1")
    return arguments


def main(argv=None):
    try:
        try:
            arguments = analyser_arguments(argv)
        except SystemExit as sortie:
            # argparse quitte en code 2 sur argument invalide et en 0 sur --help.
            return 0 if sortie.code in (0, None) else EXIT_ERROR
        executer(
            arguments.data, arguments.model, arguments.confusion, arguments.variance
        )
    except PipelineError as err:
        print(f"Erreur : {err}", file=sys.stderr)
        return EXIT_ERROR
    except KeyboardInterrupt:
        print("Erreur : interrompu par l'utilisateur.", file=sys.stderr)
        return EXIT_ERROR
    except MemoryError:
        print(
            "Erreur : memoire insuffisante. Reessayez avec --model logreg "
            "ou une --variance plus basse.",
            file=sys.stderr,
        )
        return EXIT_ERROR
    except Exception as err:  # filet de securite : rien ne doit sortir en code 1
        print(f"Erreur inattendue : {type(err).__name__} : {err}", file=sys.stderr)
        return EXIT_ERROR
    return 0


if __name__ == "__main__":
    sys.exit(main())
