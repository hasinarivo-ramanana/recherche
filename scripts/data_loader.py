"""
Chargement et préparation des données pour MobileNet-APR-Campus.

Deux modes d'entrée sont supportés :

  Mode A — fichier unique combiné (format issu de l'application Android) :
      position_angle.csv  (séparateur ',') contenant :
      image_name, latitude, longitude, azimuth, pitch, roll

  Mode B — deux fichiers séparés (format historique des scripts) :
      positions.csv  (séparateur ';') : image_name, latitude, longitude
      angles.csv     (séparateur ';') : azimuth, pitch, roll

La détection du mode est automatique :
  - si combined_csv est fourni  → Mode A
  - sinon positions_csv + angles_csv sont requis → Mode B

Dans les deux cas, les angles (azimuth, pitch, roll) sont supposés en RADIANS,
conformément à la sortie de SensorManager.getOrientation() d'Android.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from preprocessing import load_images, PositionNormalizer, euler_to_quaternion


def _load_dataframes(positions_csv_path=None, angles_csv_path=None,
                     combined_csv_path=None):
    """
    Charge les DataFrames de position et d'angle selon le mode détecté.

    Retourne (positions_df, angles_df) avec colonnes normalisées :
      positions_df : ['image_name', 'latitude', 'longitude']
      angles_df    : ['azimuth', 'pitch', 'roll']
    """
    if combined_csv_path is not None:
        # ── Mode A : fichier unique ────────────────────────────────────────
        # Détection automatique du séparateur (virgule ou point-virgule)
        with open(combined_csv_path, 'r') as f:
            first_line = f.readline()
        sep = ';' if first_line.count(';') > first_line.count(',') else ','

        df = pd.read_csv(combined_csv_path, sep=sep)
        required = {'image_name', 'latitude', 'longitude',
                    'azimuth', 'pitch', 'roll'}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"Colonnes manquantes dans {combined_csv_path} : {missing}\n"
                f"Colonnes présentes : {list(df.columns)}"
            )
        positions_df = df[['image_name', 'latitude', 'longitude']].copy()
        angles_df    = df[['azimuth', 'pitch', 'roll']].copy()
        print(f"[Mode A] Fichier unique : {combined_csv_path}")
        print(f"         Séparateur détecté : '{sep}'")

    else:
        # ── Mode B : deux fichiers séparés ─────────────────────────────────
        if positions_csv_path is None or angles_csv_path is None:
            raise ValueError(
                "Fournissez soit combined_csv_path (Mode A), "
                "soit positions_csv_path ET angles_csv_path (Mode B)."
            )
        # Détection automatique du séparateur pour chaque fichier
        def _sep(path):
            with open(path, 'r') as f:
                line = f.readline()
            return ';' if line.count(';') >= line.count(',') else ','

        positions_df = pd.read_csv(positions_csv_path, sep=_sep(positions_csv_path))
        angles_df    = pd.read_csv(angles_csv_path,    sep=_sep(angles_csv_path))
        print(f"[Mode B] positions : {positions_csv_path}")
        print(f"         angles    : {angles_csv_path}")

    print(f"         Colonnes positions : {list(positions_df.columns)}")
    print(f"         Colonnes angles    : {list(angles_df.columns)}")

    if len(positions_df) != len(angles_df):
        raise ValueError(
            f"Nombre de lignes incohérent : "
            f"positions={len(positions_df)}, angles={len(angles_df)}"
        )
    return positions_df, angles_df


def load_dataset(
    images_dir,
    positions_csv_path=None,
    angles_csv_path=None,
    combined_csv_path=None,
    image_size=(224, 224),
    test_size=0.2,
    val_size=0.15,
    random_state=42,
):
    """
    Charge le jeu de données complet, effectue la conversion Euler→quaternion
    et la normalisation de la position, puis sépare en train/val/test.

    CORRECTIF (vs version initiale) : la version précédente ne produisait
    qu'un split train/test, et le jeu de test était réutilisé comme
    validation pendant l'entraînement (early stopping, sélection de
    configuration). Cela biaise l'évaluation finale de façon optimiste :
    le jeu de test n'est alors plus "jamais vu". Cette version introduit un
    troisième ensemble, 'val', strictement dédié à l'entraînement
    (early stopping / choix de configuration), le jeu de test restant
    intouché jusqu'à l'évaluation finale.

    Paramètres
    ----------
    images_dir          : chemin vers le dossier contenant les images .jpg
    positions_csv_path  : (Mode B) chemin vers positions.csv
    angles_csv_path     : (Mode B) chemin vers angles.csv
    combined_csv_path   : (Mode A) chemin vers position_angle.csv
                          Si fourni, positions_csv_path et angles_csv_path
                          sont ignorés.
    image_size          : taille de redimensionnement (défaut 224×224)
    test_size           : fraction de l'ensemble de TEST, jamais vue avant
                          l'évaluation finale (défaut 0.2)
    val_size            : fraction de l'ensemble de VALIDATION, utilisée
                          pour l'early stopping / la sélection de config
                          pendant l'entraînement (défaut 0.15, exprimée en
                          fraction du jeu de données TOTAL, pas du reliquat)
    random_state        : graine pour la reproductibilité

    Retourne
    --------
    dict avec les clés :
      'X_train', 'X_val', 'X_test'          : images (N, H, W, 3)
      'pos_train', 'pos_val', 'pos_test'    : positions normalisées min-max (N, 2)
      'quat_train', 'quat_val', 'quat_test' : quaternions unitaires (N, 4)
      'position_normalizer'                 : PositionNormalizer ajusté sur
                                               le train uniquement (à
                                               conserver pour dénormaliser)
    """
    positions_df, angles_df = _load_dataframes(
        positions_csv_path=positions_csv_path,
        angles_csv_path=angles_csv_path,
        combined_csv_path=combined_csv_path,
    )

    image_names = positions_df["image_name"].values
    print(f"\nChargement de {len(image_names)} images depuis {images_dir} ...")
    images = load_images(image_names, images_dir, target_size=image_size)

    # Position brute (degrés décimaux)
    positions = positions_df[["latitude", "longitude"]].values

    # Conversion Euler (radians) → quaternion unitaire
    quaternions = euler_to_quaternion(
        angles_df["azimuth"].values,
        angles_df["pitch"].values,
        angles_df["roll"].values,
    )

    # Split train/val/test AVANT la normalisation de la position, pour que
    # les statistiques (moyenne, min, max) soient calculées exclusivement
    # sur l'ensemble d'entraînement (cf. Éq. norm_position du manuscrit).
    #
    # Étape 1 : isoler le test (jamais utilisé avant l'évaluation finale).
    # Étape 2 : isoler la validation à partir du reliquat (train+val).
    indices = np.arange(len(images))
    trainval_idx, test_idx = train_test_split(
        indices, test_size=test_size, random_state=random_state
    )
    # val_size est exprimé en fraction du jeu TOTAL ; on le convertit en
    # fraction du reliquat (train+val) pour le second split.
    val_fraction_of_trainval = val_size / (1.0 - test_size)
    if not 0.0 < val_fraction_of_trainval < 1.0:
        raise ValueError(
            f"val_size={val_size} incompatible avec test_size={test_size} "
            f"(fraction résultante hors de ]0,1[ : {val_fraction_of_trainval:.3f})"
        )
    train_idx, val_idx = train_test_split(
        trainval_idx, test_size=val_fraction_of_trainval, random_state=random_state
    )

    position_normalizer = PositionNormalizer()
    pos_train = position_normalizer.fit_transform(positions[train_idx])
    pos_val   = position_normalizer.transform(positions[val_idx])
    pos_test  = position_normalizer.transform(positions[test_idx])

    print(
        f"  Train : {len(train_idx)} images  |  "
        f"Val : {len(val_idx)} images  |  "
        f"Test : {len(test_idx)} images"
    )
    print(f"  Barycentre (lat, lon)      : {position_normalizer.mean_}")
    print(f"  Amplitude  (max-min)       : {position_normalizer.range_}")

    return {
        "X_train": images[train_idx],
        "X_val":   images[val_idx],
        "X_test":  images[test_idx],
        "pos_train": pos_train,
        "pos_val":   pos_val,
        "pos_test":  pos_test,
        "quat_train": quaternions[train_idx],
        "quat_val":   quaternions[val_idx],
        "quat_test":  quaternions[test_idx],
        "position_normalizer": position_normalizer,
    }
