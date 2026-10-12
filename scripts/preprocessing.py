"""
Module de prétraitement pour MobileNet-APR-Campus.

Implémente :
  - Conversion des angles d'Euler (azimuth, pitch, roll) en quaternions unitaires
  - Normalisation min-max centrée sur le barycentre pour la position (lat, lon)
  - Chargement et redimensionnement des images (interpolation bilinéaire)
  - Normalisation des images selon les statistiques ImageNet

Conforme aux équations du manuscrit :
  - Éq. norm_image       : normalisation ImageNet par canal
  - Éq. norm_position    : standardisation min-max centrée sur le barycentre
  - Éq. quat_normalize   : contrainte d'unitarité du quaternion (appliquée dans
                            le modèle via une couche L2, pas ici)
"""

import os
import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Statistiques ImageNet (cf. Éq. norm_image du manuscrit)
# ---------------------------------------------------------------------------
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ---------------------------------------------------------------------------
# Conversion Euler -> Quaternion
# ---------------------------------------------------------------------------
def euler_to_quaternion(azimuth, pitch, roll):
    """
    Convertit un triplet d'angles d'Euler (en radians) en quaternion unitaire
    (qw, qx, qy, qz), suivant la convention aéronautique standard
    yaw (azimuth, psi) - pitch (theta) - roll (phi_r), intrinsèque Z-Y-X.

    Cette convention correspond à celle utilisée dans le manuscrit pour la
    conversion (psi, theta, phi_r) -> quaternion (Éqs. qw--qz).

    Paramètres
    ----------
    azimuth, pitch, roll : array-like, en radians

    Retourne
    --------
    np.ndarray de forme (N, 4) : (qw, qx, qy, qz), unitaire par construction.
    """
    azimuth = np.asarray(azimuth, dtype=np.float64)
    pitch = np.asarray(pitch, dtype=np.float64)
    roll = np.asarray(roll, dtype=np.float64)

    cy = np.cos(azimuth * 0.5)
    sy = np.sin(azimuth * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy

    q = np.stack([qw, qx, qy, qz], axis=-1)
    # Normalisation L2 explicite pour garantir l'unitarité numérique
    # (cf. Éq. quat_normalize), même si elle est déjà ~1 par construction.
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    norm = np.where(norm == 0, 1.0, norm)
    q = q / norm
    return q.astype(np.float32)


# ---------------------------------------------------------------------------
# Normalisation de la position géographique (Éq. norm_position)
# ---------------------------------------------------------------------------
class PositionNormalizer:
    """
    Standardisation min-max centrée sur le barycentre du jeu d'entraînement :

        phi_tilde    = (phi    - mean(phi))    / (max(phi)    - min(phi))
        lambda_tilde = (lambda - mean(lambda)) / (max(lambda) - min(lambda))

    Les paramètres (moyenne, max, min) sont calculés exclusivement sur
    l'ensemble d'entraînement, puis mémorisés pour la dénormalisation.
    """

    def __init__(self):
        self.mean_ = None
        self.min_ = None
        self.max_ = None
        self.range_ = None

    def fit(self, positions):
        """positions : array (N, 2) -> colonnes (latitude, longitude)."""
        positions = np.asarray(positions, dtype=np.float64)
        self.mean_ = positions.mean(axis=0)
        self.min_ = positions.min(axis=0)
        self.max_ = positions.max(axis=0)
        self.range_ = self.max_ - self.min_
        # Évite une division par zéro si une coordonnée est constante.
        self.range_ = np.where(self.range_ == 0, 1.0, self.range_)
        return self

    def transform(self, positions):
        positions = np.asarray(positions, dtype=np.float64)
        return ((positions - self.mean_) / self.range_).astype(np.float32)

    def fit_transform(self, positions):
        self.fit(positions)
        return self.transform(positions)

    def inverse_transform(self, positions_normalized):
        positions_normalized = np.asarray(positions_normalized, dtype=np.float64)
        return positions_normalized * self.range_ + self.mean_

    def get_params(self):
        return {
            "mean": self.mean_.tolist(),
            "min": self.min_.tolist(),
            "max": self.max_.tolist(),
            "range": self.range_.tolist(),
        }


# ---------------------------------------------------------------------------
# Chargement et prétraitement des images
# ---------------------------------------------------------------------------
def load_and_preprocess_image(img_path, target_size=(224, 224)):
    """
    Charge une image, la redimensionne par interpolation bilinéaire à
    target_size, puis la normalise selon les statistiques ImageNet
    (Éq. norm_image).

    L'image source attendue est 1920x1440 (cf. manuscrit), mais la fonction
    fonctionne avec n'importe quelle taille d'entrée.
    """
    img = Image.open(img_path).convert("RGB")
    # Interpolation bilinéaire explicite afin de préserver les structures de
    # haute fréquence (contours de fenêtres, arêtes de bâtiments).
    img = img.resize(target_size, resample=Image.BILINEAR)
    img_array = np.asarray(img, dtype=np.float32) / 255.0
    img_array = (img_array - IMAGENET_MEAN) / IMAGENET_STD
    return img_array.astype(np.float32)


def load_images(image_names, image_dir, target_size=(224, 224)):
    """Charge un lot d'images normalisées ImageNet à partir de leurs noms."""
    images = []
    for img_name in image_names:
        img_path = os.path.join(image_dir, img_name)
        images.append(load_and_preprocess_image(img_path, target_size))
    return np.stack(images, axis=0)
