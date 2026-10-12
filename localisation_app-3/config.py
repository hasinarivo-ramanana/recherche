# -*- coding: utf-8 -*-
"""
config.py
=========
Constantes et chemins par defaut pour le pipeline de localisation
en trois niveaux (APR -> YOLO+SIG -> COLMAP/PG-PnP).

Tous les chemins peuvent etre surchargés en ligne de commande via main.py
(voir `python main.py --help`).
"""

import os

# ----------------------------------------------------------------------
# Chemins par defaut (a adapter / a passer en argument de main.py)
# ----------------------------------------------------------------------
DEFAULT_PATHS = {
    "apr_model":        "models/apr_model.tflite",
    "apr_norm_params":  "models/apr_norm_params.json",
    "yolo_model":       "models/yolo_model.tflite",
    "sig_geojson":      "data/sig_data.geojson",
    "colmap_points3d":  "colmap/points3D.txt",
    "colmap_images":    "colmap/images.txt",
    "colmap_cameras":   "colmap/cameras.txt",       # optionnel
    "colmap_database":  "colmap/database.db",
    "gps_csv":          "colmap/dataset_gps.csv",   # optionnel (georeferencement)
    "cache_dir":        "cache",
}

# ----------------------------------------------------------------------
# NIVEAU 1 - APR (Absolute Pose Regression)
# ----------------------------------------------------------------------
APR_INPUT_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

LEVEL1_SEARCH_RADIUS_M = 100.0   # rayon de la zone candidate (Niveau 1 -> Niveau 2)
LEVEL1_EXPECTED_ERROR_M = 58.64  # erreur mediane mesuree (Tableau tab_result1 de la these)

# ----------------------------------------------------------------------
# NIVEAU 2 - YOLO + SIG
# ----------------------------------------------------------------------
YOLO_INPUT_SIZE = 640
YOLO_CONF_THRESHOLD = 0.25
YOLO_IOU_THRESHOLD = 0.45  # utilise seulement si le modele exporte est "raw" (sans NMS integree)

# Classes du jeu de donnees SIG (6 classes, cf. section 5.3 de la these)
CLASS_NAMES = ["misa", "buvette", "boa", "pharmacie", "bni", "bfv"]

LEVEL2_SIG_QUERY_RADIUS_M = 100.0  # rayon de requete SIG autour de la position APR

# Poids du score composite multi-criteres (Tableau tab:poids)
SCORE_WEIGHTS = {
    "taille": 0.30,
    "angle": 0.35,
    "rapport": 0.20,
    "cote": 0.15,
}
SIGMA_D_M = 15.0          # tolerance distance (Eq. score_taille)
SIGMA_ALPHA_DEG = 20.0    # tolerance angulaire (Eq. score_angle)
SIGMA_R_RATIO = 0.35      # tolerance sur le rapport de distances inter-logos

LEVEL2_EXPECTED_ERROR_M = 9.12  # erreur mediane mesuree (Tableau tab:resultats)

# ----------------------------------------------------------------------
# NIVEAU 3 - COLMAP + PG-PnP
# ----------------------------------------------------------------------
LEVEL3_GEO_FILTER_RADIUS_M = 60.0   # rayon de filtrage geographique (= 3*sigma_t)
SIGMA_T_APR_M = 20.0                # ecart-type de l'erreur de position APR

RANSAC_EPS_PX = 4.0        # seuil de reprojection pour un inlier (pixels)
RANSAC_P_SUCCESS = 0.99    # probabilite de succes visee
RANSAC_N_MIN = 4           # taille de l'echantillon minimal (EPnP)
RANSAC_MAX_ITERS = 2000    # plafond de securite
RANSAC_MIN_INLIERS = 6     # nombre minimal d'inliers pour accepter la solution

SIFT_RATIO_TEST = 0.75     # ratio de Lowe pour le matching de descripteurs
SIFT_N_FEATURES = 4000

LEVEL3_TARGET_ERROR_M = 2.0

# Focale par defaut si aucune info EXIF / cameras.txt n'est disponible
# (heuristique classique : f ~= max(W,H) en pixels pour un smartphone grand angle)
DEFAULT_FOCAL_HEURISTIC_FACTOR = 1.2

CACHE_POINT_INDEX_FILE = "point3d_descriptor_index.npz"
CACHE_TRANSFORM_FILE = "colmap_to_enu_transform.json"


def resolve_path(paths_dict, key, override=None):
    """Retourne le chemin a utiliser, avec priorite a l'override CLI."""
    if override:
        return override
    return paths_dict.get(key)
