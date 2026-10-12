# -*- coding: utf-8 -*-
"""
evaluate_niveau3_pnp.py
========================
Evaluation quantitative du Niveau 3 (affinement par nuage de points COLMAP)
sur un ENSEMBLE d'images de test, en reponse a la remarque du rapporteur :

    "2.2 Comparaison insuffisante des methodes PnP (Chapitre 4, p. 73-76).
     La comparaison entre PnP classique, RANSAC et PG-PnP doit etre
     effectuee sur un ensemble de scenes et non seulement sur un exemple
     illustratif. Presenter au minimum : erreur mediane, P75, P90, taux de
     succes sous 2 metres, taux d'echec, temps de calcul. Stratifier selon
     la difficulte de la scene et la qualite du prior APR."

Ce script reutilise TEL QUEL le code deja valide du projet (config.py,
geo_utils.py, colmap_io.py, level1_apr.py, level3_colmap_pnp.py) : aucune
formule n'est reimplementee, seules les etapes d'orchestration de
level3_colmap_pnp.ColmapPnPRefiner.refine() sont "deroulees" ici afin de
pouvoir executer les TROIS variantes de PnP sur EXACTEMENT les memes
correspondances 2D-3D SIFT (extraites une seule fois par image), ce qui
isole proprement l'effet de chaque strategie de resolution :

  1. "pnp_classique"  : cv2.solvePnPRansac standard sur TOUTES les
                        correspondances brutes (sans filtrage geographique,
                        sans ponderation, sans prior APR). C'est la methode
                        de reference de la litterature (EPnP + RANSAC non
                        guide), correspond a la ligne "PnP classique (sans
                        filtrage)" du Tableau tab:ransac_iterations.
  2. "ransac_geo"     : filtrage geographique (Eq. filtre_geo) + RANSAC
                        (poids uniformes, cf. weighted_ransac_pnp) +
                        affinement L-M SANS terme de rappel (lambda=0).
                        Isole l'apport du seul filtrage geographique.
  3. "pgpnp"          : methode complete proposee dans la these (filtrage
                        + RANSAC pondere par coherence APR + terme de
                        rappel adaptatif, Eq. pgpnp_simple).

Echantillonnage automatique (mode par defaut si un CSV de positions est present)
-------------------------------------------------------------------------------
Si un fichier de positions (colonnes image_name, latitude, longitude -- ex.
`position_angle.csv`) est trouve a cote du script, N images (defaut 136) sont
tirees au hasard (graine fixe --seed, reproductible) parmi celles presentes
dans --images-dir (defaut `images_640x480`) ; leur verite terrain est extraite
du CSV et sauvegardee dans <output-dir>/ground_truth_echantillon.csv.

Colonnes ajoutees aux tableaux (cf. Tableau tab:ransac_iterations) :
  - taux_outliers_e : taux d'outliers EFFECTIF du tirage RANSAC,
        e = 1 - sum(w_i . 1[i inlier]) / sum(w_i)   (mediane sur les images
        localisees). Poids uniformes (w_i=1) pour le PnP classique et le
        RANSAC filtre -> e = 1 - N_inliers/N_total ; poids APR pour PG-PnP,
        ce qui mesure la probabilite qu'un tirage pondere tombe sur un outlier.
  - n_ransac : N = log(1-p) / log(1-(1-e)^n_min), avec p=RANSAC_P_SUCCESS et
        n_min=RANSAC_N_MIN de config.py (Eq. ransac_iter), calcule sur e median.
  - reduction : n_ransac(PnP classique) / n_ransac(variante).

Usage minimal :
    python evaluate_niveau3_pnp.py
    python evaluate_niveau3_pnp.py --images-dir images_640x480 --position-csv position_angle.csv

Au premier lancement, si aucun fichier de verite terrain n'existe, le
script genere un gabarit `images/ground_truth.csv` a completer (colonnes :
image, lat, lon, difficulty [optionnel]) puis s'arrete.

Sorties (dans --output-dir, par defaut "eval_niveau3") :
    details_par_image.csv        -- une ligne par (image x variante)
    resume_global.csv / .md      -- tableau global (les 6 metriques)
    resume_par_qualite_apr.csv   -- stratification par qualite du prior APR
    resume_par_difficulte.csv    -- stratification par difficulte de scene
                                     (si la colonne 'difficulty' est fournie)
    tableaux_latex.tex           -- tableaux prets a coller dans le manuscrit
    boxplot_erreurs.png          -- si --plot et matplotlib disponible
"""

import argparse
import csv
import math
import os
import random
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import cv2
import numpy as np

import config
import geo_utils
import level3_colmap_pnp as l3
import display_utils as disp
from level1_apr import APRLocalizer
from level3_colmap_pnp import ColmapPnPRefiner

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")
VARIANT_NAMES = ["pnp_classique", "ransac_geo", "pgpnp"]
VARIANT_LABELS = {
    "pnp_classique": "PnP classique (sans filtrage)",
    "ransac_geo": "RANSAC + filtrage geo (sans ponderation ni rappel)",
    "pgpnp": "PG-PnP complet (filtrage + ponderation + rappel APR)",
}


# ==========================================================================
# Arguments
# ==========================================================================
def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluation batch Niveau 3 : PnP classique vs RANSAC "
                     "filtre vs PG-PnP, sur un ensemble d'images de test.")
    p.add_argument("--images-dir", default="images_640x480",
                    help="Repertoire contenant les images (defaut: images_640x480)")
    p.add_argument("--ground-truth", default=None,
                    help="CSV verite terrain deja pret (image,lat,lon[,difficulty]). "
                         "Si absent : echantillonnage depuis --position-csv, "
                         "sinon <images-dir>/ground_truth.csv")
    p.add_argument("--position-csv", default=None,
                    help="CSV de positions de toutes les images (image_name,latitude,"
                         "longitude,...) ; defaut : position_angle.csv / "
                         "position_angles.csv detecte automatiquement")
    p.add_argument("--sample-size", type=int, default=136,
                    help="Nombre d'images tirees au hasard (defaut: 136)")
    p.add_argument("--seed", type=int, default=42,
                    help="Graine du tirage aleatoire (reproductibilite)")
    p.add_argument("--exclude-train-images", action="store_true",
                    help="Ecarter du tirage les images qui ont servi a construire "
                         "la reconstruction COLMAP (evite la fuite de donnees)")
    p.add_argument("--output-dir", default="eval_niveau3",
                    help="Repertoire de sortie des resultats (defaut: eval_niveau3)")

    p.add_argument("--apr-model", default=config.DEFAULT_PATHS["apr_model"])
    p.add_argument("--apr-norm", default=config.DEFAULT_PATHS["apr_norm_params"])
    p.add_argument("--colmap-points3d", default=config.DEFAULT_PATHS["colmap_points3d"])
    p.add_argument("--colmap-images", default=config.DEFAULT_PATHS["colmap_images"])
    p.add_argument("--colmap-cameras", default=config.DEFAULT_PATHS["colmap_cameras"])
    p.add_argument("--colmap-db", default=config.DEFAULT_PATHS["colmap_database"])
    p.add_argument("--gps-csv", default=config.DEFAULT_PATHS["gps_csv"],
                    help="CSV GPS des images D'ENTRAINEMENT COLMAP (pour le "
                         "georeferencement) -- distinct du --ground-truth "
                         "qui concerne les images de TEST")
    p.add_argument("--cache-dir", default=config.DEFAULT_PATHS["cache_dir"])

    p.add_argument("--max-images", type=int, default=None,
                    help="Limiter le nombre d'images traitees (debogage rapide)")
    p.add_argument("--plot", action="store_true",
                    help="Generer un boxplot des erreurs par variante (matplotlib)")
    return p.parse_args()


# ==========================================================================
# Verite terrain
# ==========================================================================
def write_ground_truth_template(gt_path, images_dir):
    image_files = sorted(f for f in os.listdir(images_dir)
                          if f.lower().endswith(IMAGE_EXTENSIONS))
    with open(gt_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image", "lat", "lon", "difficulty"])
        for name in image_files:
            w.writerow([name, "", "", ""])
    disp.warn(f"Aucun fichier de verite terrain trouve. Un gabarit a ete "
              f"cree : {gt_path}")
    disp.warn("Remplissez les colonnes 'lat' et 'lon' (verite terrain WGS84) "
              "pour chaque image, et optionnellement 'difficulty' "
              "(ex. facile / moyenne / difficile, selon l'ambiguite des "
              "facades visibles), puis relancez ce script.")


def read_ground_truth(gt_path):
    """
    Colonnes attendues (insensibles a la casse) : image, lat, lon,
    difficulty (optionnelle). Retourne dict: nom_image -> {'lat', 'lon',
    'difficulty' (ou None)}.
    """
    gt = {}
    with open(gt_path, "r", encoding="utf-8-sig") as f:
        reader = _dict_reader(f)
        fieldmap = {k.lower().strip(): k for k in reader.fieldnames}
        img_key = fieldmap.get("image") or fieldmap.get("name")
        lat_key = fieldmap.get("lat") or fieldmap.get("latitude")
        lon_key = fieldmap.get("lon") or fieldmap.get("longitude")
        diff_key = fieldmap.get("difficulty") or fieldmap.get("difficulte")
        if not (img_key and lat_key and lon_key):
            raise ValueError(
                f"Colonnes 'image', 'lat', 'lon' introuvables dans {gt_path} "
                f"(colonnes trouvees : {reader.fieldnames})")
        for row in reader:
            name = row[img_key].strip()
            lat_raw = row[lat_key].strip()
            lon_raw = row[lon_key].strip()
            if not name or not lat_raw or not lon_raw:
                continue  # ligne incomplete (gabarit pas encore rempli) -> ignoree
            difficulty = row.get(diff_key, "").strip() if diff_key else ""
            gt[name] = {
                "lat": float(lat_raw),
                "lon": float(lon_raw),
                "difficulty": difficulty or None,
            }
    return gt


def _dict_reader(f):
    """DictReader qui detecte le separateur (',' ';' ou tabulation)."""
    head = f.readline()
    f.seek(0)
    delim = max([",", ";", "\t"], key=head.count)
    return csv.DictReader(f, delimiter=delim)


def find_position_csv(explicit, images_dir):
    candidates = [explicit] if explicit else []
    for name in ("position_angle.csv", "position_angles.csv"):
        candidates += [name, os.path.join(os.path.dirname(os.path.abspath(images_dir)), name)]
    for c in candidates:
        if c and os.path.exists(c):
            return c
    return None


def read_position_csv(path):
    """Retourne dict nom_image -> (lat, lon) depuis un CSV image_name,latitude,longitude."""
    out = {}
    with open(path, "r", encoding="utf-8-sig") as f:
        reader = _dict_reader(f)
        fm = {k.lower().strip(): k for k in reader.fieldnames}
        ik = fm.get("image_name") or fm.get("image") or fm.get("name")
        la = fm.get("latitude") or fm.get("lat")
        lo = fm.get("longitude") or fm.get("lon")
        if not (ik and la and lo):
            raise ValueError(f"Colonnes image_name/latitude/longitude introuvables "
                             f"dans {path} ({reader.fieldnames})")
        for row in reader:
            if row[ik].strip() and row[la].strip() and row[lo].strip():
                out[os.path.basename(row[ik].strip())] = (float(row[la]), float(row[lo]))
    return out


def sample_ground_truth(position_csv, images_dir, sample_size, seed, out_path,
                         exclude_names=None):
    positions = read_position_csv(position_csv)
    on_disk = {n for n in os.listdir(images_dir) if n.lower().endswith(IMAGE_EXTENSIONS)}
    n_no_pos = len([n for n in on_disk if n not in positions])
    eligible = sorted(n for n in on_disk if n in positions
                      and not (exclude_names and n in exclude_names))
    if n_no_pos:
        disp.warn(f"{n_no_pos} image(s) du dossier n'ont pas de position dans "
                  f"{os.path.basename(position_csv)} (non eligibles).")
    if len(eligible) < sample_size:
        disp.warn(f"Seulement {len(eligible)} images eligibles (< {sample_size}) : "
                  f"toutes sont utilisees.")
        sample_size = len(eligible)
    chosen = sorted(random.Random(seed).sample(eligible, sample_size))
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image", "lat", "lon", "difficulty"])
        for n in chosen:
            w.writerow([n, positions[n][0], positions[n][1], ""])
    return {n: {"lat": positions[n][0], "lon": positions[n][1], "difficulty": None}
            for n in chosen}


# ==========================================================================
# Structures de resultats
# ==========================================================================
@dataclass
class VariantOutcome:
    success: bool
    lat: Optional[float] = None
    lon: Optional[float] = None
    error_m: Optional[float] = None
    n_matches_used: int = 0
    n_inliers: int = 0
    elapsed_ms: float = 0.0
    reason: str = ""
    outlier_rate: Optional[float] = None   # e effectif du tirage RANSAC


@dataclass
class ImageRecord:
    image: str
    gt_lat: float
    gt_lon: float
    difficulty: Optional[str]
    apr_lat: float
    apr_lon: float
    apr_error_m: float
    apr_quality: Optional[str] = None
    in_colmap_train: bool = False
    variants: Dict[str, VariantOutcome] = field(default_factory=dict)


# ==========================================================================
# Reprise (sans duplication de logique) des etapes internes du Niveau 3
# ==========================================================================
def _fail(reason, t0, n_matches):
    return VariantOutcome(success=False, reason=reason, n_matches_used=n_matches,
                            elapsed_ms=(time.perf_counter() - t0) * 1000.0)


def compute_apr_colmap_frame(refiner, prior):
    """Reprend exactement la logique des etapes 1b/2 de ColmapPnPRefiner.refine()."""
    if refiner.georef is not None:
        R_t, s_t, t_t, lat0, lon0 = refiner.georef
        x_apr, y_apr = geo_utils.latlon_to_local_xy(prior["lat"], prior["lon"], lat0, lon0)
        apr_enu = np.array([x_apr, y_apr, 0.0])
        apr_colmap = geo_utils.invert_similarity(R_t, s_t, t_t, apr_enu)
        radius_colmap = config.LEVEL3_GEO_FILTER_RADIUS_M / s_t
        R_apr_world = geo_utils.quat_to_rotmat(prior["quat"])
        R_apr_colmap = R_t.T @ R_apr_world
    else:
        apr_colmap = refiner.xyz.mean(axis=0)
        radius_colmap = config.LEVEL3_GEO_FILTER_RADIUS_M
        R_apr_colmap = geo_utils.quat_to_rotmat(prior["quat"])
    t_apr_colmap_cam = -R_apr_colmap @ apr_colmap
    return apr_colmap, radius_colmap, R_apr_colmap, t_apr_colmap_cam


def colmap_center_to_latlon(refiner, C_colmap):
    if refiner.georef is None or C_colmap is None:
        return None, None
    R_t, s_t, t_t, lat0, lon0 = refiner.georef
    C_enu = geo_utils.apply_similarity(R_t, s_t, t_t, C_colmap)
    lat, lon = geo_utils.local_xy_to_latlon(C_enu[0], C_enu[1], lat0, lon0)
    return lat, lon


def run_pnp_classique(pts2d_all, pts3d_all, K):
    """Baseline litterature : PnP+RANSAC standard, sans filtrage ni prior APR."""
    t0 = time.perf_counter()
    if len(pts2d_all) < config.RANSAC_N_MIN:
        return _fail(f"moins de {config.RANSAC_N_MIN} correspondances brutes",
                     t0, len(pts2d_all))
    dist_coeffs = np.zeros(4)
    try:
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            pts3d_all.astype(np.float64), pts2d_all.astype(np.float64), K, dist_coeffs,
            iterationsCount=config.RANSAC_MAX_ITERS,
            reprojectionError=config.RANSAC_EPS_PX,
            confidence=config.RANSAC_P_SUCCESS,
            flags=cv2.SOLVEPNP_EPNP)
    except cv2.error:
        ok, inliers = False, None
    if not ok or inliers is None or len(inliers) < config.RANSAC_MIN_INLIERS:
        n_in = 0 if inliers is None else len(inliers)
        return _fail(f"PnP+RANSAC standard : seulement {n_in} inliers "
                     f"(minimum {config.RANSAC_MIN_INLIERS})", t0, len(pts2d_all))
    R, _ = cv2.Rodrigues(rvec)
    C = geo_utils.camera_center(R, tvec.ravel())
    return VariantOutcome(success=True, n_matches_used=len(pts2d_all),
                            n_inliers=int(len(inliers)),
                            outlier_rate=1.0 - len(inliers) / len(pts2d_all),
                            elapsed_ms=(time.perf_counter() - t0) * 1000.0), C


def run_ransac_geo(pts2d_geo, pts3d_geo, K, t_apr_colmap_cam):
    """Filtrage geographique + RANSAC a poids uniformes + LM sans rappel (lambda=0)."""
    t0 = time.perf_counter()
    if len(pts2d_geo) < config.RANSAC_N_MIN:
        return _fail("moins de correspondances apres filtrage geographique",
                     t0, len(pts2d_geo))
    weights = np.ones(len(pts2d_geo))
    ransac_result = l3.weighted_ransac_pnp(pts2d_geo, pts3d_geo, K, weights)
    if ransac_result is None or not ransac_result.get("success"):
        n_in = 0 if ransac_result is None else len(ransac_result.get("best_inliers", []))
        return _fail(f"RANSAC (filtrage geo, poids uniformes) : {n_in} inliers "
                     f"(minimum {config.RANSAC_MIN_INLIERS})", t0, len(pts2d_geo))
    inliers = ransac_result["inliers"]
    pts2d_in, pts3d_in = pts2d_geo[inliers], pts3d_geo[inliers]
    R_f, t_f, _ = l3.refine_pose_with_apr_prior(
        pts2d_in, pts3d_in, K, ransac_result["rvec"], ransac_result["tvec"],
        t_apr_colmap_cam, lam=0.0)
    C = geo_utils.camera_center(R_f, t_f)
    return VariantOutcome(success=True, n_matches_used=len(pts2d_geo),
                            n_inliers=int(len(inliers)),
                            outlier_rate=1.0 - len(inliers) / len(pts2d_geo),
                            elapsed_ms=(time.perf_counter() - t0) * 1000.0), C


def run_pgpnp(pts2d_geo, pts3d_geo, K, R_apr_colmap, t_apr_colmap_cam):
    """Methode complete PG-PnP (Contribution 3 de la these)."""
    t0 = time.perf_counter()
    if len(pts2d_geo) < config.RANSAC_N_MIN:
        return _fail("moins de correspondances apres filtrage geographique",
                     t0, len(pts2d_geo))
    uv_apr_proj, _ = l3.project_points(K, R_apr_colmap, t_apr_colmap_cam, pts3d_geo)
    residuals_apr = np.linalg.norm(uv_apr_proj - pts2d_geo, axis=1)
    f_mean = (K[0, 0] + K[1, 1]) / 2.0
    d_bar = 0.7 * config.LEVEL3_GEO_FILTER_RADIUS_M
    sigma_pi = (f_mean * config.SIGMA_T_APR_M) / d_bar
    weights = np.exp(-(residuals_apr ** 2) / (2 * sigma_pi ** 2))

    ransac_result = l3.weighted_ransac_pnp(pts2d_geo, pts3d_geo, K, weights)
    if ransac_result is None or not ransac_result.get("success"):
        n_in = 0 if ransac_result is None else len(ransac_result.get("best_inliers", []))
        return _fail(f"PG-PnP : RANSAC pondere n'a trouve que {n_in} inliers "
                     f"(minimum {config.RANSAC_MIN_INLIERS})", t0, len(pts2d_geo))
    inliers = ransac_result["inliers"]
    pts2d_in, pts3d_in = pts2d_geo[inliers], pts3d_geo[inliers]

    R0, _ = cv2.Rodrigues(ransac_result["rvec"])
    uv0, _ = l3.project_points(K, R0, ransac_result["tvec"].ravel(), pts3d_in)
    e_reproj_mean = np.linalg.norm(uv0 - pts2d_in, axis=1).mean()
    lam = (e_reproj_mean ** 2) / (config.SIGMA_T_APR_M ** 2)

    R_f, t_f, _ = l3.refine_pose_with_apr_prior(
        pts2d_in, pts3d_in, K, ransac_result["rvec"], ransac_result["tvec"],
        t_apr_colmap_cam, lam)
    C = geo_utils.camera_center(R_f, t_f)
    w_sum = float(weights.sum())
    e_eff = (1.0 - float(weights[inliers].sum()) / w_sum) if w_sum > 0 else 1.0
    return VariantOutcome(success=True, n_matches_used=len(pts2d_geo),
                            n_inliers=int(len(inliers)), outlier_rate=e_eff,
                            elapsed_ms=(time.perf_counter() - t0) * 1000.0), C


# ==========================================================================
# Traitement d'une image
# ==========================================================================
def process_image(image_path, gt_entry, apr, refiner, train_names=frozenset()):
    image_name = os.path.basename(image_path)

    # ---- Niveau 1 : prior APR (utilise comme point de depart des 3 variantes) ----
    l1 = apr.predict(image_path, verbose=False)
    apr_error_m = geo_utils.haversine(l1["lat"], l1["lon"], gt_entry["lat"], gt_entry["lon"])
    prior = {"lat": l1["lat"], "lon": l1["lon"], "quat": l1["quat"]}

    record = ImageRecord(
        image=image_name, gt_lat=gt_entry["lat"], gt_lon=gt_entry["lon"],
        difficulty=gt_entry["difficulty"], apr_lat=l1["lat"], apr_lon=l1["lon"],
        apr_error_m=apr_error_m, in_colmap_train=image_name in train_names,
    )

    # ---- SIFT + appariement 2D-3D : UNE SEULE FOIS, partage par les 3 variantes ----
    pts_q, desc_q, (img_w, img_h) = l3.extract_sift(image_path)
    matches = l3.match_query_to_pointcloud(desc_q, refiner.point_ids, refiner.cloud_desc)

    if len(matches) == 0:
        reason = "aucune correspondance 2D-3D SIFT trouvee"
        for vname in VARIANT_NAMES:
            record.variants[vname] = VariantOutcome(success=False, reason=reason)
        return record

    q_idx = np.array([m[0] for m in matches])
    pid_matched = np.array([m[1] for m in matches])
    id_to_row = {pid: i for i, pid in enumerate(refiner.point_ids)}
    rows = np.array([id_to_row[p] for p in pid_matched])
    pts2d_all = pts_q[q_idx]
    pts3d_all = refiner.xyz[rows]

    K, _ = l3.resolve_intrinsics(refiner.cameras, refiner.images, img_w, img_h)

    apr_colmap, radius_colmap, R_apr_colmap, t_apr_colmap_cam = \
        compute_apr_colmap_frame(refiner, prior)
    d_to_apr = np.linalg.norm(pts3d_all - apr_colmap, axis=1)
    geo_mask = d_to_apr <= radius_colmap
    pts2d_geo, pts3d_geo = pts2d_all[geo_mask], pts3d_all[geo_mask]

    # ---- Variante 1 : PnP classique (sans filtrage) ----
    out = run_pnp_classique(pts2d_all, pts3d_all, K)
    outcome, C = out if isinstance(out, tuple) else (out, None)
    if outcome.success:
        lat, lon = colmap_center_to_latlon(refiner, C)
        outcome.lat, outcome.lon = lat, lon
        if lat is not None:
            outcome.error_m = geo_utils.haversine(lat, lon, gt_entry["lat"], gt_entry["lon"])
    record.variants["pnp_classique"] = outcome

    # ---- Variante 2 : RANSAC + filtrage geo (sans ponderation ni rappel) ----
    out = run_ransac_geo(pts2d_geo, pts3d_geo, K, t_apr_colmap_cam)
    outcome, C = out if isinstance(out, tuple) else (out, None)
    if outcome.success:
        lat, lon = colmap_center_to_latlon(refiner, C)
        outcome.lat, outcome.lon = lat, lon
        if lat is not None:
            outcome.error_m = geo_utils.haversine(lat, lon, gt_entry["lat"], gt_entry["lon"])
    record.variants["ransac_geo"] = outcome

    # ---- Variante 3 : PG-PnP complet ----
    out = run_pgpnp(pts2d_geo, pts3d_geo, K, R_apr_colmap, t_apr_colmap_cam)
    outcome, C = out if isinstance(out, tuple) else (out, None)
    if outcome.success:
        lat, lon = colmap_center_to_latlon(refiner, C)
        outcome.lat, outcome.lon = lat, lon
        if lat is not None:
            outcome.error_m = geo_utils.haversine(lat, lon, gt_entry["lat"], gt_entry["lon"])
    record.variants["pgpnp"] = outcome

    return record


# ==========================================================================
# Agregation / statistiques
# ==========================================================================
def aggregate(records: List[ImageRecord], variant: str, filter_fn=None):
    sub = [r for r in records if filter_fn is None or filter_fn(r)]
    total = len(sub)
    outcomes = [r.variants[variant] for r in sub if variant in r.variants]
    successes = [o for o in outcomes if o.success]
    errors = [o.error_m for o in successes if o.error_m is not None]
    times = [o.elapsed_ms for o in outcomes]
    n_fail = total - len(successes)
    n_under_target = sum(1 for e in errors if e <= config.LEVEL3_TARGET_ERROR_M)
    return {
        "variante": VARIANT_LABELS[variant],
        "n_images": total,
        "n_localisees": len(successes),
        "n_echecs": n_fail,
        "taux_echec_pct": round(100.0 * n_fail / total, 1) if total else float("nan"),
        f"taux_succes_{config.LEVEL3_TARGET_ERROR_M:.0f}m_pct":
            round(100.0 * n_under_target / total, 1) if total else float("nan"),
        "erreur_mediane_m": round(float(np.median(errors)), 2) if errors else float("nan"),
        "erreur_p75_m": round(float(np.percentile(errors, 75)), 2) if errors else float("nan"),
        "erreur_p90_m": round(float(np.percentile(errors, 90)), 2) if errors else float("nan"),
        "temps_median_ms": round(float(np.median(times)), 1) if times else float("nan"),
        "taux_outliers_e": _median_outlier_rate(successes),
        "n_ransac": n_ransac_theorique(_median_outlier_rate(successes)),
        "reduction": "--",
    }


def _median_outlier_rate(successes):
    rates = [o.outlier_rate for o in successes if o.outlier_rate is not None]
    return round(float(np.median(rates)), 2) if rates else float("nan")


def n_ransac_theorique(e, p=None, n_min=None):
    """Eq. ransac_iter : N = log(1-p) / log(1-(1-e)^n_min). e deja arrondi (2 dec.)
    pour que le tableau soit verifiable a la main."""
    p = config.RANSAC_P_SUCCESS if p is None else p
    n_min = config.RANSAC_N_MIN if n_min is None else n_min
    if e is None or (isinstance(e, float) and math.isnan(e)):
        return float("nan")
    q = (1.0 - e) ** n_min
    if q <= 0.0:
        return float("inf")
    if q >= 1.0:
        return 1
    return int(math.ceil(math.log(1.0 - p) / math.log(1.0 - q)))


def _format_reduction(n_base, n_var):
    if any(isinstance(v, float) and (math.isnan(v) or math.isinf(v)) for v in (n_base, n_var)):
        return "--"
    ratio = n_base / n_var if n_var else float("nan")
    return f"x{ratio:.0f}" if ratio >= 10 else f"x{ratio:.1f}"


def aggregate_group(records, filter_fn=None):
    """Statistiques des 3 variantes pour un meme sous-ensemble, avec la colonne
    'reduction' = N_RANSAC(PnP classique) / N_RANSAC(variante)."""
    stats = [aggregate(records, v, filter_fn) for v in VARIANT_NAMES]
    n_base = stats[0]["n_ransac"]
    for st in stats[1:]:
        st["reduction"] = _format_reduction(n_base, st["n_ransac"])
    return stats


LATEX_VARIANT_LABELS = {
    "pnp_classique": "PnP classique (sans filtrage)",
    "ransac_geo": "+ Filtrage géographique seul",
    "pgpnp": "+ Filtrage + pondération (PG-PnP)",
}
LATEX_LABEL_BY_FULL = {VARIANT_LABELS[k]: LATEX_VARIANT_LABELS[k] for k in VARIANT_NAMES}


def compact_latex_table(stats_list, group_key=None):
    """Colonnes du tableau final : e, N_RANSAC, Reduction + les 6 metriques."""
    succ_key = f"taux_succes_{config.LEVEL3_TARGET_ERROR_M:.0f}m_pct"
    headers = ["Configuration", "Taux outliers $e$", "$N_{\\text{RANSAC}}$", "Réduction",
               "Médiane (m)", "$P_{75}$ (m)", "$P_{90}$ (m)",
               f"Succès $<{config.LEVEL3_TARGET_ERROR_M:.0f}$\\,m (\\%)",
               "Échec (\\%)", "Temps (ms)"]
    if group_key:
        headers = ["Strate"] + headers
    rows = []
    for st in stats_list:
        row = [LATEX_LABEL_BY_FULL.get(st["variante"], st["variante"]),
               st["taux_outliers_e"], st["n_ransac"], st["reduction"].replace("x", "$\\times$"),
               st["erreur_mediane_m"], st["erreur_p75_m"], st["erreur_p90_m"],
               st[succ_key], st["taux_echec_pct"], st["temps_median_ms"]]
        if group_key:
            row = [st[group_key]] + row
        rows.append(row)
    return headers, rows


def apr_quality_buckets(records: List[ImageRecord]):
    errs = np.array([r.apr_error_m for r in records])
    q1, q2 = np.percentile(errs, [33.33, 66.67])
    for r in records:
        if r.apr_error_m <= q1:
            r.apr_quality = "bon prior (tercile 1)"
        elif r.apr_error_m <= q2:
            r.apr_quality = "prior moyen (tercile 2)"
        else:
            r.apr_quality = "prior degrade (tercile 3)"
    return q1, q2


def write_csv(path, headers, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(headers)
        for row in rows:
            w.writerow(row)


def stats_rows(stats_list):
    headers = list(stats_list[0].keys())
    rows = [[s[h] for h in headers] for s in stats_list]
    return headers, rows


def write_markdown_table(f, title, headers, rows):
    f.write(f"### {title}\n\n")
    f.write("| " + " | ".join(str(h) for h in headers) + " |\n")
    f.write("|" + "|".join(["---"] * len(headers)) + "|\n")
    for row in rows:
        f.write("| " + " | ".join(str(v) for v in row) + " |\n")
    f.write("\n")


def write_latex_table(f, caption, label, headers, rows, resize=True):
    ncols = len(headers)
    f.write("\\begin{table}[H]\n\\centering\n")
    f.write(f"\\caption{{{caption}}}\n\\label{{{label}}}\n")
    if resize:
        f.write("\\resizebox{\\textwidth}{!}{%\n")
    f.write("\\begin{tabular}{l" + "c" * (ncols - 1) + "}\n\\toprule\n")
    f.write(" & ".join(str(h) for h in headers) + " \\\\\n\\midrule\n")
    for row in rows:
        cells = [("$\\infty$" if str(v) == "inf" else "--" if str(v) == "nan" else str(v))
                 for v in row]
        f.write(" & ".join(cells) + " \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}%\n")
    if resize:
        f.write("}\n")
    f.write("\\end{table}\n\n")


# ==========================================================================
# Programme principal
# ==========================================================================
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    gt_path = args.ground_truth or os.path.join(args.images_dir, "ground_truth.csv")

    if not os.path.isdir(args.images_dir):
        disp.fail(f"Repertoire d'images introuvable : {args.images_dir}")
        sys.exit(1)

    position_csv = None if args.ground_truth else find_position_csv(args.position_csv, args.images_dir)
    if not position_csv and not os.path.exists(gt_path):
        write_ground_truth_template(gt_path, args.images_dir)
        sys.exit(0)

    disp.title("EVALUATION BATCH NIVEAU 3 - PnP classique vs RANSAC vs PG-PnP", "#")
    disp.kv("Repertoire d'images", args.images_dir)

    disp.section("Chargement des modeles (une seule fois)")
    apr = APRLocalizer(args.apr_model, args.apr_norm)
    refiner = ColmapPnPRefiner(
        args.colmap_points3d, args.colmap_images, args.colmap_cameras,
        args.colmap_db, gps_csv_path=args.gps_csv, cache_dir=args.cache_dir,
        verbose=True,
    )
    # Noms des images ayant servi a construire la reconstruction COLMAP
    train_names = {os.path.basename(im["name"]) for im in refiner.images.values()}

    # ---- Verite terrain : tirage aleatoire OU fichier fourni ----
    if position_csv:
        gt_path = os.path.join(args.output_dir, "ground_truth_echantillon.csv")
        ground_truth = sample_ground_truth(
            position_csv, args.images_dir, args.sample_size, args.seed, gt_path,
            exclude_names=train_names if args.exclude_train_images else None)
        disp.kv("Positions source", position_csv)
        disp.kv("Tirage aleatoire", f"{len(ground_truth)} images (graine {args.seed})")
    else:
        ground_truth = read_ground_truth(gt_path)
        if not ground_truth:
            disp.fail(f"Le fichier {gt_path} ne contient aucune ligne complete "
                      f"(lat/lon manquants). Completez-le puis relancez.")
            sys.exit(1)
    disp.kv("Verite terrain", gt_path)

    image_names = sorted(n for n in os.listdir(args.images_dir)
                          if n.lower().endswith(IMAGE_EXTENSIONS) and n in ground_truth)
    missing = sorted(set(ground_truth) - set(image_names))
    if missing:
        disp.warn(f"{len(missing)} entree(s) de {os.path.basename(gt_path)} "
                  f"ne correspondent a aucune image presente dans "
                  f"{args.images_dir} (ignorees) : {missing[:5]}"
                  f"{'...' if len(missing) > 5 else ''}")
    if args.max_images:
        image_names = image_names[:args.max_images]
    if not image_names:
        disp.fail("Aucune image de test exploitable (voir verite terrain). Arret.")
        sys.exit(1)
    disp.kv("Nombre d'images evaluees", len(image_names))

    n_in_train = sum(1 for n in image_names if n in train_names)
    if n_in_train:
        disp.warn(f"{n_in_train}/{len(image_names)} images evaluees font PARTIE des images "
                  f"d'entrainement de la reconstruction COLMAP : leurs points SIFT sont "
                  f"deja dans le nuage 3D, les erreurs seront donc optimistes (fuite de "
                  f"donnees). Utilisez des images hors reconstruction (ex. dossier test/) "
                  f"pour une evaluation non biaisee.")

    if refiner.georef is None:
        disp.fail("Aucun georeferencement disponible (CSV GPS d'entrainement "
                  "manquant ou insuffisant) : impossible de convertir les poses "
                  "en lat/lon pour les comparer a la verite terrain. "
                  "Fournissez --gps-csv pour continuer.")
        sys.exit(1)

    # ---- Boucle d'evaluation ----
    records: List[ImageRecord] = []
    for i, name in enumerate(image_names, start=1):
        image_path = os.path.join(args.images_dir, name)
        print(f"[{i}/{len(image_names)}] {name} ...", end=" ", flush=True)
        try:
            rec = process_image(image_path, ground_truth[name], apr, refiner, train_names)
            records.append(rec)
            errs = {v: (f"{o.error_m:.1f}m" if o.error_m is not None
                        else ("echec" if not o.success else "n/a"))
                    for v, o in rec.variants.items()}
            print(f"APR={rec.apr_error_m:.1f}m | "
                  f"classique={errs['pnp_classique']} | "
                  f"ransac_geo={errs['ransac_geo']} | pgpnp={errs['pgpnp']}")
        except Exception as e:
            print(f"ECHEC ({e})")
            disp.warn(f"Image {name} ignoree suite a une erreur : {e}")

    if not records:
        disp.fail("Aucune image n'a pu etre traitee. Arret.")
        sys.exit(1)

    q1, q2 = apr_quality_buckets(records)
    disp.kv("Terciles d'erreur APR (bon / moyen / degrade)",
            f"<= {q1:.1f} m / <= {q2:.1f} m / > {q2:.1f} m")

    # ---- Details par image x variante (format long) ----
    detail_headers = ["image", "gt_lat", "gt_lon", "difficulty", "apr_error_m",
                       "apr_quality", "variante", "success", "lat", "lon",
                       "error_m", "n_matches_used", "n_inliers", "outlier_rate_e", "elapsed_ms",
                       "in_colmap_train", "reason"]
    detail_rows = []
    for r in records:
        for vname in VARIANT_NAMES:
            o = r.variants.get(vname)
            if o is None:
                continue
            detail_rows.append([
                r.image, r.gt_lat, r.gt_lon, r.difficulty or "", round(r.apr_error_m, 2),
                r.apr_quality, VARIANT_LABELS[vname], o.success,
                o.lat, o.lon, round(o.error_m, 3) if o.error_m is not None else "",
                o.n_matches_used, o.n_inliers,
                round(o.outlier_rate, 3) if o.outlier_rate is not None else "",
                round(o.elapsed_ms, 1), r.in_colmap_train, o.reason,
            ])
    write_csv(os.path.join(args.output_dir, "details_par_image.csv"),
              detail_headers, detail_rows)

    # ---- Resume global ----
    global_stats = aggregate_group(records)
    headers, rows = stats_rows(global_stats)
    write_csv(os.path.join(args.output_dir, "resume_global.csv"), headers, rows)

    disp.section("RESUME GLOBAL (toutes scenes confondues)")
    disp.table(headers, rows)

    # ---- Stratification par qualite du prior APR ----
    quality_labels = ["bon prior (tercile 1)", "prior moyen (tercile 2)",
                      "prior degrade (tercile 3)"]
    quality_stats = []
    for q in quality_labels:
        for s in aggregate_group(records, filter_fn=lambda r, q=q: r.apr_quality == q):
            quality_stats.append({"qualite_prior_APR": q, **s})
    q_headers, q_rows = stats_rows(quality_stats)
    write_csv(os.path.join(args.output_dir, "resume_par_qualite_apr.csv"), q_headers, q_rows)

    disp.section("STRATIFICATION PAR QUALITE DU PRIOR APR")
    disp.table(q_headers, q_rows)

    # ---- Stratification par difficulte de scene (si fournie) ----
    difficulty_values = sorted({r.difficulty for r in records if r.difficulty})
    difficulty_stats = []
    if difficulty_values:
        for d in difficulty_values:
            for s in aggregate_group(records, filter_fn=lambda r, d=d: r.difficulty == d):
                difficulty_stats.append({"difficulte_scene": d, **s})
        d_headers, d_rows = stats_rows(difficulty_stats)
        write_csv(os.path.join(args.output_dir, "resume_par_difficulte.csv"), d_headers, d_rows)
        disp.section("STRATIFICATION PAR DIFFICULTE DE SCENE")
        disp.table(d_headers, d_rows)
    else:
        disp.warn("Aucune colonne 'difficulty' renseignee dans le fichier de "
                  "verite terrain : la stratification par difficulte de scene "
                  "est omise. Ajoutez une colonne 'difficulty' (ex. facile / "
                  "moyenne / difficile) pour l'obtenir.")

    # ---- Export Markdown (pretes a inclure dans la reponse au rapporteur) ----
    md_path = os.path.join(args.output_dir, "resume.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Evaluation quantitative du Niveau 3 (reponse a la remarque 2.2)\n\n")
        f.write(f"Nombre d'images de test : **{len(records)}** "
                f"(dont {sum(r.in_colmap_train for r in records)} appartenant a la "
                f"reconstruction COLMAP d'entrainement)\n\n")
        f.write("`taux_outliers_e` : taux d'outliers effectif du tirage RANSAC (mediane) ; "
                "`n_ransac` : Eq. ransac_iter avec p="
                f"{config.RANSAC_P_SUCCESS}, n_min={config.RANSAC_N_MIN} ; "
                "`reduction` : N_RANSAC(PnP classique) / N_RANSAC(variante).\n\n")
        write_markdown_table(f, "Resume global (toutes scenes confondues)", headers, rows)
        write_markdown_table(f, "Stratification par qualite du prior APR", q_headers, q_rows)
        if difficulty_values:
            write_markdown_table(f, "Stratification par difficulte de scene", d_headers, d_rows)

    # ---- Export LaTeX (tableaux prets a coller dans le manuscrit) ----
    tex_path = os.path.join(args.output_dir, "tableaux_latex.tex")
    with open(tex_path, "w", encoding="utf-8") as f:
        lh, lr = compact_latex_table(global_stats)
        write_latex_table(
            f, "Comparaison PnP classique / RANSAC filtré / PG-PnP sur "
               f"{len(records)} scènes de test.", "tab:comparaison_pnp_globale", lh, lr)
        lh, lr = compact_latex_table(quality_stats, group_key="qualite_prior_APR")
        write_latex_table(
            f, "Comparaison des méthodes PnP stratifiée par qualité du prior APR.",
            "tab:comparaison_pnp_qualite_apr", lh, lr)
        if difficulty_values:
            lh, lr = compact_latex_table(difficulty_stats, group_key="difficulte_scene")
            write_latex_table(
                f, "Comparaison des méthodes PnP stratifiée par difficulté de scène.",
                "tab:comparaison_pnp_difficulte", lh, lr)

    disp.ok(f"Resultats ecrits dans : {args.output_dir}/ "
            f"(details_par_image.csv, resume_global.csv, resume_par_qualite_apr.csv"
            f"{', resume_par_difficulte.csv' if difficulty_values else ''}, "
            f"resume.md, tableaux_latex.tex)")

    # ---- Boxplot optionnel ----
    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            data, labels = [], []
            for v in VARIANT_NAMES:
                errs = [r.variants[v].error_m for r in records
                        if r.variants[v].success and r.variants[v].error_m is not None]
                data.append(errs)
                labels.append(VARIANT_LABELS[v])

            fig, ax = plt.subplots(figsize=(8, 5))
            ax.boxplot(data, labels=[l.replace(" (", "\n(") for l in labels], showmeans=True)
            ax.axhline(config.LEVEL3_TARGET_ERROR_M, color="red", linestyle="--",
                       linewidth=1, label=f"Objectif {config.LEVEL3_TARGET_ERROR_M:.0f} m")
            ax.set_ylabel("Erreur de position (m)")
            ax.set_title(f"Distribution des erreurs de localisation - Niveau 3 "
                         f"(n={len(records)} scenes)")
            ax.legend()
            fig.tight_layout()
            plot_path = os.path.join(args.output_dir, "boxplot_erreurs.png")
            fig.savefig(plot_path, dpi=150)
            disp.ok(f"Boxplot enregistre : {plot_path}")
        except ImportError:
            disp.warn("matplotlib non disponible : boxplot ignore "
                      "(pip install matplotlib).")


if __name__ == "__main__":
    main()
