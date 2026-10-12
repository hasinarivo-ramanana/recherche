# -*- coding: utf-8 -*-
"""
level3_colmap_pnp.py
=====================
NIVEAU 3 : affinement fin de la pose par mise en correspondance 2D-3D avec
le nuage de points COLMAP et resolution PnP guidee par le prior APR
(PG-PnP), cf. Contribution 3 de la these.

Etapes (Algorithme "Selection du candidat optimal / PG-PnP") :
  1.  Extraction SIFT de l'image requete + appariement 2D-3D contre l'index
      de descripteurs construit a partir de database.db / images.txt.
  1b. Filtrage geographique des correspondances (rayon r=60 m autour du
      prior de position, Niveau 1 ou Niveau 2).
  2.  Calcul des poids de coherence avec la pose APR (Eq. poids_ransac).
  2b. RANSAC pondere (tirage selon les poids) -> ensemble d'inliers.
  3.  Affinement non lineaire avec terme de rappel vers la position APR
      (Eq. pgpnp_simple) -> pose finale.
  Repli : si inliers insuffisants, on retourne la pose du Niveau 2 (ou 1).
"""

import csv
import json
import os
import time

import cv2
import numpy as np
from scipy.optimize import least_squares

import config
import colmap_io
import geo_utils
import display_utils as disp


# ==========================================================================
# Georeferencement : alignement COLMAP (repere arbitraire) <-> ENU (metres)
# ==========================================================================
def _read_gps_csv(path):
    """
    Lit un fichier de reference GPS des images d'entrainement COLMAP.
    Deux formats sont acceptes automatiquement :

    (a) CSV avec en-tete, colonnes detectees par nom (insensible a la
        casse) : 'name'/'image'/'image_id'/'image_name', 'latitude'/'lat',
        'longitude'/'lon'/'lng'. Exemple :
            name,latitude,longitude
            img_0001.jpg,-18.914591,47.552270

    (b) Texte sans en-tete, separe par des espaces/tabulations, colonnes
        positionnelles : NOM LAT LON [ALTITUDE]. Exemple :
            img_0335.jpg -18.914827200 47.552101300 0.0

    Retourne une liste de dicts {'name':..., 'lat':..., 'lon':...}
    """
    with open(path, "r", encoding="utf-8") as f:
        raw_lines = [l.rstrip("\n") for l in f if l.strip()]

    if not raw_lines:
        raise ValueError(f"Fichier GPS vide : {path}")

    first = raw_lines[0]
    looks_like_header = ("," in first) and any(
        tok.strip().lower() in ("name", "image", "image_id", "image_name",
                                 "latitude", "lat", "longitude", "lon", "lng")
        for tok in first.split(","))

    rows = []
    if looks_like_header:
        reader = csv.DictReader(raw_lines)
        fieldmap = {k.lower(): k for k in reader.fieldnames}
        name_key = next((fieldmap[k] for k in ("name", "image", "image_id", "image_name")
                          if k in fieldmap), None)
        lat_key = next((fieldmap[k] for k in ("latitude", "lat") if k in fieldmap), None)
        lon_key = next((fieldmap[k] for k in ("longitude", "lon", "lng") if k in fieldmap), None)
        if not (name_key and lat_key and lon_key):
            raise ValueError(f"Colonnes GPS introuvables dans {path} "
                              f"(colonnes trouvees: {reader.fieldnames})")
        for row in reader:
            rows.append({
                "name": str(row[name_key]).strip(),
                "lat": float(row[lat_key]),
                "lon": float(row[lon_key]),
            })
    else:
        # Format sans en-tete, separe par des espaces : NOM LAT LON [ALT]
        # (ex. fichier "reference_gps_colmap.txt" : img_XXXX.jpg lat lon 0.0)
        for i, line in enumerate(raw_lines, start=1):
            parts = line.replace(",", " ").split()
            if len(parts) < 3:
                continue  # ligne mal formee, ignoree
            name = parts[0].strip()
            try:
                lat = float(parts[1])
                lon = float(parts[2])
            except ValueError:
                raise ValueError(
                    f"{path}, ligne {i} : impossible de lire lat/lon dans '{line}'. "
                    "Format attendu : NOM LAT LON [ALTITUDE]")
            rows.append({"name": name, "lat": lat, "lon": lon})
        if not rows:
            raise ValueError(f"Aucune ligne exploitable dans {path} "
                              "(format attendu : NOM LAT LON [ALTITUDE], ou CSV avec en-tete).")

    return rows


def build_or_load_georef_transform(images_dict, gps_csv_path, cache_path, verbose=True):
    """
    Calcule (ou recharge depuis le cache) la transformation de similarite
    permettant de passer du repere COLMAP au repere ENU (metres) ancre sur
    une origine geographique (lat0, lon0).

    Retourne (R, s, t, lat0, lon0) ou None si le georeferencement est
    impossible (pas de CSV GPS fourni).
    """
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        R = np.array(data["R"])
        s = data["s"]
        t = np.array(data["t"])
        lat0, lon0 = data["lat0"], data["lon0"]
        if verbose:
            disp.ok(f"Transformation COLMAP->ENU rechargee depuis le cache ({cache_path}).")
        return R, s, t, lat0, lon0

    if not gps_csv_path or not os.path.exists(gps_csv_path):
        return None

    gps_rows = _read_gps_csv(gps_csv_path)
    gps_by_name = {r["name"]: r for r in gps_rows}

    src_pts, dst_pts, used_names = [], [], []
    lat0 = float(np.mean([r["lat"] for r in gps_rows]))
    lon0 = float(np.mean([r["lon"] for r in gps_rows]))

    for img_id, img in images_dict.items():
        base = os.path.basename(img["name"])
        gps = gps_by_name.get(img["name"]) or gps_by_name.get(base)
        if gps is None:
            continue
        R_cw = geo_utils.quat_to_rotmat(img["qvec"])
        C = geo_utils.camera_center(R_cw, img["tvec"])  # centre camera, repere COLMAP
        x, y = geo_utils.latlon_to_local_xy(gps["lat"], gps["lon"], lat0, lon0)
        src_pts.append(C)
        dst_pts.append([x, y, 0.0])
        used_names.append(img["name"])

    if len(src_pts) < 3:
        if verbose:
            disp.warn(f"Georeferencement impossible : seulement {len(src_pts)} image(s) "
                      "commune(s) entre images.txt et le CSV GPS (3 minimum requises).")
        return None

    R, s, t = geo_utils.umeyama_alignment(np.array(src_pts), np.array(dst_pts))

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump({"R": R.tolist(), "s": s, "t": t.tolist(),
                   "lat0": lat0, "lon0": lon0,
                   "n_images_used": len(used_names)}, f, indent=2)

    if verbose:
        disp.ok(f"Transformation COLMAP->ENU calculee sur {len(used_names)} images "
                 f"communes (echelle s={s:.4f}) et mise en cache.")
    return R, s, t, lat0, lon0


# ==========================================================================
# Index de descripteurs point3D <-> SIFT (construit une fois, mis en cache)
# ==========================================================================
def build_or_load_point_descriptor_index(points3D, images_dict, db_path, cache_path,
                                          verbose=True):
    """
    Associe a chaque point 3D triangule un descripteur SIFT representatif,
    pris depuis la premiere observation de son track (image_id, point2D_idx).

    Retourne (point_ids (N,) int64, xyz (N,3) float64, descriptors (N,128) uint8).
    """
    if os.path.exists(cache_path):
        data = np.load(cache_path)
        if verbose:
            disp.ok(f"Index descripteurs/points3D recharge depuis le cache "
                     f"({len(data['point_ids'])} points).")
        return data["point_ids"], data["xyz"], data["descriptors"]

    t0 = time.perf_counter()
    keypoints_cache = {}
    point_ids, xyz_list, desc_list = [], [], []
    missing = 0

    for pid, pdata in points3D.items():
        found = False
        for (image_id, point2d_idx) in pdata["track"]:
            if image_id not in keypoints_cache:
                kp, desc = colmap_io.read_keypoints_and_descriptors(db_path, image_id)
                keypoints_cache[image_id] = (kp, desc)
            kp, desc = keypoints_cache[image_id]
            if desc is None or point2d_idx >= len(desc):
                continue
            point_ids.append(pid)
            xyz_list.append(pdata["xyz"])
            desc_list.append(desc[point2d_idx])
            found = True
            break
        if not found:
            missing += 1

    point_ids = np.array(point_ids, dtype=np.int64)
    xyz = np.array(xyz_list, dtype=np.float64)
    descriptors = np.array(desc_list, dtype=np.uint8)

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez_compressed(cache_path, point_ids=point_ids, xyz=xyz, descriptors=descriptors)

    if verbose:
        elapsed = time.perf_counter() - t0
        disp.ok(f"Index descripteurs/points3D construit : {len(point_ids)} points "
                 f"({missing} sans descripteur exploitable) en {elapsed:.1f}s. "
                 f"Mis en cache dans {cache_path}.")
    return point_ids, xyz, descriptors


# ==========================================================================
# Extraction / appariement SIFT
# ==========================================================================
def extract_sift(image_path, n_features=config.SIFT_N_FEATURES):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Impossible de lire l'image : {image_path}")
    sift = cv2.SIFT_create(nfeatures=n_features)
    kp, desc = sift.detectAndCompute(img, None)
    pts = np.array([k.pt for k in kp], dtype=np.float64) if kp else np.zeros((0, 2))
    return pts, desc, img.shape[::-1]  # (W,H)


def match_query_to_pointcloud(query_desc, point_ids, cloud_desc, ratio=config.SIFT_RATIO_TEST):
    """
    Appariement des descripteurs de l'image requete contre l'index de
    descripteurs du nuage de points, avec test du ratio de Lowe.
    Retourne une liste de tuples (query_idx, point3D_id, distance).
    """
    if query_desc is None or len(query_desc) == 0 or len(cloud_desc) == 0:
        return []
    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.knnMatch(query_desc.astype(np.float32), cloud_desc.astype(np.float32), k=2)
    good = []
    for m_n in matches:
        if len(m_n) < 2:
            continue
        m, n = m_n
        if m.distance < ratio * n.distance:
            good.append((m.queryIdx, int(point_ids[m.trainIdx]), m.distance))
    return good


# ==========================================================================
# Intrinseques camera
# ==========================================================================
def resolve_intrinsics(cameras_dict, images_dict, img_w, img_h):
    if cameras_dict:
        cam = next(iter(cameras_dict.values()))
        K = colmap_io.camera_to_K(cam)
        # remise a l'echelle si la resolution de la requete differe du modele
        sx = img_w / cam["width"]
        sy = img_h / cam["height"]
        K = K.copy()
        K[0, 0] *= sx
        K[1, 1] *= sy
        K[0, 2] *= sx
        K[1, 2] *= sy
        source = f"cameras.txt (camera_id={next(iter(cameras_dict))}, remis a l'echelle)"
    else:
        f = config.DEFAULT_FOCAL_HEURISTIC_FACTOR * max(img_w, img_h)
        K = np.array([[f, 0, img_w / 2], [0, f, img_h / 2], [0, 0, 1]])
        source = "heuristique par defaut (pas de cameras.txt)"
    return K, source


# ==========================================================================
# PG-PnP : RANSAC pondere + affinement guide
# ==========================================================================
def project_points(K, R, t, pts3d):
    pts_cam = (R @ pts3d.T).T + t
    z = pts_cam[:, 2:3]
    z = np.where(np.abs(z) < 1e-9, 1e-9, z)
    uv = (K[:2, :2] @ (pts_cam[:, :2] / z).T).T + K[:2, 2]
    return uv, pts_cam[:, 2]


def weighted_ransac_pnp(pts2d, pts3d, K, weights, eps_px=config.RANSAC_EPS_PX,
                         n_min=config.RANSAC_N_MIN, p_success=config.RANSAC_P_SUCCESS,
                         max_iters=config.RANSAC_MAX_ITERS, verbose=True):
    """
    RANSAC dont l'echantillonnage des correspondances minimales est pondere
    par `weights` (coherence avec la pose APR). A chaque iteration : tirage
    pondere de n_min correspondances -> EPnP -> comptage des inliers sur
    l'ensemble complet (seuil eps_px). Conserve le meilleur modele.
    """
    n = len(pts2d)
    if n < n_min:
        return None

    w = np.asarray(weights, dtype=np.float64)
    w = np.clip(w, 1e-6, None)
    p = w / w.sum()

    best_inliers = np.array([], dtype=int)
    best_rvec, best_tvec = None, None
    iters_done = 0
    n_needed = max_iters  # sera reduit de facon adaptative

    rng = np.random.default_rng(42)
    dist_coeffs = np.zeros(4)

    while iters_done < min(max_iters, n_needed):
        idx = rng.choice(n, size=n_min, replace=False, p=p)
        obj_pts = pts3d[idx].astype(np.float64)
        img_pts = pts2d[idx].astype(np.float64)
        try:
            ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, dist_coeffs,
                                           flags=cv2.SOLVEPNP_EPNP)
        except cv2.error:
            ok = False
        iters_done += 1
        if not ok:
            continue

        R, _ = cv2.Rodrigues(rvec)
        uv_proj, depth = project_points(K, R, tvec.ravel(), pts3d)
        errs = np.linalg.norm(uv_proj - pts2d, axis=1)
        inliers = np.where((errs <= eps_px) & (depth > 0))[0]

        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_rvec, best_tvec = rvec, tvec
            inlier_ratio = max(len(inliers) / n, 1e-3)
            e_out = 1 - inlier_ratio
            denom = np.log(max(1 - (1 - e_out) ** n_min, 1e-12))
            n_needed = int(np.ceil(np.log(1 - p_success) / denom)) if denom < 0 else max_iters
            n_needed = max(n_needed, 1)

    if best_rvec is None or len(best_inliers) < config.RANSAC_MIN_INLIERS:
        return {
            "success": False, "iters_done": iters_done, "n_needed": n_needed,
            "best_inliers": best_inliers, "n_total": n,
        }

    return {
        "success": True, "rvec": best_rvec, "tvec": best_tvec,
        "inliers": best_inliers, "iters_done": iters_done, "n_needed": n_needed,
        "n_total": n,
    }


def refine_pose_with_apr_prior(pts2d_in, pts3d_in, K, rvec0, tvec0, t_apr_colmap,
                                lam):
    """
    Minimisation non lineaire (Levenberg-Marquardt) de l'erreur de
    reprojection + terme de rappel vers la position APR (Eq. pgpnp_simple) :
        min_{R,t} sum ||x_i - pi(K,R,t,X_i)||^2 + lambda * ||t - t_APR||^2
    Parametrisation : x = [rvec(3), tvec(3)].
    """
    x0 = np.concatenate([rvec0.ravel(), tvec0.ravel()])

    def residuals(x):
        rvec, tvec = x[:3], x[3:]
        R, _ = cv2.Rodrigues(rvec)
        uv_proj, _ = project_points(K, R, tvec, pts3d_in)
        reproj_res = (uv_proj - pts2d_in).ravel()
        reg_res = np.sqrt(lam) * (tvec - t_apr_colmap)
        return np.concatenate([reproj_res, reg_res])

    result = least_squares(residuals, x0, method="lm", max_nfev=200)
    rvec_f, tvec_f = result.x[:3], result.x[3:]
    R_f, _ = cv2.Rodrigues(rvec_f)
    return R_f, tvec_f, result


# ==========================================================================
# Orchestrateur Niveau 3
# ==========================================================================
class ColmapPnPRefiner:
    def __init__(self, points3d_path, images_path, cameras_path, database_path,
                 gps_csv_path=None, cache_dir="cache", verbose=True):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

        disp.step(0, "Chargement de la reconstruction COLMAP...")
        t0 = time.perf_counter()
        self.points3D = colmap_io.read_points3D_text(points3d_path)
        self.images = colmap_io.read_images_text(images_path)
        self.cameras = colmap_io.read_cameras_text(cameras_path)
        disp.ok(f"{len(self.points3D)} points 3D, {len(self.images)} images "
                 f"d'entrainement, {len(self.cameras)} camera(s) "
                 f"({time.perf_counter() - t0:.1f}s).")

        self.point_ids, self.xyz, self.cloud_desc = build_or_load_point_descriptor_index(
            self.points3D, self.images, database_path,
            os.path.join(cache_dir, config.CACHE_POINT_INDEX_FILE), verbose=verbose)

        self.georef = build_or_load_georef_transform(
            self.images, gps_csv_path,
            os.path.join(cache_dir, config.CACHE_TRANSFORM_FILE), verbose=verbose)
        if self.georef is None:
            disp.warn("Aucun georeferencement disponible (CSV GPS absent ou insuffisant). "
                      "Le Niveau 3 fonctionnera dans le repere local COLMAP "
                      "(pas de conversion lat/lon finale).")

        self.database_path = database_path

    # ------------------------------------------------------------------
    def refine(self, image_path, prior_result, prior_level=1, verbose=True):
        """
        prior_result : dict issu du Niveau 1 (lat,lon,quat,yaw_deg,...) ou,
                       si disponible, position affinee du Niveau 2 (dans ce
                       cas fournir un dict compatible avec 'lat','lon' et
                       reutiliser l'orientation du Niveau 1 pour le prior R).
        """
        t_start = time.perf_counter()
        timings = {}

        disp.title("NIVEAU 3 - AFFINEMENT PAR NUAGE DE POINTS COLMAP (PG-PnP)")
        disp.kv("Image requete", image_path)
        disp.kv("Position prior utilisee",
                 f"({prior_result['lat']:.7f}, {prior_result['lon']:.7f}) "
                 f"[Niveau {prior_level}]")

        # ---- Etape 1 : extraction SIFT + appariement 2D-3D --------------
        disp.step(1, "Extraction SIFT et appariement 2D-3D contre le nuage COLMAP")
        t0 = time.perf_counter()
        pts_q, desc_q, (img_w, img_h) = extract_sift(image_path)
        matches = match_query_to_pointcloud(desc_q, self.point_ids, self.cloud_desc)
        timings["sift_matching_s"] = time.perf_counter() - t0

        disp.kv("Keypoints SIFT extraits (image requete)", len(pts_q))
        disp.kv("Points dans le nuage COLMAP (total)", len(self.point_ids))
        disp.kv("Correspondances 2D-3D brutes (ratio de Lowe)", len(matches))
        disp.kv("Temps extraction + appariement", f"{timings['sift_matching_s']*1000:.0f}", "ms")

        if len(matches) < config.RANSAC_MIN_INLIERS:
            return self._fallback(prior_result, prior_level, timings, t_start,
                                   reason=f"seulement {len(matches)} correspondances 2D-3D brutes "
                                          f"(minimum requis: {config.RANSAC_MIN_INLIERS}).")

        # index -> tableaux numpy pour la suite
        q_idx = np.array([m[0] for m in matches])
        pid_matched = np.array([m[1] for m in matches])
        id_to_row = {pid: i for i, pid in enumerate(self.point_ids)}
        rows = np.array([id_to_row[p] for p in pid_matched])
        pts2d_all = pts_q[q_idx]
        pts3d_all = self.xyz[rows]

        # ---- Etape 1b : filtrage geographique ---------------------------
        disp.step("1b", f"Filtrage geographique des correspondances "
                         f"(rayon r={config.LEVEL3_GEO_FILTER_RADIUS_M:.0f} m)")
        t0 = time.perf_counter()

        if self.georef is not None:
            R_t, s_t, t_t, lat0, lon0 = self.georef
            x_apr, y_apr = geo_utils.latlon_to_local_xy(
                prior_result["lat"], prior_result["lon"], lat0, lon0)
            apr_enu = np.array([x_apr, y_apr, 0.0])
            apr_colmap = geo_utils.invert_similarity(R_t, s_t, t_t, apr_enu)
            radius_colmap = config.LEVEL3_GEO_FILTER_RADIUS_M / s_t
        else:
            apr_colmap = self.xyz.mean(axis=0)
            radius_colmap = config.LEVEL3_GEO_FILTER_RADIUS_M  # hypothese echelle=1

        d_to_apr = np.linalg.norm(pts3d_all - apr_colmap, axis=1)
        geo_mask = d_to_apr <= radius_colmap
        n_before = len(pts2d_all)
        pts2d = pts2d_all[geo_mask]
        pts3d = pts3d_all[geo_mask]
        outlier_rate_before = 1.0  # tous les points du nuage, par def. hors zone au depart
        n_after = len(pts2d)
        timings["geo_filter_s"] = time.perf_counter() - t0

        disp.kv("Correspondances avant filtrage", n_before)
        disp.kv("Correspondances apres filtrage geographique", n_after)
        disp.kv("Taux de rejet geographique",
                 f"{100*(1 - n_after/max(n_before,1)):.1f}", "%")
        disp.kv("Position APR dans le repere COLMAP",
                 f"({apr_colmap[0]:.2f}, {apr_colmap[1]:.2f}, {apr_colmap[2]:.2f})")
        disp.kv("Rayon de filtrage (repere COLMAP)", f"{radius_colmap:.4f}")

        if n_after < config.RANSAC_MIN_INLIERS:
            return self._fallback(prior_result, prior_level, timings, t_start,
                                   reason=f"seulement {n_after} correspondances apres filtrage "
                                          f"geographique (minimum requis: "
                                          f"{config.RANSAC_MIN_INLIERS}).")

        # ---- Intrinseques camera -----------------------------------------
        K, k_source = resolve_intrinsics(self.cameras, self.images, img_w, img_h)
        disp.kv("Matrice intrinseque K", f"fx={K[0,0]:.1f}, fy={K[1,1]:.1f}, "
                 f"cx={K[0,2]:.1f}, cy={K[1,2]:.1f}")
        disp.kv("Source des intrinseques", k_source)

        # ---- Etape 2 : poids de coherence avec la pose APR ---------------
        disp.step(2, "Calcul des poids de coherence avec la pose APR")
        t0 = time.perf_counter()

        if self.georef is not None:
            R_apr_world = geo_utils.quat_to_rotmat(prior_result["quat"])
            R_apr_colmap = R_t.T @ R_apr_world
        else:
            R_apr_colmap = geo_utils.quat_to_rotmat(prior_result["quat"])
        t_apr_colmap_cam = -R_apr_colmap @ apr_colmap  # convention monde->camera

        uv_apr_proj, _ = project_points(K, R_apr_colmap, t_apr_colmap_cam, pts3d)
        residuals_apr = np.linalg.norm(uv_apr_proj - pts2d, axis=1)

        f_mean = (K[0, 0] + K[1, 1]) / 2.0
        d_bar = 0.7 * config.LEVEL3_GEO_FILTER_RADIUS_M
        sigma_pi = (f_mean * config.SIGMA_T_APR_M) / d_bar
        weights = np.exp(-(residuals_apr ** 2) / (2 * sigma_pi ** 2))
        timings["weighting_s"] = time.perf_counter() - t0

        disp.kv("Sigma_pi (tolerance de pondération, Eq. sigma_pi_adaptatif)",
                 f"{sigma_pi:.1f}", "px")
        disp.kv("Residu moyen sous la pose APR", f"{residuals_apr.mean():.1f}", "px")
        disp.kv("Residu median sous la pose APR", f"{np.median(residuals_apr):.1f}", "px")
        disp.kv("Poids moyen", f"{weights.mean():.4f}")

        # ---- Etape 2b : RANSAC pondere ------------------------------------
        disp.step("2b", "RANSAC pondere (echantillonnage guide par les poids APR)")
        t0 = time.perf_counter()
        ransac_result = weighted_ransac_pnp(pts2d, pts3d, K, weights)
        timings["ransac_s"] = time.perf_counter() - t0

        if ransac_result is None or not ransac_result.get("success"):
            n_in = 0 if ransac_result is None else len(ransac_result.get("best_inliers", []))
            return self._fallback(prior_result, prior_level, timings, t_start,
                                   reason=f"RANSAC pondere n'a trouve que {n_in} inliers "
                                          f"(minimum requis: {config.RANSAC_MIN_INLIERS}).")

        inliers = ransac_result["inliers"]
        disp.kv("Iterations RANSAC executees", ransac_result["iters_done"])
        disp.kv("Iterations necessaires estimees (adaptatif, Eq. ransac_iter)",
                 ransac_result["n_needed"])
        disp.kv("Correspondances totales testees", ransac_result["n_total"])
        disp.kv("Inliers retenus", f"{len(inliers)} "
                 f"({100*len(inliers)/ransac_result['n_total']:.1f} %)")
        disp.kv("Temps RANSAC", f"{timings['ransac_s']*1000:.0f}", "ms")

        # ---- Etape 3 : affinement guide par le prior APR ------------------
        disp.step(3, "Affinement non lineaire (Levenberg-Marquardt) avec rappel vers l'APR")
        t0 = time.perf_counter()

        pts2d_in = pts2d[inliers]
        pts3d_in = pts3d[inliers]

        R0, _ = cv2.Rodrigues(ransac_result["rvec"])
        uv0, _ = project_points(K, R0, ransac_result["tvec"].ravel(), pts3d_in)
        reproj_err_before = np.linalg.norm(uv0 - pts2d_in, axis=1)
        e_reproj_mean = reproj_err_before.mean()
        lam = (e_reproj_mean ** 2) / (config.SIGMA_T_APR_M ** 2)

        R_final, t_final, opt_result = refine_pose_with_apr_prior(
            pts2d_in, pts3d_in, K, ransac_result["rvec"], ransac_result["tvec"],
            t_apr_colmap_cam, lam)

        uv_final, _ = project_points(K, R_final, t_final, pts3d_in)
        reproj_err_after = np.linalg.norm(uv_final - pts2d_in, axis=1)
        timings["refine_s"] = time.perf_counter() - t0

        disp.kv("Erreur de reprojection moyenne (avant affinement)",
                 f"{e_reproj_mean:.2f}", "px")
        disp.kv("Erreur de reprojection moyenne (apres affinement)",
                 f"{reproj_err_after.mean():.2f}", "px")
        disp.kv("Erreur de reprojection mediane (apres affinement)",
                 f"{np.median(reproj_err_after):.2f}", "px")
        disp.kv("Lambda (poids du terme de rappel APR, Eq. lambda)", f"{lam:.4f}")
        disp.kv("Convergence de l'optimiseur", str(opt_result.success))
        disp.kv("Temps d'affinement", f"{timings['refine_s']*1000:.0f}", "ms")

        # ---- Pose finale : conversion vers lat/lon ------------------------
        disp.step(4, "Calcul de la pose finale et conversion geographique")
        C_colmap = geo_utils.camera_center(R_final, t_final)
        yaw_f, pitch_f, roll_f = geo_utils.quat_to_euler_deg(
            geo_utils.rotmat_to_quat(R_final.T))  # camera->monde pour l'affichage

        result = {
            "success": True,
            "n_matches_raw": len(matches),
            "n_matches_geo_filtered": n_after,
            "n_inliers": len(inliers),
            "inlier_ratio": len(inliers) / ransac_result["n_total"],
            "reproj_error_before_px": float(e_reproj_mean),
            "reproj_error_after_px": float(reproj_err_after.mean()),
            "lambda": float(lam),
            "sigma_pi_px": float(sigma_pi),
            "ransac_iterations": ransac_result["iters_done"],
            "ransac_iterations_needed": ransac_result["n_needed"],
            "camera_center_colmap": C_colmap,
            "yaw_deg": yaw_f, "pitch_deg": pitch_f, "roll_deg": roll_f,
            "timings": timings,
            "target_error_m": config.LEVEL3_TARGET_ERROR_M,
        }

        if self.georef is not None:
            R_t, s_t, t_t, lat0, lon0 = self.georef
            C_enu = geo_utils.apply_similarity(R_t, s_t, t_t, C_colmap)
            lat_f, lon_f = geo_utils.local_xy_to_latlon(C_enu[0], C_enu[1], lat0, lon0)
            result["lat"] = lat_f
            result["lon"] = lon_f
            result["altitude_m"] = C_enu[2]
            result["displacement_from_prior_m"] = geo_utils.haversine(
                prior_result["lat"], prior_result["lon"], lat_f, lon_f)
        else:
            result["lat"] = None
            result["lon"] = None
            result["local_xyz_colmap"] = C_colmap

        result["elapsed_total_s"] = time.perf_counter() - t_start

        if verbose:
            self._display_final(result)
        return result

    # ------------------------------------------------------------------
    def _fallback(self, prior_result, prior_level, timings, t_start, reason):
        disp.warn(f"Repli sur la pose du Niveau {prior_level} : {reason}")
        result = {
            "success": False,
            "reason": reason,
            "lat": prior_result["lat"],
            "lon": prior_result["lon"],
            "fallback_level": prior_level,
            "timings": timings,
            "elapsed_total_s": time.perf_counter() - t_start,
        }
        disp.section("Resultat Niveau 3 (repli)")
        disp.kv("Position finale (= position de repli)",
                 f"({result['lat']:.7f}, {result['lon']:.7f})")
        disp.kv("Temps total Niveau 3", f"{result['elapsed_total_s']:.2f}", "s")
        print()
        return result

    # ------------------------------------------------------------------
    @staticmethod
    def _display_final(r):
        disp.section("RESULTAT FINAL - Niveau 3 (PG-PnP)")
        disp.kv("Correspondances 2D-3D brutes", r["n_matches_raw"])
        disp.kv("Correspondances apres filtrage geographique", r["n_matches_geo_filtered"])
        disp.kv("Inliers RANSAC finaux", r["n_inliers"])
        disp.kv("Taux d'inliers", f"{100*r['inlier_ratio']:.1f}", "%")
        disp.kv("Iterations RANSAC (executees / estimees necessaires)",
                 f"{r['ransac_iterations']} / {r['ransac_iterations_needed']}")
        disp.kv("Erreur de reprojection finale (moyenne)",
                 f"{r['reproj_error_after_px']:.2f}", "px")
        disp.kv("Lambda (rappel APR)", f"{r['lambda']:.4f}")
        disp.kv("Orientation finale (yaw / pitch / roll)",
                 f"{r['yaw_deg']:.2f} / {r['pitch_deg']:.2f} / {r['roll_deg']:.2f}", "deg")

        if r["lat"] is not None:
            disp.kv("Position finale (lat, lon)", f"({r['lat']:.7f}, {r['lon']:.7f})")
            disp.kv("Altitude relative estimee", f"{r.get('altitude_m', 0):.2f}", "m")
            disp.kv("Deplacement par rapport au prior (Niveau 1/2)",
                     f"{r['displacement_from_prior_m']:.2f}", "m")
        else:
            disp.warn("Pas de georeferencement disponible : position finale exprimee "
                      "uniquement dans le repere local COLMAP (voir 'local_xyz_colmap').")
            disp.kv("Position finale (repere COLMAP, X/Y/Z)",
                     f"({r['camera_center_colmap'][0]:.3f}, "
                     f"{r['camera_center_colmap'][1]:.3f}, "
                     f"{r['camera_center_colmap'][2]:.3f})")

        disp.kv("Objectif de precision (design)", f"< {r['target_error_m']:.1f}", "m")

        disp.section("Repartition du temps de calcul (Niveau 3)")
        t = r["timings"]
        rows = [
            ["SIFT + appariement", f"{t.get('sift_matching_s', 0)*1000:.0f} ms"],
            ["Filtrage geographique", f"{t.get('geo_filter_s', 0)*1000:.0f} ms"],
            ["Calcul des poids (prior APR)", f"{t.get('weighting_s', 0)*1000:.0f} ms"],
            ["RANSAC pondere", f"{t.get('ransac_s', 0)*1000:.0f} ms"],
            ["Affinement (Levenberg-Marquardt)", f"{t.get('refine_s', 0)*1000:.0f} ms"],
            ["TOTAL Niveau 3", f"{r['elapsed_total_s']*1000:.0f} ms"],
        ]
        disp.table(["Etape", "Temps"], rows)
        print()
