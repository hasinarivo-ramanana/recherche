# -*- coding: utf-8 -*-
"""
evaluate_dataset.py
===================
Evalue la cascade de localisation sur un dossier d'images et produit un
tableau comparatif (console + LaTeX + CSV) au format de tab:resultats :

    Methode | e_med (m) | e_75 (m) | tau_10 (%) | tau_5 (%)

Lignes calculees :
  - APR seul                      (Niveau 1)
  - APR + YOLO + Scoring geom.    (Niveau 2 ; repli sur APR si aucun logo/candidat)
  - + PG-PnP                      (Niveau 3 ; optionnel, --with-level3)
  - GPS standard                  (seulement si le CSV contient gps_lat/gps_lon ;
                                   sinon valeurs de reference de la litterature)

Verite terrain (par ordre de priorite) :
  1. --gt-csv : CSV avec colonnes  name,lat,lon  [,heading_deg][,gps_lat,gps_lon]
  2. tags EXIF GPS de l'image (si presents)
Les images sans verite terrain sont ignorees (signale).

Usage :
    python evaluate_dataset.py --images-dir data/test_images --gt-csv data/test_gt.csv
    python evaluate_dataset.py --images-dir data/test_images --with-level3 --out-dir results
"""

import argparse
import csv
import math
import os
import sys
import time

import numpy as np
from PIL import Image

import config
import geo_utils
from level1_apr import APRLocalizer
from level2_yolo_sig import YoloSigRefiner

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


# ----------------------------------------------------------------------
# Utilitaires
# ----------------------------------------------------------------------
def haversine_m(lat1, lon1, lat2, lon2):
    return geo_utils.haversine(lat1, lon1, lat2, lon2)


def exif_gps(path):
    """Lit (lat, lon) depuis l'EXIF, ou None."""
    try:
        exif = Image.open(path).getexif()
        gps = exif.get_ifd(0x8825)
        if not gps or 2 not in gps or 4 not in gps:
            return None

        def dms(v, ref):
            d, m, s = (float(x) for x in v)
            val = d + m / 60.0 + s / 3600.0
            return -val if ref in ("S", "W") else val

        return dms(gps[2], gps.get(1, "N")), dms(gps[4], gps.get(3, "E"))
    except Exception:
        return None


def load_gt_csv(path):
    gt = {}
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        sample = f.read(2048)
        f.seek(0)
        delim = ";" if sample.count(";") > sample.count(",") else ","
        for row in csv.DictReader(f, delimiter=delim):
            row = {k.strip().lower(): (v or "").strip() for k, v in row.items() if k}
            name = os.path.basename(row.get("name") or row.get("image") or row.get("filename", ""))
            if not name:
                continue

            def fl(k):
                try:
                    return float(row[k])
                except (KeyError, ValueError):
                    return None

            gt[name] = {
                "lat": fl("lat"), "lon": fl("lon"),
                "heading": fl("heading_deg") if fl("heading_deg") is not None else fl("heading"),
                "gps_lat": fl("gps_lat"), "gps_lon": fl("gps_lon"),
            }
    return gt


def stats(errors, thr_a=10.0, thr_b=5.0):
    e = np.asarray(errors, dtype=float)
    return {
        "median": float(np.median(e)),
        "p75": float(np.percentile(e, 75)),
        "tau10": float(100.0 * np.mean(e <= thr_a)),
        "tau5": float(100.0 * np.mean(e <= thr_b)),
        "n": int(e.size),
    }


# ----------------------------------------------------------------------
# Rendu
# ----------------------------------------------------------------------
def fmt(x, d=2):
    return f"{x:.{d}f}"


def print_console(rows, n_images):
    print(f"\nResultats sur {n_images} images\n")
    header = ["Methode", "e_med (m)", "e_75 (m)", "tau_10 (%)", "tau_5 (%)"]
    table = [header] + [[r["label"], r["c1"], r["c2"], r["c3"], r["c4"]] for r in rows]
    w = [max(len(str(line[i])) for line in table) for i in range(5)]
    for k, line in enumerate(table):
        print("  ".join(str(c).ljust(w[i]) for i, c in enumerate(line)))
        if k == 0:
            print("  ".join("-" * wi for wi in w))
    print()


def latex_table(rows, n_images, n_total=None):
    L = []
    L.append(r"\begin{table}[H]")
    L.append(r"	\centering")
    L.append(r"	\caption{Comparaison des méthodes de localisation.")
    L.append(rf"		Résultats sur le jeu de données de test")
    if n_total and n_total != n_images:
        L.append(rf"		({n_images} images contenant un logo détecté, sur {n_total} images de la zone campus).}}")
    else:
        L.append(rf"		({n_images} images, zone campus).}}")
    L.append(r"	\label{tab:resultats}")
    L.append(r"	\begin{tabularx}{\textwidth}{Xcccc}")
    L.append(r"		\toprule")
    L.append(r"		Méthode &")
    L.append(r"		$\tilde{e}$ (m) &")
    L.append(r"		$e_{75}$ (m) &")
    L.append(r"		$\tau_{10}$ (\%) &")
    L.append(r"		$\tau_5$ (\%) \\")
    L.append(r"		\midrule")
    for r in rows:
        if r.get("bold"):
            lab = rf"\textbf{{{r['label']}}}"
            cells = [rf"$\bm{{{r['v'][i]}}}$" for i in range(4)]
        else:
            lab = r["label"]
            cells = r["tex"]
        L.append(f"		{lab} &")
        L.append(f"		{' & '.join(cells)} \\\\")
    L.append(r"		\bottomrule")
    L.append(r"	\end{tabularx}")
    L.append(r"\end{table}")
    return "\n".join(L)


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Evaluation de la cascade sur un dossier d'images")
    ap.add_argument("--images-dir", required=True, help="Dossier des images de test")
    ap.add_argument("--gt-csv", default=None, help="CSV verite terrain (name,lat,lon[,heading_deg,gps_lat,gps_lon])")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--with-level3", action="store_true", help="Evalue aussi le Niveau 3 (COLMAP/PG-PnP)")

    ap.add_argument("--apr-model", default=config.DEFAULT_PATHS["apr_model"])
    ap.add_argument("--apr-norm", default=config.DEFAULT_PATHS["apr_norm_params"])
    ap.add_argument("--yolo-model", default=config.DEFAULT_PATHS["yolo_model"])
    ap.add_argument("--sig", default=config.DEFAULT_PATHS["sig_geojson"])
    ap.add_argument("--colmap-points3d", default=config.DEFAULT_PATHS["colmap_points3d"])
    ap.add_argument("--colmap-images", default=config.DEFAULT_PATHS["colmap_images"])
    ap.add_argument("--colmap-cameras", default=config.DEFAULT_PATHS["colmap_cameras"])
    ap.add_argument("--colmap-db", default=config.DEFAULT_PATHS["colmap_database"])
    ap.add_argument("--gps-csv", default=config.DEFAULT_PATHS["gps_csv"])
    ap.add_argument("--cache-dir", default=config.DEFAULT_PATHS["cache_dir"])
    ap.add_argument("--focal-mm", type=float, default=None)
    ap.add_argument("--sensor-width-mm", type=float, default=None)
    args = ap.parse_args()

    if not os.path.isdir(args.images_dir):
        sys.exit(f"Dossier introuvable : {args.images_dir}")
    os.makedirs(args.out_dir, exist_ok=True)

    images = sorted(f for f in os.listdir(args.images_dir) if f.lower().endswith(IMG_EXT))
    if not images:
        sys.exit("Aucune image trouvee dans le dossier.")

    gt_csv = load_gt_csv(args.gt_csv) if args.gt_csv else {}

    # --- Chargement unique des moteurs ---
    print("Chargement des modeles...")
    apr = APRLocalizer(args.apr_model, args.apr_norm)
    l2 = YoloSigRefiner(args.yolo_model, args.sig)
    l3 = None
    if args.with_level3:
        from level3_colmap_pnp import ColmapPnPRefiner
        l3 = ColmapPnPRefiner(args.colmap_points3d, args.colmap_images, args.colmap_cameras,
                              args.colmap_db, gps_csv_path=args.gps_csv,
                              cache_dir=args.cache_dir, verbose=False)

    per_image = []
    skipped = []
    for i, name in enumerate(images, 1):
        path = os.path.join(args.images_dir, name)
        g = gt_csv.get(name, {})
        lat_gt, lon_gt = g.get("lat"), g.get("lon")
        # Placeholder (0,0) = verite terrain non renseignee -> traite comme manquant
        if lat_gt is not None and lon_gt is not None and abs(lat_gt) < 1e-9 and abs(lon_gt) < 1e-9:
            lat_gt = lon_gt = None
        if lat_gt is None or lon_gt is None:
            ex = exif_gps(path)
            if ex:
                lat_gt, lon_gt = ex
        if lat_gt is None or lon_gt is None:
            skipped.append(name)
            continue

        try:
            r1 = apr.predict(path, verbose=False)
            heading = g.get("heading")
            r2 = l2.refine(path, r1, heading_deg=heading, focal_mm=args.focal_mm,
                           sensor_width_mm=args.sensor_width_mm, verbose=False)
        except Exception as ex_:
            print(f"[{i}/{len(images)}] {name} : ERREUR ({ex_})")
            skipped.append(name)
            continue

        e1 = haversine_m(lat_gt, lon_gt, r1["lat"], r1["lon"])

        logo = bool(r2.get("logo_found")) and r2.get("best_candidate") is not None
        if logo:
            lat2, lon2 = r2["best_candidate"]["lat"], r2["best_candidate"]["lon"]
        else:
            lat2, lon2 = r1["lat"], r1["lon"]  # repli sur APR
        e2 = haversine_m(lat_gt, lon_gt, lat2, lon2)

        e3 = None
        if l3 is not None:
            prior = ({"lat": lat2, "lon": lon2, "quat": r1["quat"], "yaw_deg": r1["yaw_deg"]}
                     if logo else r1)
            try:
                r3 = l3.refine(path, prior, prior_level=2 if logo else 1, verbose=False)
                if r3.get("success") and r3.get("lat") is not None:
                    e3 = haversine_m(lat_gt, lon_gt, r3["lat"], r3["lon"])
                else:
                    e3 = e2  # repli sur le meilleur niveau precedent
            except Exception:
                e3 = e2

        e_gps = None
        if g.get("gps_lat") is not None and g.get("gps_lon") is not None:
            e_gps = haversine_m(lat_gt, lon_gt, g["gps_lat"], g["gps_lon"])

        per_image.append({"name": name, "logo_detected": int(bool(r2.get("logo_found"))),
                          "logo_found": int(logo), "err_apr_m": e1,
                          "err_level2_m": e2, "err_level3_m": e3, "err_gps_m": e_gps})
        print(f"[{i}/{len(images)}] {name} : APR={e1:6.2f} m | N2={e2:6.2f} m"
              + (f" | N3={e3:6.2f} m" if e3 is not None else "")
              + (" (logo)" if logo else " (repli APR)"))

    if not per_image:
        sys.exit("Aucune image evaluable (verite terrain manquante ?).")

    # ------------------------------------------------------------------
    # Etat des donnees
    # ------------------------------------------------------------------
    n_total = len(images)
    n_eval = len(per_image)
    n_detect = sum(r["logo_detected"] for r in per_image)
    subset = [r for r in per_image if r["logo_found"]]   # logo detecte + candidat SIG
    n = len(subset)
    n_detect_no_cand = n_detect - n

    print("\n" + "=" * 60)
    print("ETAT DES DONNEES")
    print("=" * 60)
    print(f"Images dans le dossier                     : {n_total}")
    print(f"  - sans verite terrain / en erreur        : {len(skipped)}")
    print(f"  - evaluables (avec verite terrain)       : {n_eval}")
    print(f"      dont logo detecte (YOLO)             : {n_detect}")
    print(f"      dont logo + candidat SIG retenu      : {n}  <- base des statistiques")
    print(f"      dont sans logo (repli sur APR)       : {n_eval - n_detect}")
    if n_detect_no_cand:
        print(f"      dont logo detecte sans candidat SIG  : {n_detect_no_cand}")
    print("=" * 60)

    if n == 0:
        sys.exit("Aucune image avec logo detecte : pas de statistiques a calculer.")
    if n < 10:
        print(f"ATTENTION : seulement {n} images dans le sous-ensemble ; "
              f"tau_10 et tau_5 sont des multiples de {100.0 / n:.1f} %, "
              f"a interpreter avec prudence.")

    s_apr = stats([r["err_apr_m"] for r in subset])
    s_l2 = stats([r["err_level2_m"] for r in subset])
    s_l3 = stats([r["err_level3_m"] for r in subset]) if l3 is not None else None
    gps_vals = [r["err_gps_m"] for r in subset if r["err_gps_m"] is not None]
    s_gps = stats(gps_vals) if len(gps_vals) >= 3 else None

    def row_from(label, s, bold=False):
        v = [fmt(s["median"]), fmt(s["p75"]), fmt(s["tau10"], 1), fmt(s["tau5"], 1)]
        return {"label": label, "c1": v[0], "c2": v[1], "c3": v[2], "c4": v[3],
                "v": v, "tex": [f"${x}$" for x in v], "bold": bold}

    rows = []
    if s_gps:
        rows.append(row_from("GPS standard (mesure)", s_gps))
    else:  # valeurs de reference (litterature), comme dans la these
        rows.append({"label": "GPS standard", "c1": "15--40", "c2": "60", "c3": "40", "c4": "15",
                     "tex": ["15--40", "60", "40", "15"], "bold": False})
    rows.append(row_from("APR seul (MobileNet)", s_apr))
    rows.append(row_from("APR + YOLO + Scoring géométrique", s_l2, bold=(s_l3 is None)))
    if s_l3:
        rows.append(row_from("APR + YOLO + Scoring + PG-PnP", s_l3, bold=True))

    print(f"\nSTATISTIQUES SUR LES {n} IMAGES AVEC LOGO DETECTE "
          f"(APR et APR+YOLO evalues sur les memes images)")
    print_console(rows, n)

    # Pour information : APR sur l'ensemble des images evaluables
    s_apr_all = stats([r["err_apr_m"] for r in per_image])
    print(f"Pour memoire, APR seul sur les {n_eval} images evaluables : "
          f"e_med={s_apr_all['median']:.2f} m, e_75={s_apr_all['p75']:.2f} m, "
          f"tau_10={s_apr_all['tau10']:.1f} %, tau_5={s_apr_all['tau5']:.1f} %\n")

    tex = latex_table(rows, n, n_total=n_total)
    with open(os.path.join(args.out_dir, "tab_resultats.tex"), "w", encoding="utf-8") as f:
        f.write(tex + "\n")
    with open(os.path.join(args.out_dir, "per_image_errors.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_image[0].keys()))
        w.writeheader()
        w.writerows(per_image)
    with open(os.path.join(args.out_dir, "data_summary.txt"), "w", encoding="utf-8") as f:
        f.write(f"images_dossier={n_total}\nevaluables={n_eval}\nlogo_detecte={n_detect}\n"
                f"logo_plus_candidat={n}\nsans_logo={n_eval - n_detect}\n")

    print(f"Tableau LaTeX : {os.path.join(args.out_dir, 'tab_resultats.tex')}")
    print(f"Erreurs par image : {os.path.join(args.out_dir, 'per_image_errors.csv')}\n")
    print(tex)



if __name__ == "__main__":
    main()
