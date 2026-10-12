# -*- coding: utf-8 -*-
"""
main.py
=======
Point d'entree de l'application de localisation en cascade (Niveau 1 : APR
-> Niveau 2 : YOLO+SIG -> Niveau 3 : COLMAP/PG-PnP), cf. these, chapitres
"Contribution 1/2/3".

Usage :
    python main.py --image chemin/vers/photo.jpg [options]

Exemple minimal (chemins par defaut definis dans config.py) :
    python main.py --image test.jpg

Exemple complet :
    python main.py --image test.jpg \\
        --apr-model models/apr_model.tflite \\
        --apr-norm models/apr_norm_params.json \\
        --yolo-model models/yolo_model.tflite \\
        --sig data/sig_data.geojson \\
        --colmap-points3d colmap/points3D.txt \\
        --colmap-images colmap/images.txt \\
        --colmap-cameras colmap/cameras.txt \\
        --colmap-db colmap/database.db \\
        --gps-csv colmap/dataset_gps.csv \\
        --heading 87.5
"""

import argparse
import os
import sys
import time

import config
import display_utils as disp
from level1_apr import APRLocalizer
from level2_yolo_sig import YoloSigRefiner
from level3_colmap_pnp import ColmapPnPRefiner


def parse_args():
    p = argparse.ArgumentParser(
        description="Localisation en cascade sans GNSS (APR -> YOLO+SIG -> COLMAP/PG-PnP)")
    p.add_argument("--image", required=True, help="Image de requete (JPEG/PNG)")

    # Niveau 1
    p.add_argument("--apr-model", default=config.DEFAULT_PATHS["apr_model"])
    p.add_argument("--apr-norm", default=config.DEFAULT_PATHS["apr_norm_params"])

    # Niveau 2
    p.add_argument("--yolo-model", default=config.DEFAULT_PATHS["yolo_model"])
    p.add_argument("--sig", default=config.DEFAULT_PATHS["sig_geojson"])
    p.add_argument("--heading", type=float, default=None,
                    help="Cap de la camera en degres (sinon: yaw du Niveau 1)")
    p.add_argument("--focal-mm", type=float, default=None, help="Focale EXIF (mm)")
    p.add_argument("--sensor-width-mm", type=float, default=None,
                    help="Largeur du capteur (mm), pour convertir la focale en pixels")

    # Niveau 3
    p.add_argument("--colmap-points3d", default=config.DEFAULT_PATHS["colmap_points3d"])
    p.add_argument("--colmap-images", default=config.DEFAULT_PATHS["colmap_images"])
    p.add_argument("--colmap-cameras", default=config.DEFAULT_PATHS["colmap_cameras"])
    p.add_argument("--colmap-db", default=config.DEFAULT_PATHS["colmap_database"])
    p.add_argument("--gps-csv", default=config.DEFAULT_PATHS["gps_csv"],
                    help="CSV (name,lat,lon,...) des images d'entrainement COLMAP, "
                         "necessaire pour georeferencer la sortie du Niveau 3")
    p.add_argument("--cache-dir", default=config.DEFAULT_PATHS["cache_dir"])

    p.add_argument("--skip-level3", action="store_true",
                    help="S'arreter apres le Niveau 2 (debogage rapide)")
    return p.parse_args()


def main():
    args = parse_args()
    t_global = time.perf_counter()

    disp.title("PIPELINE DE LOCALISATION VISUELLE SANS GNSS - 3 NIVEAUX", "#")
    disp.kv("Image d'entree", args.image)
    if not os.path.exists(args.image):
        disp.fail(f"Image introuvable : {args.image}")
        sys.exit(1)

    # ======================================================================
    # NIVEAU 1 - APR
    # ======================================================================
    apr = APRLocalizer(args.apr_model, args.apr_norm)
    level1_result = apr.predict(args.image)

    # ======================================================================
    # NIVEAU 2 - YOLO + SIG
    # ======================================================================
    refiner2 = YoloSigRefiner(args.yolo_model, args.sig)
    level2_result = refiner2.refine(
        args.image, level1_result,
        heading_deg=args.heading, focal_mm=args.focal_mm,
        sensor_width_mm=args.sensor_width_mm)

    # Position utilisee comme prior pour le Niveau 3 : celle du Niveau 2 si
    # un logo a ete detecte et qu'un candidat SIG a ete trouve, sinon celle
    # du Niveau 1 (branche "technique nuage de points" du schema global).
    if level2_result["logo_found"] and level2_result["best_candidate"] is not None:
        prior_for_l3 = {
            "lat": level2_result["best_candidate"]["lat"],
            "lon": level2_result["best_candidate"]["lon"],
            "quat": level1_result["quat"],   # orientation toujours issue du Niveau 1
            "yaw_deg": level1_result["yaw_deg"],
        }
        prior_level = 2
    else:
        prior_for_l3 = level1_result
        prior_level = 1

    if args.skip_level3:
        disp.title("PIPELINE ARRETE APRES LE NIVEAU 2 (--skip-level3)", "#")
        return

    # ======================================================================
    # NIVEAU 3 - COLMAP + PG-PnP
    # ======================================================================
    refiner3 = ColmapPnPRefiner(
        args.colmap_points3d, args.colmap_images, args.colmap_cameras,
        args.colmap_db, gps_csv_path=args.gps_csv, cache_dir=args.cache_dir)
    level3_result = refiner3.refine(args.image, prior_for_l3, prior_level=prior_level)

    # ======================================================================
    # SYNTHESE GLOBALE
    # ======================================================================
    disp.title("SYNTHESE DE LA CASCADE DE LOCALISATION", "#")
    rows = [
        ["1 - APR", f"{level1_result['lat']:.7f}", f"{level1_result['lon']:.7f}",
         f"~{level1_result['expected_error_m']:.0f} m", f"{level1_result['elapsed_ms']:.0f} ms"],
        ["2 - YOLO+SIG",
         (f"{level2_result['best_candidate']['lat']:.7f}"
          if level2_result.get("best_candidate") else "-"),
         (f"{level2_result['best_candidate']['lon']:.7f}"
          if level2_result.get("best_candidate") else "-"),
         (f"~{config.LEVEL2_EXPECTED_ERROR_M:.0f} m"
          if level2_result.get("best_candidate") else "n/a (pas de logo)"),
         f"{level2_result['detect_ms']:.0f} ms"],
    ]
    if level3_result.get("success"):
        rows.append(["3 - PG-PnP", f"{level3_result['lat']:.7f}", f"{level3_result['lon']:.7f}",
                     f"< {level3_result['target_error_m']:.0f} m (objectif)",
                     f"{level3_result['elapsed_total_s']*1000:.0f} ms"])
    else:
        rows.append(["3 - PG-PnP", "repli", "repli", "n/a", "-"])
    disp.table(["Niveau", "Latitude finale", "Longitude finale", "Erreur", "Temps"], rows)

    disp.kv("\nTemps total pipeline", f"{time.perf_counter() - t_global:.2f}", "s")


if __name__ == "__main__":
    main()
