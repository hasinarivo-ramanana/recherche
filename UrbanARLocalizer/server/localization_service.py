# -*- coding: utf-8 -*-
"""
localization_service.py
========================
Enveloppe la cascade de localisation existante (level1_apr.py, level2_yolo_sig.py,
level3_colmap_pnp.py -- inchanges) dans un service reutilisable par le serveur FastAPI.

Difference avec main.py : ici les 3 moteurs (APR, YOLO+SIG, COLMAP/PG-PnP) sont charges
UNE SEULE FOIS au demarrage du serveur (cf. server.py, lifespan), pas a chaque requete --
sinon chaque appel /localize rechargerait le modele COLMAP (potentiellement plusieurs
centaines de Mo / plusieurs secondes), ce qui serait beaucoup trop lent pour un usage
"un clic dans l'app -> une reponse".

Cote resultat, on renvoie une structure plate {level, lat, lon, yaw_deg, ...} qui
correspond exactement a ce que l'app Android attend en retour du bouton
"Localiser ma position" (cf. README / RemoteLocalizationClient.kt cote Kotlin).
"""

import logging
import os
import tempfile
import time
from dataclasses import dataclass, field
from typing import Optional

import config
from level1_apr import APRLocalizer
from level2_yolo_sig import YoloSigRefiner
from level3_colmap_pnp import ColmapPnPRefiner

log = logging.getLogger("localization_service")


@dataclass
class LocalizationResponse:
    level: int                       # 1, 2 ou 3 : niveau qui a produit la pose finale
    lat: float
    lon: float
    yaw_deg: float
    pitch_deg: float = 0.0
    roll_deg: float = 0.0
    sigma_pos_m: Optional[float] = None   # incertitude approx. selon le niveau (voir _sigma_for_level)
    level1: dict = field(default_factory=dict)   # {lat, lon} -- toujours present, pour affichage/log
    level2: Optional[dict] = None                # {lat, lon} -- present si logo detecte + candidat SIG trouve
    level3: Optional[dict] = None                # {lat, lon} -- present si PG-PnP a reussi
    elapsed_ms: float = 0.0
    detail: str = ""                 # message humain (ex: raison du repli niveau 3)


class LocalizationService:
    """Singleton applicatif : charge les 3 niveaux une fois, expose localize()."""

    def __init__(
        self,
        apr_model: str,
        apr_norm: str,
        yolo_model: str,
        sig_geojson: str,
        colmap_points3d: str,
        colmap_images: str,
        colmap_cameras: str,
        colmap_db: str,
        gps_csv: str,
        cache_dir: str = "cache",
    ):
        log.info("Chargement Niveau 1 (APR)...")
        self.apr = APRLocalizer(apr_model, apr_norm)

        log.info("Chargement Niveau 2 (YOLO+SIG)...")
        self.level2 = YoloSigRefiner(yolo_model, sig_geojson)

        log.info("Chargement Niveau 3 (COLMAP/PG-PnP) -- peut prendre du temps...")
        self.level3 = ColmapPnPRefiner(
            colmap_points3d, colmap_images, colmap_cameras, colmap_db,
            gps_csv_path=gps_csv, cache_dir=cache_dir, verbose=False,
        )
        log.info("Cascade prete.")

    @staticmethod
    def _sigma_for_level(level: int) -> float:
        return {1: config.LEVEL1_EXPECTED_ERROR_M, 2: config.LEVEL2_EXPECTED_ERROR_M}.get(
            level, config.LEVEL3_TARGET_ERROR_M if hasattr(config, "LEVEL3_TARGET_ERROR_M") else 2.0
        )

    def localize(
        self,
        image_path: str,
        heading_deg: Optional[float] = None,
        focal_mm: Optional[float] = None,
        sensor_width_mm: Optional[float] = None,
    ) -> LocalizationResponse:
        t0 = time.perf_counter()

        # ---- Niveau 1 : toujours execute ----
        l1 = self.apr.predict(image_path, verbose=False)

        # ---- Niveau 2 : logos + SIG ----
        l2 = self.level2.refine(
            image_path, l1, heading_deg=heading_deg,
            focal_mm=focal_mm, sensor_width_mm=sensor_width_mm, verbose=False,
        )
        if l2["logo_found"] and l2.get("best_candidate") is not None:
            prior_for_l3 = {
                "lat": l2["best_candidate"]["lat"],
                "lon": l2["best_candidate"]["lon"],
                "quat": l1["quat"],
                "yaw_deg": l1["yaw_deg"],
            }
            prior_level = 2
        else:
            prior_for_l3 = l1
            prior_level = 1

        # ---- Niveau 3 : COLMAP / PG-PnP ----
        l3 = self.level3.refine(image_path, prior_for_l3, prior_level=prior_level, verbose=False)

        # ---- Selection de la pose finale : 3 > 2 > 1, avec repli garanti ----
        level2_dict = (
            {"lat": l2["best_candidate"]["lat"], "lon": l2["best_candidate"]["lon"]}
            if l2["logo_found"] and l2.get("best_candidate") is not None
            else None
        )
        level3_dict = {"lat": l3["lat"], "lon": l3["lon"]} if l3.get("success") else None

        if level3_dict is not None:
            final_level, final_lat, final_lon = 3, l3["lat"], l3["lon"]
            detail = "Niveau 3 (PG-PnP) reussi."
        elif level2_dict is not None:
            final_level, final_lat, final_lon = 2, level2_dict["lat"], level2_dict["lon"]
            detail = l3.get("fallback_reason", "Niveau 3 en repli -> Niveau 2 retenu.")
        else:
            final_level, final_lat, final_lon = 1, l1["lat"], l1["lon"]
            detail = l3.get("fallback_reason", "Niveau 2/3 en repli -> Niveau 1 retenu.")

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        return LocalizationResponse(
            level=final_level,
            lat=final_lat,
            lon=final_lon,
            yaw_deg=l1["yaw_deg"],
            pitch_deg=l1.get("pitch_deg", 0.0),
            roll_deg=l1.get("roll_deg", 0.0),
            sigma_pos_m=self._sigma_for_level(final_level),
            level1={"lat": l1["lat"], "lon": l1["lon"]},
            level2=level2_dict,
            level3=level3_dict,
            elapsed_ms=elapsed_ms,
            detail=detail,
        )


def save_upload_to_tempfile(raw_bytes: bytes, suffix: str = ".jpg") -> str:
    """Ecrit les octets recus (photo JPEG envoyee par l'app) dans un fichier temporaire,
    car level1/2/3 attendent tous un chemin fichier (image_path), pas des octets en memoire."""
    fd, path = tempfile.mkstemp(suffix=suffix, prefix="arloc_")
    with os.fdopen(fd, "wb") as f:
        f.write(raw_bytes)
    return path
