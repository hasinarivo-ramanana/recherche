# -*- coding: utf-8 -*-
"""
level2_yolo_sig.py
===================
NIVEAU 2 : detection des logos/enseignes par YOLOv10 (.tflite), requete
dans le SIG (sig_data.geojson) et selection du candidat par scoring
multi-criteres (Contribution 2 de la these).

Si aucun logo n'est detecte, ce niveau est court-circuite et le pipeline
passe directement du Niveau 1 au Niveau 3 (avec la position APR comme prior).
"""

import json
import math
import time

import numpy as np
from PIL import Image

import config
import geo_utils
import display_utils as disp
from level1_apr import _make_interpreter, _TFLITE_BACKEND  # reuse TFLite backend


# ==========================================================================
# Chargement / requete du SIG
# ==========================================================================
def load_sig(geojson_path):
    with open(geojson_path, "r", encoding="utf-8") as f:
        gj = json.load(f)
    points = []
    for feat in gj["features"]:
        lon, lat = feat["geometry"]["coordinates"][:2]
        props = feat["properties"]
        points.append({
            "id": props.get("id"),
            "classe": props.get("classe"),
            "lat": lat, "lon": lon,
            "hauteur_reelle_m": props.get("hauteur_reelle_m", 1.0),
            "azimut_facade_deg": props.get("azimut_facade_deg", 0.0),
            "nom_batiment": props.get("nom_batiment", props.get("id")),
        })
    return points


def query_sig_candidates(sig_points, classe, center_lat, center_lon, radius_m):
    candidates = []
    for p in sig_points:
        if p["classe"] != classe:
            continue
        d = geo_utils.haversine(center_lat, center_lon, p["lat"], p["lon"])
        if d <= radius_m:
            cand = dict(p)
            cand["dist_apr_m"] = d
            candidates.append(cand)
    candidates.sort(key=lambda c: c["dist_apr_m"])
    return candidates


# ==========================================================================
# YOLOv10 TFLite : pre/post-traitement
# ==========================================================================
class YoloDetector:
    def __init__(self, model_path, class_names, conf_threshold=None, iou_threshold=None):
        self.class_names = class_names
        self.conf_threshold = conf_threshold or config.YOLO_CONF_THRESHOLD
        self.iou_threshold = iou_threshold or config.YOLO_IOU_THRESHOLD
        self.interpreter = _make_interpreter(model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self._resolve_input_layout()
        self._print_signature()

    # ------------------------------------------------------------------
    def _resolve_input_layout(self):
        """
        Determine automatiquement, a partir du tenseur d'entree reel du
        modele :
          - le format spatial : NHWC (1,H,W,3) ou NCHW (1,3,H,W)
          - la taille d'entree (H, W)
          - le type de donnees attendu : float32 (0..1) ou quantifie
            (uint8/int8), auquel cas les parametres de quantification
            (scale, zero_point) sont utilises pour convertir les pixels.
        Evite de supposer un format fixe : les exports TFLite de YOLO
        varient (NHWC est le plus courant, mais NCHW existe aussi selon
        l'outil de conversion utilise).
        """
        shape = list(self.input_details[0]["shape"])
        if len(shape) != 4:
            raise ValueError(f"Forme d'entree YOLO inattendue : {shape} "
                              "(4 dimensions attendues : batch, H, W, C ou batch, C, H, W).")

        # Le canal (3 = RGB) est soit en position 1 (NCHW) soit en position 3 (NHWC)
        if shape[1] in (1, 3) and shape[1] != shape[3]:
            self.layout = "NCHW"
            self.channels, self.in_h, self.in_w = shape[1], shape[2], shape[3]
        else:
            self.layout = "NHWC"
            self.in_h, self.in_w, self.channels = shape[1], shape[2], shape[3]

        self.input_dtype = self.input_details[0]["dtype"]
        self.quant_scale, self.quant_zero_point = self.input_details[0]["quantization"]

    def _print_signature(self):
        disp.kv("Modele YOLO - forme d'entree brute",
                 str(list(self.input_details[0]["shape"])))
        disp.kv("Modele YOLO - format detecte",
                 f"{self.layout}, taille {self.in_w}x{self.in_h}, {self.channels} canaux")
        disp.kv("Modele YOLO - type d'entree", str(self.input_dtype)
                 + (f" (quantifie: scale={self.quant_scale}, zero_point={self.quant_zero_point})"
                    if self.quant_scale else " (flottant, normalise 0..1)"))
        for i, od in enumerate(self.output_details):
            disp.kv(f"Modele YOLO - sortie #{i}", f"forme={list(od['shape'])}, dtype={od['dtype']}")

    # ------------------------------------------------------------------
    def _preprocess(self, image_path):
        img = Image.open(image_path).convert("RGB")
        W, H = img.size
        img_resized = img.resize((self.in_w, self.in_h), Image.BILINEAR)
        arr = np.asarray(img_resized).astype(np.float32)  # (H, W, 3), 0..255

        if np.issubdtype(self.input_dtype, np.floating):
            arr = arr / 255.0
        elif self.quant_scale:
            # Modele quantifie : q = pixel_normalise/scale + zero_point
            arr = (arr / 255.0) / self.quant_scale + self.quant_zero_point
            info = np.iinfo(self.input_dtype)
            arr = np.clip(np.round(arr), info.min, info.max)
        # sinon (uint8 non quantifie explicitement) : on garde 0..255 tel quel

        if self.layout == "NCHW":
            arr = np.transpose(arr, (2, 0, 1))  # (H,W,C) -> (C,H,W)

        arr = np.expand_dims(arr, axis=0).astype(self.input_dtype)
        return arr, (W, H)

    def _run(self, tensor):
        self.interpreter.set_tensor(self.input_details[0]["index"], tensor)
        self.interpreter.invoke()
        return [self.interpreter.get_tensor(od["index"]) for od in self.output_details]

    @staticmethod
    def _nms(boxes, scores, iou_thr):
        if len(boxes) == 0:
            return []
        boxes = np.asarray(boxes, dtype=np.float64)
        scores = np.asarray(scores, dtype=np.float64)
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1).clip(0) * (y2 - y1).clip(0)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            inter = (xx2 - xx1).clip(0) * (yy2 - yy1).clip(0)
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
            order = order[1:][iou <= iou_thr]
        return keep

    def _decode(self, outputs, img_w, img_h):
        """
        Gere deux formats d'export courants :
          (a) NMS integree a l'export  -> sortie (1, N, 6) = [x1,y1,x2,y2,score,cls]
          (b) sortie brute YOLO        -> (1, 4+num_classes, N) ou (1, N, 4+num_classes)
        Les coordonnees sont supposees exprimees dans le repere de l'image
        redimensionnee (input_size x input_size) ; elles sont remises a
        l'echelle de l'image originale.
        """
        out = outputs[0]
        out = np.squeeze(out)

        detections = []
        scale_x = img_w / self.in_w
        scale_y = img_h / self.in_h

        if out.ndim == 2 and out.shape[-1] == 6:
            # format (N, 6) avec NMS deja appliquee par l'export
            for row in out:
                x1, y1, x2, y2, score, cls = row
                if score < self.conf_threshold:
                    continue
                cls = int(cls)
                detections.append({
                    "bbox": (x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y),
                    "score": float(score),
                    "class_id": cls,
                    "class_name": self.class_names[cls] if cls < len(self.class_names) else str(cls),
                })
            return detections

        # format brut : (4+num_classes, N) ou (N, 4+num_classes) -> transposer si besoin
        if out.shape[0] < out.shape[1]:
            preds = out.T  # -> (N, 4+num_classes)
        else:
            preds = out

        boxes_xywh = preds[:, :4]
        class_scores = preds[:, 4:]
        class_ids = np.argmax(class_scores, axis=1)
        confs = class_scores[np.arange(len(class_ids)), class_ids]

        mask = confs >= self.conf_threshold
        boxes_xywh, class_ids, confs = boxes_xywh[mask], class_ids[mask], confs[mask]

        boxes_xyxy = []
        for (cx, cy, w, h) in boxes_xywh:
            boxes_xyxy.append([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])

        keep = self._nms(boxes_xyxy, confs, self.iou_threshold)
        for i in keep:
            x1, y1, x2, y2 = boxes_xyxy[i]
            cls = int(class_ids[i])
            detections.append({
                "bbox": (x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y),
                "score": float(confs[i]),
                "class_id": cls,
                "class_name": self.class_names[cls] if cls < len(self.class_names) else str(cls),
            })
        return detections

    def detect(self, image_path):
        tensor, (img_w, img_h) = self._preprocess(image_path)
        outputs = self._run(tensor)
        detections = self._decode(outputs, img_w, img_h)
        detections.sort(key=lambda d: -d["score"])
        return detections, (img_w, img_h)


# ==========================================================================
# Scoring multi-criteres (Section "Scoring Multi-Criteres" de la these)
# ==========================================================================
def score_taille(h_px, focal_px, H_reel_m, d_sig_m, sigma_d=config.SIGMA_D_M):
    d_estimee = (focal_px * H_reel_m) / max(h_px, 1.0)
    erreur = abs(d_estimee - d_sig_m)
    return math.exp(-(erreur ** 2) / (2 * sigma_d ** 2)), d_estimee


def score_angle(alpha_image_deg, alpha_sig_deg, sigma_alpha=config.SIGMA_ALPHA_DEG):
    diff = (alpha_image_deg - alpha_sig_deg + 180) % 360 - 180  # ecart angulaire signe
    return math.exp(-(diff ** 2) / (2 * sigma_alpha ** 2)), diff


def score_rapport(r_image, r_sig, sigma_r=config.SIGMA_R_RATIO):
    return math.exp(-((r_image - r_sig) ** 2) / (2 * sigma_r ** 2))


def score_cote(u_logo, img_w, apr_lat, apr_lon, cap_deg, cand_lat, cand_lon):
    cote_image = "gauche" if u_logo < img_w / 2 else "droite"
    cap_rad = math.radians(cap_deg)
    vx_cap, vy_cap = math.cos(cap_rad), math.sin(cap_rad)
    dx, dy = (cand_lon - apr_lon), (cand_lat - apr_lat)
    norme = math.hypot(dx, dy)
    if norme < 1e-12:
        return 0.5, "indetermine"
    vx_cand, vy_cand = dx / norme, dy / norme
    produit_croise = vx_cap * vy_cand - vy_cap * vx_cand
    cote_sig = "gauche" if produit_croise > 0 else "droite"
    return (1.0 if cote_image == cote_sig else 0.1), cote_sig


def estimate_focal_px(img_w, img_h, focal_mm=None, sensor_width_mm=None):
    """Estimation de la focale en pixels. Utilise EXIF si fourni, sinon heuristique."""
    if focal_mm and sensor_width_mm:
        return (focal_mm / sensor_width_mm) * img_w
    return config.DEFAULT_FOCAL_HEURISTIC_FACTOR * max(img_w, img_h)


# ==========================================================================
# Orchestrateur Niveau 2
# ==========================================================================
class YoloSigRefiner:
    def __init__(self, yolo_model_path, sig_geojson_path,
                 class_names=None, conf_threshold=None):
        self.class_names = class_names or config.CLASS_NAMES
        self.detector = YoloDetector(yolo_model_path, self.class_names, conf_threshold)
        self.sig_points = load_sig(sig_geojson_path)

    def refine(self, image_path, level1_result, heading_deg=None,
               focal_mm=None, sensor_width_mm=None, verbose=True):
        t0 = time.perf_counter()
        detections, (img_w, img_h) = self.detector.detect(image_path)
        detect_ms = (time.perf_counter() - t0) * 1000.0

        result = {
            "detections": detections,
            "image_size": (img_w, img_h),
            "detect_ms": detect_ms,
            "logo_found": len(detections) > 0,
            "best_candidate": None,
            "candidates_scored": [],
            "used_heading": heading_deg is not None,
        }

        if not detections:
            if verbose:
                self._display_no_detection(image_path, result)
            return result

        focal_px = estimate_focal_px(img_w, img_h, focal_mm, sensor_width_mm)
        cap_deg = heading_deg if heading_deg is not None else level1_result["yaw_deg"]

        best_det = detections[0]  # meilleure detection (score de confiance max)
        u1, v1, u2, v2 = best_det["bbox"]
        u_center = (u1 + u2) / 2.0
        h_px = max(v2 - v1, 1.0)
        u0 = img_w / 2.0

        candidates = query_sig_candidates(
            self.sig_points, best_det["class_name"],
            level1_result["lat"], level1_result["lon"],
            config.LEVEL2_SIG_QUERY_RADIUS_M
        )

        # Detection secondaire (classe differente) pour la contrainte "rapport"
        second_det = None
        for d in detections[1:]:
            if d["class_name"] != best_det["class_name"]:
                second_det = d
                break

        w = config.SCORE_WEIGHTS
        scored = []
        for cand in candidates:
            s1, d_est = score_taille(h_px, focal_px, cand["hauteur_reelle_m"],
                                      cand["dist_apr_m"])

            alpha_image = math.degrees(math.atan2(u_center - u0, focal_px)) + cap_deg
            alpha_sig = math.degrees(math.atan2(cand["lon"] - level1_result["lon"],
                                                 cand["lat"] - level1_result["lat"]))
            s2, angle_diff = score_angle(alpha_image, alpha_sig)

            s4, cote_sig = score_cote(u_center, img_w, level1_result["lat"],
                                       level1_result["lon"], cap_deg,
                                       cand["lat"], cand["lon"])

            s3 = 1.0  # valeur neutre par defaut (pas de 2e logo co-detecte)
            r_image = r_sig = None
            if second_det is not None:
                u1b, v1b, u2b, v2b = second_det["bbox"]
                c1 = np.array([u_center, (v1 + v2) / 2.0])
                c2 = np.array([(u1b + u2b) / 2.0, (v1b + v2b) / 2.0])
                d_image = np.linalg.norm(c1 - c2)
                sec_candidates = query_sig_candidates(
                    self.sig_points, second_det["class_name"],
                    level1_result["lat"], level1_result["lon"],
                    config.LEVEL2_SIG_QUERY_RADIUS_M)
                if sec_candidates and d_image > 1e-6:
                    nearest_second = sec_candidates[0]
                    d_geo_12 = geo_utils.haversine(cand["lat"], cand["lon"],
                                                    nearest_second["lat"], nearest_second["lon"])
                    d_geo_1apr = max(cand["dist_apr_m"], 1e-6)
                    r_image = d_image / max(h_px, 1e-6)  # ratio image normalise par la taille du logo
                    r_sig = d_geo_12 / max(d_geo_1apr, 1e-6)
                    s3 = score_rapport(r_image, r_sig)

            total = (w["taille"] * s1 + w["angle"] * s2
                     + w["rapport"] * s3 + w["cote"] * s4)

            scored.append({
                **cand,
                "s_taille": s1, "d_estimee_m": d_est,
                "s_angle": s2, "alpha_image_deg": alpha_image,
                "alpha_sig_deg": alpha_sig, "angle_diff_deg": angle_diff,
                "s_rapport": s3, "r_image": r_image, "r_sig": r_sig,
                "s_cote": s4, "cote_sig": cote_sig,
                "score_total": total,
            })

        scored.sort(key=lambda c: -c["score_total"])
        result["candidates_scored"] = scored
        result["best_candidate"] = scored[0] if scored else None
        result["best_detection"] = best_det
        result["second_detection"] = second_det
        result["focal_px"] = focal_px
        result["cap_deg"] = cap_deg
        result["expected_error_m"] = config.LEVEL2_EXPECTED_ERROR_M

        if verbose:
            self._display(image_path, result)
        return result

    # ------------------------------------------------------------------
    @staticmethod
    def _display_no_detection(image_path, r):
        disp.title("NIVEAU 2 - DETECTION YOLO + REQUETE SIG")
        disp.kv("Image requete", image_path)
        disp.kv("Detections YOLO", "aucune")
        disp.warn("Aucun logo/enseigne detecte : le Niveau 2 est ignore.")
        disp.warn("Le pipeline passe directement au Niveau 3 avec le prior "
                   "de position du Niveau 1 (branche 'Technique nuage de points').")
        print()

    @staticmethod
    def _display(image_path, r):
        disp.title("NIVEAU 2 - DETECTION YOLO + REQUETE SIG")
        disp.kv("Image requete", image_path)
        disp.kv("Taille image", f"{r['image_size'][0]} x {r['image_size'][1]}", "px")
        disp.kv("Temps de detection YOLO", f"{r['detect_ms']:.1f}", "ms")
        disp.kv("Cap utilise (heading)", f"{r['cap_deg']:.1f}", "deg"
                 + (" (fourni)" if r["used_heading"] else " (issu du Niveau 1)"))
        disp.kv("Focale estimee", f"{r['focal_px']:.1f}", "px")

        disp.section(f"Detections YOLO ({len(r['detections'])})")
        rows = [[i, d["class_name"], f"{d['score']:.3f}",
                 f"({d['bbox'][0]:.0f},{d['bbox'][1]:.0f})-({d['bbox'][2]:.0f},{d['bbox'][3]:.0f})"]
                for i, d in enumerate(r["detections"])]
        disp.table(["#", "Classe", "Confiance", "BBox (px)"], rows)

        best_det = r["best_detection"]
        disp.section("Detection retenue (confiance maximale)")
        disp.kv("Classe", best_det["class_name"])
        disp.kv("Confiance", f"{best_det['score']:.3f}")
        if r["second_detection"] is not None:
            disp.kv("2e logo co-detecte (classe differente)",
                     r["second_detection"]["class_name"])
        else:
            disp.bullet("Aucun 2e logo de classe differente co-detecte "
                        "-> contrainte 'rapport' neutralisee (score = 1.0)")

        disp.section(f"Candidats SIG (classe='{best_det['class_name']}', "
                     f"rayon={config.LEVEL2_SIG_QUERY_RADIUS_M:.0f} m)")
        if not r["candidates_scored"]:
            disp.warn("Aucun candidat SIG trouve dans le rayon de requete.")
        else:
            rows = []
            for c in r["candidates_scored"]:
                rows.append([
                    c["id"], f"{c['dist_apr_m']:.1f}",
                    f"{c['s_taille']:.3f}", f"{c['s_angle']:.3f}",
                    f"{c['s_rapport']:.3f}", f"{c['s_cote']:.3f}",
                    f"{c['score_total']:.3f}",
                ])
            disp.table(["ID SIG", "d(APR) m", "s_taille", "s_angle",
                        "s_rapport", "s_cote", "SCORE"], rows)

            best = r["best_candidate"]
            disp.section("Candidat selectionne (argmax du score composite)")
            disp.kv("ID SIG", best["id"])
            disp.kv("Nom du batiment", best["nom_batiment"])
            disp.kv("Position affinee (lat, lon)",
                     f"({best['lat']:.7f}, {best['lon']:.7f})")
            disp.kv("Distance a la position APR (Niveau 1)",
                     f"{best['dist_apr_m']:.2f}", "m")
            disp.kv("Distance estimee par taille apparente",
                     f"{best['d_estimee_m']:.2f}", "m")
            disp.kv("Ecart angulaire (image vs SIG)",
                     f"{best['angle_diff_deg']:.2f}", "deg")
            disp.kv("Cote de la rue (image / SIG)",
                     f"{'gauche' if best['s_cote'] == 1.0 else 'droite'} / {best['cote_sig']}")
            disp.kv("Score total", f"{best['score_total']:.4f}")
            disp.kv("Erreur mediane attendue (jeu de test)",
                     f"{config.LEVEL2_EXPECTED_ERROR_M:.2f}", "m")
        print()
