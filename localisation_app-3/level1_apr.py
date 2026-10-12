# -*- coding: utf-8 -*-
"""
level1_apr.py
=============
NIVEAU 1 : initialisation grossiere de la position par regression de pose
absolue (MobileNet-APR-Campus, cf. Contribution 1 de la these).

Entree  : image de requete (chemin fichier)
Sortie  : position (lat, lon) + orientation (quaternion + Euler) + zone
          candidate de rayon ~100 m transmise au Niveau 2.
"""

import json
import time

import numpy as np
from PIL import Image

import config
import geo_utils
import display_utils as disp

try:
    import tensorflow as tf
    _TFLITE_BACKEND = "tensorflow"
except ImportError:
    try:
        # ai-edge-litert est le successeur officiel de tflite-runtime
        # (tflite-runtime n'est plus publie sur PyPI depuis 2024).
        from ai_edge_litert.interpreter import Interpreter as _LiteRTInterpreter
        _TFLITE_BACKEND = "ai_edge_litert"
    except ImportError:
        try:
            import tflite_runtime.interpreter as tflite
            _TFLITE_BACKEND = "tflite_runtime"
        except ImportError:
            _TFLITE_BACKEND = None


def _make_interpreter(model_path):
    if _TFLITE_BACKEND == "tensorflow":
        return tf.lite.Interpreter(model_path=model_path)
    elif _TFLITE_BACKEND == "ai_edge_litert":
        return _LiteRTInterpreter(model_path=model_path)
    elif _TFLITE_BACKEND == "tflite_runtime":
        return tflite.Interpreter(model_path=model_path)
    else:
        raise RuntimeError(
            "Aucun backend TFLite disponible. Installez l'un des packages "
            "suivants : 'tensorflow', 'ai-edge-litert' (recommande, leger), "
            "ou 'tflite-runtime' (deprecie)."
        )


class APRLocalizer:
    """Charge le modele MobileNet-APR-Campus (.tflite) et le normalizer."""

    def __init__(self, model_path, norm_params_path):
        self.model_path = model_path
        with open(norm_params_path, "r", encoding="utf-8") as f:
            self.norm = json.load(f)

        self.interpreter = _make_interpreter(model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    # ------------------------------------------------------------------
    def _preprocess(self, image_path):
        img = Image.open(image_path).convert("RGB")
        orig_size = img.size  # (W, H)
        img_resized = img.resize((config.APR_INPUT_SIZE, config.APR_INPUT_SIZE),
                                  Image.BILINEAR)
        arr = np.asarray(img_resized).astype(np.float32) / 255.0
        mean = np.array(config.IMAGENET_MEAN, dtype=np.float32)
        std = np.array(config.IMAGENET_STD, dtype=np.float32)
        arr = (arr - mean) / std
        arr = np.expand_dims(arr, axis=0).astype(self.input_details[0]["dtype"])
        return arr, orig_size

    # ------------------------------------------------------------------
    def _run_inference(self, input_tensor):
        self.interpreter.set_tensor(self.input_details[0]["index"], input_tensor)
        self.interpreter.invoke()
        outputs = [self.interpreter.get_tensor(od["index"])
                   for od in self.output_details]
        return outputs

    # ------------------------------------------------------------------
    def _parse_outputs(self, outputs):
        """
        Gere plusieurs formats de sortie possibles selon l'export du modele :
          - un seul tenseur (1,6) = [phi_n, lambda_n, qw,qx,qy,qz]
          - deux tenseurs : (1,2) position + (1,4) quaternion
        """
        flat = [np.squeeze(o).astype(np.float64) for o in outputs]
        if len(flat) == 1 and flat[0].shape[-1] == 6:
            vec = flat[0]
            pos_n, quat = vec[:2], vec[2:]
        elif len(flat) == 2:
            a, b = flat
            if a.shape[-1] == 2 and b.shape[-1] == 4:
                pos_n, quat = a, b
            elif a.shape[-1] == 4 and b.shape[-1] == 2:
                pos_n, quat = b, a
            else:
                raise ValueError(f"Formes de sortie inattendues : {a.shape}, {b.shape}")
        else:
            raise ValueError(
                f"Format de sortie du modele APR non reconnu : "
                f"{[o.shape for o in flat]}. Adaptez _parse_outputs()."
            )
        quat = geo_utils.quat_normalize(quat)
        return pos_n, quat

    # ------------------------------------------------------------------
    def _denormalize_position(self, pos_n):
        lat_n, lon_n = pos_n
        lat = lat_n * (self.norm["lat_max"] - self.norm["lat_min"]) + self.norm["lat_mean"]
        lon = lon_n * (self.norm["lon_max"] - self.norm["lon_min"]) + self.norm["lon_mean"]
        return lat, lon

    # ------------------------------------------------------------------
    def predict(self, image_path, verbose=True):
        t0 = time.perf_counter()
        input_tensor, orig_size = self._preprocess(image_path)
        outputs = self._run_inference(input_tensor)
        pos_n, quat = self._parse_outputs(outputs)
        lat, lon = self._denormalize_position(pos_n)
        yaw, pitch, roll = geo_utils.quat_to_euler_deg(quat)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        result = {
            "lat": lat, "lon": lon,
            "quat": quat,
            "yaw_deg": yaw, "pitch_deg": pitch, "roll_deg": roll,
            "search_radius_m": config.LEVEL1_SEARCH_RADIUS_M,
            "expected_error_m": config.LEVEL1_EXPECTED_ERROR_M,
            "elapsed_ms": elapsed_ms,
            "image_size": orig_size,
        }
        if verbose:
            self._display(image_path, result)
        return result

    # ------------------------------------------------------------------
    @staticmethod
    def _display(image_path, r):
        disp.title("NIVEAU 1 - LOCALISATION GROSSIERE PAR APR (MobileNet-APR-Campus)")
        disp.kv("Image requete", image_path)
        disp.kv("Backend TFLite", _TFLITE_BACKEND or "indisponible")
        disp.section("Pose predite (brute)")
        disp.kv("Latitude estimee", f"{r['lat']:.7f}", "deg")
        disp.kv("Longitude estimee", f"{r['lon']:.7f}", "deg")
        disp.kv("Quaternion (qw,qx,qy,qz)",
                 f"({r['quat'][0]:.4f}, {r['quat'][1]:.4f}, "
                 f"{r['quat'][2]:.4f}, {r['quat'][3]:.4f})")
        disp.kv("Orientation (yaw / pitch / roll)",
                 f"{r['yaw_deg']:.2f} / {r['pitch_deg']:.2f} / {r['roll_deg']:.2f}", "deg")
        disp.section("Zone candidate transmise au Niveau 2")
        disp.kv("Rayon de recherche", f"{r['search_radius_m']:.0f}", "m")
        disp.kv("Erreur mediane attendue (jeu de test)", f"{r['expected_error_m']:.2f}", "m")
        disp.section("Performance")
        disp.kv("Temps d'inference", f"{r['elapsed_ms']:.1f}", "ms")
        print()
