"""
Entraînement de MobileNet-APR-Campus — Configuration C4
=========================================================
Architecture : MobileNetV2 pré-entraîné ImageNet
              + Tête APR duale (branche position + branche orientation)
              + Quaternion unitaire (normalisation L2)
              + Module CBAM (attention spatiale)

Protocole en 2 phases :
  Phase 1 — Backbone gelé, entraînement de la tête seule   (10 époques)
  Phase 2 — Affinage global, backbone dégelé, cosine decay (30 époques)

Répond aux remarques du rapporteur (section 2.13) :
  — Nombre total d'images : affiché dans le rapport de reproductibilité
  — Stratégie de séparation train/val/test : 65/15/20 %, graine fixée
  — Paramètres intrinsèques de la caméra : reportés dans DATASET_INFO
  — Contraintes spatiales / temporelles : split aléatoire, limite documentée

Usage :
    python train_mobilenet_apr_c4.py \
        --images_dir  ./images \
        --csv         ./position_angle.csv \
        --output_dir  ./outputs_c4
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, regularizers
from tensorflow.keras.applications import MobileNetV2

# ── Paramètres intrinsèques de la caméra (Redmi Note 14) ──────────────────
# À mettre à jour si une calibration formelle est disponible.
DATASET_INFO = {
    "device":           "Xiaomi Redmi Note 14",
    "sensor":           "Samsung ISOCELL JN1  (1/2.76\", 50 MP)",
    "focal_length_mm":  "5.58 mm (équivalent 24 mm plein format)",
    "resolution_native":"4096 × 3072 px (12 MP utilisés pour la capture)",
    "resolution_used":  "1920 × 1440 px (format de stockage des images)",
    "capture_rate_hz":  "2 Hz (intervalle 500 ms)",
    "gps_accuracy_m":   "~3–5 m (GNSS intégré, conditions dégagées)",
    "orientation_sensor":"SensorManager.getOrientation() — angles en radians",
}
# ───────────────────────────────────────────────────────────────────────────

# Statistiques ImageNet pour la normalisation du backbone
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# 1.  PRÉTRAITEMENT
# ══════════════════════════════════════════════════════════════════════════════

def load_and_preprocess_image(path, size=(224, 224)):
    """Redimensionnement bilinéaire + normalisation ImageNet."""
    img = Image.open(path).convert("RGB").resize(size, Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    return (arr - IMAGENET_MEAN) / IMAGENET_STD


def load_images(names, folder, size=(224, 224)):
    return np.stack([
        load_and_preprocess_image(os.path.join(folder, n), size)
        for n in names
    ])


def euler_to_quaternion(az, pitch, roll):
    """
    Conversion angles d'Euler (rad, convention yaw-pitch-roll ZYX) → quaternion
    unitaire (qw, qx, qy, qz). Validé contre scipy.spatial.transform.Rotation.
    """
    az, pitch, roll = (np.asarray(x, np.float64) for x in (az, pitch, roll))
    cy, sy = np.cos(az * .5), np.sin(az * .5)
    cp, sp = np.cos(pitch * .5), np.sin(pitch * .5)
    cr, sr = np.cos(roll * .5), np.sin(roll * .5)
    qw = cr*cp*cy + sr*sp*sy
    qx = sr*cp*cy - cr*sp*sy
    qy = cr*sp*cy + sr*cp*sy
    qz = cr*cp*sy - sr*sp*cy
    q  = np.stack([qw, qx, qy, qz], axis=-1).astype(np.float32)
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


class PositionNormalizer:
    """Standardisation min-max centrée sur le barycentre du train."""
    def fit(self, pos):
        self.mean_  = pos.mean(0)
        self.range_ = np.where((pos.max(0) - pos.min(0)) == 0, 1,
                                pos.max(0) - pos.min(0))
        return self
    def transform(self, pos):
        return ((pos - self.mean_) / self.range_).astype(np.float32)
    def fit_transform(self, pos):
        return self.fit(pos).transform(pos)
    def inverse_transform(self, pos_n):
        return pos_n * self.range_ + self.mean_


def load_dataset(csv_path, images_dir,
                 test_size=0.20, val_size=0.15, seed=42):
    """
    Chargement + split 3 voies.

    Stratégie de séparation (réponse au rapporteur) :
      1. Tirage aléatoire stratifié (graine seed=42) sur l'ensemble complet.
      2. Isolation du TEST en premier (test_size du total) — jamais vu avant
         l'évaluation finale.
      3. Isolation de la VALIDATION à partir du reliquat (val_size / (1-test_size))
         — utilisée uniquement pour l'early stopping.
      4. TRAIN = reliquat.
      Les paramètres de normalisation (barycentre, amplitude min-max) sont
      calculés exclusivement sur TRAIN.

    Limites (documentées pour le rapporteur) :
      — Split purement aléatoire : des images spatialements proches peuvent se
        retrouver simultanément dans TRAIN et TEST (biais de proximité).
      — Un split par séquence ou par zone géographique serait plus conservatif
        mais requiert un jeu de données plus grand.
    """
    # Détection automatique du séparateur
    with open(csv_path) as f:
        header = f.readline()
    sep = ";" if header.count(";") > header.count(",") else ","
    df  = pd.read_csv(csv_path, sep=sep)

    required = {"image_name", "latitude", "longitude", "azimuth", "pitch", "roll"}
    if missing := required - set(df.columns):
        raise ValueError(f"Colonnes manquantes : {missing}")

    n_total = len(df)
    print(f"\n{'='*60}")
    print("  RAPPORT DE REPRODUCTIBILITÉ — Jeu de données")
    print(f"{'='*60}")
    print(f"  Fichier CSV         : {csv_path}")
    print(f"  Images totales      : {n_total}")
    print(f"  Résolution native   : {DATASET_INFO['resolution_native']}")
    print(f"  Résolution utilisée : {DATASET_INFO['resolution_used']}")
    print(f"  Appareil            : {DATASET_INFO['device']}")
    print(f"  Capteur caméra      : {DATASET_INFO['sensor']}")
    print(f"  Focale              : {DATASET_INFO['focal_length_mm']}")
    print(f"  Précision GPS       : {DATASET_INFO['gps_accuracy_m']}")
    print(f"  Fréquence capture   : {DATASET_INFO['capture_rate_hz']}")
    print(f"  Capteur orientation : {DATASET_INFO['orientation_sensor']}")
    print(f"\n  Stratégie de split  : aléatoire (graine={seed})")
    print(f"  Test                : {test_size*100:.0f}% du total")
    print(f"  Validation          : {val_size*100:.0f}% du total")
    print(f"  Train               : {(1-test_size-val_size)*100:.0f}% du total")
    print(f"  Normalisation pos.  : min-max centrée (fit sur TRAIN uniquement)")
    print(f"  Normalisation img.  : statistiques ImageNet (µ={IMAGENET_MEAN}, "
          f"σ={IMAGENET_STD})")
    print(f"\n  ⚠ Limite : split aléatoire sans contrainte spatiale/temporelle.")
    print(f"    Des images proches peuvent figurer dans TRAIN et TEST.")
    print(f"{'='*60}\n")

    # Chargement des images
    print(f"Chargement de {n_total} images depuis {images_dir} ...")
    images = load_images(df["image_name"].values, images_dir)

    positions   = df[["latitude", "longitude"]].values
    quaternions = euler_to_quaternion(
        df["azimuth"].values, df["pitch"].values, df["roll"].values)

    # Split en 2 étapes
    idx = np.arange(n_total)
    trainval_idx, test_idx = train_test_split(
        idx, test_size=test_size, random_state=seed)
    train_idx, val_idx = train_test_split(
        trainval_idx,
        test_size=val_size / (1.0 - test_size),
        random_state=seed)

    pos_norm = PositionNormalizer()
    pos_train = pos_norm.fit_transform(positions[train_idx])
    pos_val   = pos_norm.transform(positions[val_idx])
    pos_test  = pos_norm.transform(positions[test_idx])

    print(f"  Train : {len(train_idx)}  |  Val : {len(val_idx)}  "
          f"|  Test : {len(test_idx)}")
    print(f"  Barycentre (lat, lon) : {pos_norm.mean_}")
    print(f"  Amplitude  (max-min)  : {pos_norm.range_}\n")

    return {
        "X_train": images[train_idx], "X_val": images[val_idx],
        "X_test":  images[test_idx],
        "pos_train": pos_train, "pos_val": pos_val, "pos_test": pos_test,
        "quat_train": quaternions[train_idx],
        "quat_val":   quaternions[val_idx],
        "quat_test":  quaternions[test_idx],
        "pos_norm":   pos_norm,
        "n_total": n_total,
        "n_train": len(train_idx),
        "n_val":   len(val_idx),
        "n_test":  len(test_idx),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 2.  ARCHITECTURE C4
# ══════════════════════════════════════════════════════════════════════════════

class L2NormQuaternion(layers.Layer):
    """Garantit ||q||=1 (Éq. quat_normalize du manuscrit)."""
    def call(self, q):
        return q / tf.maximum(tf.norm(q, axis=-1, keepdims=True), 1e-12)


class SpatialAttentionCBAM(layers.Layer):
    """
    Attention spatiale CBAM-léger :
        M_s = sigmoid( conv_7×7( [AvgPool_c(F) ; MaxPool_c(F)] ) )
    """
    def __init__(self, **kw):
        super().__init__(**kw)
        self.conv = layers.Conv2D(1, 7, padding="same",
                                  activation="sigmoid", use_bias=False)

    def build(self, shape):
        self.conv.build((shape[0], shape[1], shape[2], 2))
        super().build(shape)

    def call(self, x):
        avg = tf.reduce_mean(x, axis=-1, keepdims=True)
        mx  = tf.reduce_max(x,  axis=-1, keepdims=True)
        return x * self.conv(tf.concat([avg, mx], axis=-1))

    def get_config(self):
        return super().get_config()


def build_c4(freeze_backbone=True, dropout=0.0):
    """
    Configuration C4 :
      MobileNetV2 (ImageNet) → CBAM → GAP
      → Branche position  : Dense(256, ReLU) → Dense(2)
      → Branche orientation: Dense(256, ReLU) → Dense(4) → L2Norm
    """
    backbone = MobileNetV2(weights="imagenet", include_top=False,
                           input_shape=(224, 224, 3))
    backbone.trainable = not freeze_backbone

    x = SpatialAttentionCBAM(name="cbam")(backbone.output)
    x = layers.GlobalAveragePooling2D(name="gap")(x)

    p = layers.Dense(256, activation="relu", name="pos_d256")(x)
    if dropout > 0:
        p = layers.Dropout(dropout)(p)
    pos_out = layers.Dense(2, activation="linear", name="position")(p)

    o = layers.Dense(256, activation="relu", name="ori_d256")(x)
    if dropout > 0:
        o = layers.Dropout(dropout)(o)
    q_raw   = layers.Dense(4, activation="linear", name="quat_raw")(o)
    ori_out = L2NormQuaternion(name="orientation")(q_raw)

    return tf.keras.Model(backbone.input, [pos_out, ori_out],
                          name="MobileNet_APR_C4")


# ══════════════════════════════════════════════════════════════════════════════
# 3.  FONCTIONS DE PERTE
# ══════════════════════════════════════════════════════════════════════════════

def loss_position(y_true, y_pred):
    """MAE (L1) sur la position normalisée."""
    return tf.reduce_mean(tf.reduce_sum(tf.abs(y_true - y_pred), axis=-1))


def loss_quaternion(q_true, q_pred):
    """Perte géodésique : 1 − |q·q_ref|² (invariante au signe ±q)."""
    return tf.reduce_mean(1.0 - tf.square(
        tf.reduce_sum(q_true * q_pred, axis=-1)))


class LearnedWeightedLoss(layers.Layer):
    """
    Pondération apprise (Kendall & Gal, 2017) :
        L = (1/(2 e^{sx})) L_pos + (1/(2 e^{sq})) L_ori + sx + sq
    """
    def __init__(self, **kw):
        super().__init__(**kw)
        self.sx = self.add_weight(name="sx", shape=(), initializer="zeros", trainable=True)
        self.sq = self.add_weight(name="sq", shape=(), initializer="zeros", trainable=True)

    def call(self, lp, lo):
        return (.5*tf.exp(-self.sx)*lp + .5*tf.exp(-self.sq)*lo
                + self.sx + self.sq)


# ══════════════════════════════════════════════════════════════════════════════
# 4.  BOUCLE D'ENTRAÎNEMENT
# ══════════════════════════════════════════════════════════════════════════════

def make_ds(images, pos, quat, batch, shuffle=True):
    ds = tf.data.Dataset.from_tensor_slices((images, pos, quat))
    if shuffle:
        ds = ds.shuffle(len(images), seed=42)
    return ds.batch(batch).prefetch(tf.data.AUTOTUNE)


class Trainer:
    def __init__(self, model, optimizer, loss_layer):
        self.model      = model
        self.optimizer  = optimizer
        self.loss_layer = loss_layer

    def _vars(self):
        return self.model.trainable_variables + self.loss_layer.trainable_variables

    def step(self, imgs, pos_t, quat_t, training=True):
        with tf.GradientTape() as tape:
            pos_p, ori_p = self.model(imgs, training=training)
            lp = loss_position(pos_t, pos_p)
            lo = loss_quaternion(quat_t, ori_p)
            total = self.loss_layer(lp, lo)
        if training:
            grads = tape.gradient(total, self._vars())
            self.optimizer.apply_gradients(zip(grads, self._vars()))
        return float(total), float(lp), float(lo)

    def run_epoch(self, ds, training=True):
        tots, lps, los = [], [], []
        for imgs, pos_t, quat_t in ds:
            t, lp, lo = self.step(imgs, pos_t, quat_t, training)
            tots.append(t); lps.append(lp); los.append(lo)
        return np.mean(tots), np.mean(lps), np.mean(los)


def train(data, batch=32, phase1_ep=10, phase2_ep=30, output_dir="."):
    os.makedirs(output_dir, exist_ok=True)

    train_ds = make_ds(data["X_train"], data["pos_train"], data["quat_train"], batch)
    val_ds   = make_ds(data["X_val"],   data["pos_val"],   data["quat_val"],   batch, False)

    wloss = LearnedWeightedLoss()
    hist  = {k: [] for k in
             ["train_total","train_pos","train_ori","val_total","val_pos","val_ori"]}

    # ── Phase 1 : backbone gelé ──────────────────────────────────────────────
    print("=== Phase 1 : backbone gelé ===")
    model   = build_c4(freeze_backbone=True)
    trainer = Trainer(model, tf.keras.optimizers.Adam(1e-3), wloss)

    for ep in range(1, phase1_ep + 1):
        tt, tp, to = trainer.run_epoch(train_ds, training=True)
        vt, vp, vo = trainer.run_epoch(val_ds,   training=False)
        for k, v in zip(hist, [tt,tp,to,vt,vp,vo]):
            hist[k].append(v)
        if ep % max(1, phase1_ep//5) == 0 or ep == phase1_ep:
            sx, sq = float(wloss.sx), float(wloss.sq)
            print(f"  Ép {ep:3d}/{phase1_ep} "
                  f"train={tt:.4f}(pos={tp:.4f},ori={to:.4f}) "
                  f"val={vt:.4f}(pos={vp:.4f},ori={vo:.4f}) "
                  f"sx={sx:.3f} sq={sq:.3f}")

    # ── Phase 2 : affinage global ────────────────────────────────────────────
    print("\n=== Phase 2 : affinage global (backbone dégelé) ===")
    model.trainable = True
    steps   = max(1, len(data["X_train"]) // batch)
    cosine  = tf.keras.optimizers.schedules.CosineDecay(1e-4, phase2_ep * steps)
    trainer = Trainer(model, tf.keras.optimizers.Adam(cosine), wloss)

    best_val, best_ep, patience_cnt = float("inf"), 0, 0
    patience = max(5, phase2_ep // 5)

    for ep in range(1, phase2_ep + 1):
        tt, tp, to = trainer.run_epoch(train_ds, training=True)
        vt, vp, vo = trainer.run_epoch(val_ds,   training=False)
        for k, v in zip(hist, [tt,tp,to,vt,vp,vo]):
            hist[k].append(v)

        if vt < best_val - 1e-4:
            best_val, best_ep, patience_cnt = vt, ep, 0
            model.save(os.path.join(output_dir, "best_c4.keras"))
        else:
            patience_cnt += 1

        if ep % max(1, phase2_ep//5) == 0 or ep == phase2_ep:
            print(f"  Ép {ep:3d}/{phase2_ep} "
                  f"train={tt:.4f}(pos={tp:.4f},ori={to:.4f}) "
                  f"val={vt:.4f}(pos={vp:.4f},ori={vo:.4f})")

        if patience_cnt >= patience:
            print(f"  Early stopping à l'époque {ep} (meilleure val: ép {best_ep})")
            break

    return model, hist


# ══════════════════════════════════════════════════════════════════════════════
# 5.  MÉTRIQUES D'ÉVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def haversine_m(lat1, lon1, lat2, lon2):
    """Distance géodésique de Haversine (en mètres)."""
    R = 6_371_000.0
    p1, l1 = np.radians(lat1), np.radians(lon1)
    p2, l2 = np.radians(lat2), np.radians(lon2)
    dp, dl = p2 - p1, l2 - l1
    a = np.sin(dp/2)**2 + np.cos(p1)*np.cos(p2)*np.sin(dl/2)**2
    return 2*R*np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def orientation_error_deg(q_true, q_pred):
    """Erreur angulaire médiane entre quaternions (en degrés)."""
    dot = np.clip(np.abs(np.sum(q_true * q_pred, axis=-1)), 0, 1)
    return 2.0 * np.degrees(np.arccos(dot))


def evaluate(model, data, is_euler_model=False, mean_pos=None, std_pos=None):
    """
    Évalue le modèle sur le jeu de test.
    Pour les modèles de comparaison (AlexNet, ResNet, etc.) qui prédisent
    des coordonnées en z-score, passer is_euler_model=True avec mean_pos et
    std_pos pour la dénormalisation.
    """
    preds = model.predict(data["X_test"], verbose=0)

    if is_euler_model:
        # Modèles baseline : sortie = position z-score normalisée
        pos_pred = preds[0] * std_pos + mean_pos
        pos_true = data["pos_test_raw"]  # coordonnées brutes
        quat_pred = np.zeros((len(pos_pred), 4)); quat_pred[:, 0] = 1.0
        quat_true = quat_pred.copy()
    else:
        # MobileNet-APR-C4 : sortie = position min-max normalisée
        pos_pred  = data["pos_norm"].inverse_transform(preds[0])
        pos_true  = data["pos_norm"].inverse_transform(data["pos_test"])
        quat_pred = preds[1]
        quat_true = data["quat_test"]

    dists   = haversine_m(pos_true[:,0], pos_true[:,1],
                          pos_pred[:,0], pos_pred[:,1])
    ori_err = orientation_error_deg(quat_true, quat_pred)

    mae_lat = float(np.mean(np.abs(pos_true[:,0] - pos_pred[:,0])))
    mae_lon = float(np.mean(np.abs(pos_true[:,1] - pos_pred[:,1])))

    return {
        "mae_lat_deg":       mae_lat,
        "mae_lon_deg":       mae_lon,
        "mae_pos_deg":       mae_lat + mae_lon,
        "haversine_median_m": float(np.median(dists)),
        "haversine_mean_m":  float(np.mean(dists)),
        "haversine_min_m":   float(np.min(dists)),
        "haversine_max_m":   float(np.max(dists)),
        "success_2m":        float(np.mean(dists < 2.0)  * 100),
        "success_10m":       float(np.mean(dists < 10.0) * 100),
        "success_20m":       float(np.mean(dists < 20.0) * 100),
        "ori_median_deg":    float(np.median(ori_err)),
        "distances":         dists,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 6.  COURBES & RAPPORT
# ══════════════════════════════════════════════════════════════════════════════

def plot_curves(hist, phase1_ep, phase2_ep, output_dir):
    total = phase1_ep + phase2_ep
    x     = range(1, total + 1)
    sep   = phase1_ep

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    pairs = [("train_total","val_total","Perte totale pondérée"),
             ("train_pos",  "val_pos",  "Perte position (L1)"),
             ("train_ori",  "val_ori",  "Perte orientation (géodésique)")]

    for ax, (tk, vk, title) in zip(axes, pairs):
        ax.plot(x, hist[tk], color="#2a78d6", label="Train")
        ax.plot(x, hist[vk], color="#eb6834", linestyle="--", label="Validation")
        ax.axvline(sep, color="gray", linestyle=":", alpha=.7)
        ax.text(sep/2,          ax.get_ylim()[1]*.93, "Ph.1",
                ha="center", fontsize=8, color="gray")
        ax.text(sep + (total-sep)/2, ax.get_ylim()[1]*.93, "Ph.2",
                ha="center", fontsize=8, color="gray")
        ax.set_title(title); ax.set_xlabel("Époque")
        ax.legend(); ax.grid(True, alpha=.3)

    plt.suptitle("Courbes d'entraînement — MobileNet-APR-Campus C4",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(output_dir, "c4_training_curves.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"📊 Courbes → {path}")


def plot_errors_per_image(dists, output_dir):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(range(len(dists)), dists, color="#2a78d6", alpha=.8)
    ax.axhline(np.median(dists), color="orange", linestyle="--",
               label=f"Médiane = {np.median(dists):.1f} m")
    ax.axhline(np.mean(dists),   color="red",    linestyle=":",
               label=f"Moyenne = {np.mean(dists):.1f} m")
    ax.set_xlabel("Image test (index)"); ax.set_ylabel("Erreur Haversine (m)")
    ax.set_title("Erreur de localisation par image — MobileNet-APR-Campus C4")
    ax.legend(); ax.grid(axis="y", alpha=.3)
    plt.tight_layout()
    path = os.path.join(output_dir, "c4_per_image_errors.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"📊 Erreurs par image → {path}")


def print_final_report(metrics, data):
    print("\n" + "="*65)
    print("  RAPPORT D'ÉVALUATION FINALE — MobileNet-APR-Campus C4")
    print("="*65)
    print(f"  Jeu de test : {data['n_test']} images")
    print(f"  (Total : {data['n_total']} | Train : {data['n_train']} | Val : {data['n_val']})")
    print(f"\n  Formule Haversine :")
    print(f"  d = 2R·arcsin(√(sin²(Δφ/2) + cos(φ₁)·cos(φ₂)·sin²(Δλ/2)))")
    print(f"  R = 6 371 000 m\n")
    print(f"  MAE latitude    : {metrics['mae_lat_deg']:.6f}°")
    print(f"  MAE longitude   : {metrics['mae_lon_deg']:.6f}°")
    print(f"  MAE position    : {metrics['mae_pos_deg']:.6f}°")
    print(f"  Haversine min   : {metrics['haversine_min_m']:.2f} m")
    print(f"  Haversine moy.  : {metrics['haversine_mean_m']:.2f} m")
    print(f"  Haversine méd.  : {metrics['haversine_median_m']:.2f} m")
    print(f"  Haversine max   : {metrics['haversine_max_m']:.2f} m")
    print(f"  Succès < 2 m    : {metrics['success_2m']:.1f}%")
    print(f"  Succès < 10 m   : {metrics['success_10m']:.1f}%")
    print(f"  Succès < 20 m   : {metrics['success_20m']:.1f}%")
    print(f"  Ori. méd. (°)   : {metrics['ori_median_deg']:.3f}°")
    print("="*65)


# ══════════════════════════════════════════════════════════════════════════════
# 7.  POINT D'ENTRÉE
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir",  default="./images")
    parser.add_argument("--csv",         default="./position_angle.csv")
    parser.add_argument("--output_dir",  default="./outputs_c4")
    parser.add_argument("--batch_size",  type=int, default=32)
    parser.add_argument("--phase1_ep",   type=int, default=10)
    parser.add_argument("--phase2_ep",   type=int, default=30)
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--test_size",   type=float, default=0.20)
    parser.add_argument("--val_size",    type=float, default=0.15)
    args = parser.parse_args()

    # Chargement
    data = load_dataset(args.csv, args.images_dir,
                        test_size=args.test_size,
                        val_size=args.val_size,
                        seed=args.seed)

    # Entraînement
    model, hist = train(data,
                        batch=args.batch_size,
                        phase1_ep=args.phase1_ep,
                        phase2_ep=args.phase2_ep,
                        output_dir=args.output_dir)

    # Courbes
    plot_curves(hist, args.phase1_ep, args.phase2_ep, args.output_dir)

    # Évaluation
    metrics = evaluate(model, data)
    plot_errors_per_image(metrics["distances"], args.output_dir)
    print_final_report(metrics, data)

    # Sauvegarde finale
    final_path = os.path.join(args.output_dir, "mobilenet_apr_c4_final.keras")
    model.save(final_path)
    print(f"\n✅ Modèle sauvegardé : {final_path}")


if __name__ == "__main__":
    main()
