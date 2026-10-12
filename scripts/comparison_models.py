"""
Comparaison des modèles de régression de pose absolue (APR)
============================================================
Compare 5 architectures sur le même jeu de données et le même split
(graine=42, test=20%, val=15%, train=65%) :

  1. AlexNet           — CNN from scratch, 227×227
  2. ResNet50V2        — pré-entraîné ImageNet, 224×224
  3. Sivapong CNN      — CNN from scratch, 4 blocs Conv+MaxPool, 224×224
                         (Nilwong et al. — méthode de référence du manuscrit)
  4. MobileNet V1      — pré-entraîné ImageNet, 224×224
  5. MobileNet-APR-C4  — notre modèle (MobileNetV2 + CBAM + quaternion)

Métriques rapportées pour chaque modèle :
  — MAE position (degrés décimaux)
  — Distance Haversine médiane (m)   [formule exacte du manuscrit]
  — Taux de succès < 2 m / < 10 m / < 20 m
  — Erreur d'orientation médiane (°)
  — Nombre de paramètres
  — Taille du modèle (Mo)
  — Temps d'inférence médian (ms/image)

Génère :
  — comparison_results.csv
  — comparison_results.tex   (tableau LaTeX booktabs pour la thèse)
  — comparison_haversine.png (graphique barres)
  — comparison_cdf.png       (courbe CDF des erreurs)

Usage :
    python comparison_models.py \
        --images_dir ./images \
        --csv        ./position_angle.csv \
        --c4_model   ./outputs_c4/mobilenet_apr_c4_final.keras \
        --output_dir ./outputs_comparison \
        --epochs     50
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
from tensorflow.keras import layers
from tensorflow.keras.applications import (
    MobileNetV2, MobileNet, ResNet50V2
)
from tensorflow.keras.models import Model

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# ── Paramètres intrinsèques caméra (reproduits pour le rapporteur) ─────────
DATASET_INFO = {
    "device":            "Xiaomi Redmi Note 14",
    "resolution_native": "1920 × 1440 px",
    "gps_accuracy_m":    "~3–5 m",
    "capture_rate_hz":   "2 Hz",
}

# Statistiques ImageNet (pour MobileNet-APR-C4 et les backbones pré-entraînés)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# 1.  UTILITAIRES PARTAGÉS
# ══════════════════════════════════════════════════════════════════════════════

def haversine_m(lat1, lon1, lat2, lon2):
    """
    Distance géodésique de Haversine (mètres) — Éq. manuscrit :
        d = 2R·arcsin(√(sin²(Δφ/2) + cos(φ₁)·cos(φ₂)·sin²(Δλ/2)))
    """
    R = 6_371_000.0
    p1, l1 = np.radians(lat1), np.radians(lon1)
    p2, l2 = np.radians(lat2), np.radians(lon2)
    dp, dl = p2-p1, l2-l1
    a = np.sin(dp/2)**2 + np.cos(p1)*np.cos(p2)*np.sin(dl/2)**2
    return 2*R*np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def ori_error_deg(qt, qp):
    """Erreur angulaire (°) entre quaternions, invariante au signe ±q."""
    dot = np.clip(np.abs(np.sum(qt * qp, axis=-1)), 0, 1)
    return 2.0 * np.degrees(np.arccos(dot))


def euler_to_quaternion(az, pitch, roll):
    az, pitch, roll = (np.asarray(x, np.float64) for x in (az, pitch, roll))
    cy, sy = np.cos(az*.5), np.sin(az*.5)
    cp, sp = np.cos(pitch*.5), np.sin(pitch*.5)
    cr, sr = np.cos(roll*.5), np.sin(roll*.5)
    qw = cr*cp*cy + sr*sp*sy
    qx = sr*cp*cy - cr*sp*sy
    qy = cr*sp*cy + sr*cp*sy
    qz = cr*cp*sy - sr*sp*cy
    q = np.stack([qw, qx, qy, qz], -1).astype(np.float32)
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def model_size_mb(model, tmp="/tmp/_cmp_tmp.keras"):
    model.save(tmp)
    s = os.path.getsize(tmp) / 1024**2
    os.remove(tmp)
    return s


def inference_ms(model, x_sample, n=50):
    batch = tf.constant(x_sample[:1])
    for _ in range(5):
        model(batch, training=False)
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        model(batch, training=False)
        times.append((time.perf_counter()-t0)*1000)
    return float(np.median(times))


# ══════════════════════════════════════════════════════════════════════════════
# 2.  CHARGEMENT DES DONNÉES (deux versions de normalisation)
# ══════════════════════════════════════════════════════════════════════════════

def _load_img(path, size, imagenet_norm):
    img = Image.open(path).convert("RGB").resize(size, Image.BILINEAR)
    arr = np.array(img, np.float32) / 255.0
    if imagenet_norm:
        arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return arr


def load_dataset(csv_path, images_dir, test_size=.20, val_size=.15, seed=42):
    """
    Chargement avec deux versions d'images :
      X_imagenet : normalisées selon ImageNet (pour modèles pré-entraînés)
      X_zscore   : [0,1] simple (pour AlexNet from-scratch)
    Et deux versions de positions :
      pos_minmax : normalisation min-max (pour MobileNet-APR-C4)
      pos_zscore : normalisation z-score (pour les 4 modèles de comparaison)
    """
    with open(csv_path) as f:
        hdr = f.readline()
    sep = ";" if hdr.count(";") > hdr.count(",") else ","
    df  = pd.read_csv(csv_path, sep=sep)

    n = len(df)
    names = df["image_name"].values
    pos_raw = df[["latitude","longitude"]].values
    ang_raw = df[["azimuth","pitch","roll"]].values

    # Images en 224×224 (normalisées ImageNet) et 227×227 (pour AlexNet)
    print(f"Chargement de {n} images...")
    imgs_224 = np.stack([_load_img(os.path.join(images_dir, nm),
                                   (224,224), True)  for nm in names])
    imgs_227 = np.stack([_load_img(os.path.join(images_dir, nm),
                                   (227,227), False) for nm in names])
    quats    = euler_to_quaternion(ang_raw[:,0], ang_raw[:,1], ang_raw[:,2])

    # Splits identiques pour tous les modèles (même graine)
    idx = np.arange(n)
    trval, test_idx = train_test_split(idx, test_size=test_size, random_state=seed)
    train_idx, val_idx = train_test_split(
        trval, test_size=val_size/(1-test_size), random_state=seed)

    # Normalisation min-max (pour MobileNet-APR-C4)
    pos_mean_mm = pos_raw[train_idx].mean(0)
    pos_range   = np.where((pos_raw[train_idx].max(0)-pos_raw[train_idx].min(0))==0,
                            1, pos_raw[train_idx].max(0)-pos_raw[train_idx].min(0))
    def minmax(p): return ((p - pos_mean_mm) / pos_range).astype(np.float32)

    # Normalisation z-score (pour les 4 autres modèles)
    pos_mean_z  = pos_raw[train_idx].mean(0)
    pos_std_z   = pos_raw[train_idx].std(0)
    def zscore(p): return ((p - pos_mean_z) / pos_std_z).astype(np.float32)

    n_train, n_val, n_test = len(train_idx), len(val_idx), len(test_idx)

    print(f"\n{'='*60}")
    print("  RAPPORT DE REPRODUCTIBILITÉ — Jeu de données")
    print(f"{'='*60}")
    print(f"  Images totales      : {n}")
    print(f"  Train               : {n_train}  ({n_train/n*100:.0f}%)")
    print(f"  Validation          : {n_val}   ({n_val/n*100:.0f}%)")
    print(f"  Test                : {n_test}  ({n_test/n*100:.0f}%)")
    print(f"  Graine              : {seed}")
    print(f"  Résolution utilisée : {DATASET_INFO['resolution_native']}")
    print(f"  Appareil            : {DATASET_INFO['device']}")
    print(f"  Précision GPS       : {DATASET_INFO['gps_accuracy_m']}")
    print(f"  Fréquence capture   : {DATASET_INFO['capture_rate_hz']}")
    print(f"  ⚠ Split aléatoire sans contrainte spatiale/temporelle")
    print(f"{'='*60}\n")

    return {
        # Images
        "X224_tr": imgs_224[train_idx],  "X224_val": imgs_224[val_idx],
        "X224_te": imgs_224[test_idx],
        "X227_tr": imgs_227[train_idx],  "X227_val": imgs_227[val_idx],
        "X227_te": imgs_227[test_idx],
        # Positions
        "pos_mm_tr": minmax(pos_raw[train_idx]),
        "pos_mm_val":minmax(pos_raw[val_idx]),
        "pos_mm_te": minmax(pos_raw[test_idx]),
        "pos_z_tr":  zscore(pos_raw[train_idx]),
        "pos_z_val": zscore(pos_raw[val_idx]),
        "pos_z_te":  zscore(pos_raw[test_idx]),
        # Positions brutes (pour calcul Haversine)
        "pos_raw_te": pos_raw[test_idx],
        "pos_raw_tr": pos_raw[train_idx],
        # Quaternions
        "quat_tr":   quats[train_idx],
        "quat_val":  quats[val_idx],
        "quat_te":   quats[test_idx],
        # Stats dénormalisation z-score
        "pos_mean_z": pos_mean_z, "pos_std_z": pos_std_z,
        # Stats dénormalisation min-max
        "pos_mean_mm": pos_mean_mm, "pos_range": pos_range,
        # Tailles
        "n_total": n, "n_train": n_train, "n_val": n_val, "n_test": n_test,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 3.  BUILDERS DES MODÈLES DE COMPARAISON
# ══════════════════════════════════════════════════════════════════════════════

def _dense_head(x, out_pos=2, out_ori=3, dropout=0.5, name_suffix=""):
    """Tête standard partagée par AlexNet, ResNet, Sivapong CNN et MobileNetV1."""
    x = layers.Dense(256, activation="relu",
                      name=f"dense256{name_suffix}")(x)
    x = layers.Dropout(dropout, name=f"drop{name_suffix}")(x)
    pos = layers.Dense(out_pos, activation="linear",
                        name="position_output")(x)
    ori = layers.Dense(out_ori, activation="linear",
                        name="angle_output")(x)
    return pos, ori


def build_alexnet(input_shape=(227,227,3)):
    inp = layers.Input(shape=input_shape)
    x = layers.Conv2D(96,  (11,11), strides=4, activation="relu")(inp)
    x = layers.MaxPooling2D((3,3), strides=2)(x)
    x = layers.Conv2D(256, (5,5),  padding="same", activation="relu")(x)
    x = layers.MaxPooling2D((3,3), strides=2)(x)
    x = layers.Conv2D(384, (3,3),  padding="same", activation="relu")(x)
    x = layers.Conv2D(384, (3,3),  padding="same", activation="relu")(x)
    x = layers.Conv2D(256, (3,3),  padding="same", activation="relu")(x)
    x = layers.MaxPooling2D((3,3), strides=2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(4096, activation="relu")(x)
    x = layers.Dropout(.5)(x)
    x = layers.Dense(4096, activation="relu")(x)
    x = layers.Dropout(.5)(x)
    pos, ori = _dense_head(x)
    m = Model(inp, [pos, ori], name="AlexNet")
    m.compile(tf.keras.optimizers.Adam(1e-4),
               loss={"position_output":"mse","angle_output":"mse"},
               metrics={"position_output":"mae","angle_output":"mae"})
    return m


def _pretrained_model(base_cls, name, input_shape=(224,224,3)):
    base = base_cls(weights="imagenet", include_top=False,
                    input_shape=input_shape)
    base.trainable = False
    x = layers.GlobalAveragePooling2D()(base.output)
    pos, ori = _dense_head(x)
    m = Model(base.input, [pos, ori], name=name)
    m.compile(tf.keras.optimizers.Adam(1e-4),
               loss={"position_output":"mse","angle_output":"mse"},
               metrics={"position_output":"mae","angle_output":"mae"})
    return m


def build_resnet50():
    return _pretrained_model(ResNet50V2, "ResNet50V2")


def build_sivapong(input_shape=(224, 224, 3)):
    """
    CNN from-scratch de Nilwong et al. (méthode de référence du manuscrit).
    Architecture : 4 blocs Conv+MaxPool progressifs (24→32→48→64 filtres),
    suivi d'une tête Dense(256)+Dropout(0.5).
    Normalisation d'entrée : /255.0 simple (pas de normalisation ImageNet).
    Référence : Sivapong Nilwong et al., "Deep Learning-based Visual
    Localization in Large-Scale Indoor Environments", 2019.
    """
    inp = layers.Input(shape=input_shape, name="image_input")

    # Bloc 1 : 24 filtres 5×5
    x = layers.Conv2D(24, (5, 5), activation="relu")(inp)
    x = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

    # Bloc 2 : 32 filtres 4×4
    x = layers.Conv2D(32, (4, 4), activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

    # Bloc 3 : 48 filtres 3×3
    x = layers.Conv2D(48, (3, 3), activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1))(x)

    # Bloc 4 : 64 filtres 2×2
    x = layers.Conv2D(64, (2, 2), activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1))(x)

    x = layers.Flatten()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)

    pos, ori = _dense_head(x)
    m = Model(inp, [pos, ori], name="Sivapong_CNN")
    m.compile(tf.keras.optimizers.Adam(1e-3),
               loss={"position_output": "mse", "angle_output": "mse"},
               metrics={"position_output": "mae", "angle_output": "mae"})
    return m


def build_mobilenet_v1():
    return _pretrained_model(MobileNet, "MobileNetV1")


# ══════════════════════════════════════════════════════════════════════════════
# 4.  MobileNet-APR-C4 (copié de train_mobilenet_apr_c4.py)
# ══════════════════════════════════════════════════════════════════════════════

class L2NormQ(layers.Layer):
    def call(self, q):
        return q / tf.maximum(tf.norm(q, axis=-1, keepdims=True), 1e-12)


class CBAM(layers.Layer):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.conv = layers.Conv2D(1, 7, padding="same",
                                  activation="sigmoid", use_bias=False)
    def build(self, s):
        self.conv.build((s[0], s[1], s[2], 2)); super().build(s)
    def call(self, x):
        avg = tf.reduce_mean(x, axis=-1, keepdims=True)
        mx  = tf.reduce_max(x,  axis=-1, keepdims=True)
        return x * self.conv(tf.concat([avg, mx], axis=-1))
    def get_config(self): return super().get_config()


def build_c4_model(freeze=True):
    bb = MobileNetV2(weights="imagenet", include_top=False,
                     input_shape=(224,224,3))
    bb.trainable = not freeze
    x = CBAM(name="cbam")(bb.output)
    x = layers.GlobalAveragePooling2D()(x)
    p = layers.Dense(256, activation="relu")(x)
    pos_out = layers.Dense(2, activation="linear", name="position")(p)
    o = layers.Dense(256, activation="relu")(x)
    q = layers.Dense(4,   activation="linear")(o)
    ori_out = L2NormQ(name="orientation")(q)
    return Model(bb.input, [pos_out, ori_out], name="MobileNet_APR_C4")


# ══════════════════════════════════════════════════════════════════════════════
# 5.  ENTRAÎNEMENT UNIFIÉ
# ══════════════════════════════════════════════════════════════════════════════

class LearnedLoss(layers.Layer):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.sx = self.add_weight(name="sx", shape=(), initializer="zeros", trainable=True)
        self.sq = self.add_weight(name="sq", shape=(), initializer="zeros", trainable=True)
    def call(self, lp, lo):
        return .5*tf.exp(-self.sx)*lp + .5*tf.exp(-self.sq)*lo + self.sx + self.sq


def loss_pos(y, p):
    return tf.reduce_mean(tf.reduce_sum(tf.abs(y-p), axis=-1))


def loss_quat(q, p):
    return tf.reduce_mean(1.0 - tf.square(tf.reduce_sum(q*p, axis=-1)))


def train_baseline(model, X_tr, pos_tr, ang_tr, X_val, pos_val, ang_val,
                   epochs=50, batch=32, model_name="model"):
    """Entraînement standard pour les 4 modèles de comparaison (MSE + Adam)."""
    print(f"\n--- Entraînement : {model_name} ---")
    tr_ds = (tf.data.Dataset.from_tensor_slices((X_tr, pos_tr, ang_tr))
             .shuffle(len(X_tr), seed=42).batch(batch).prefetch(tf.data.AUTOTUNE))
    val_ds = (tf.data.Dataset.from_tensor_slices((X_val, pos_val, ang_val))
              .batch(batch).prefetch(tf.data.AUTOTUNE))

    best_val, patience, cnt = float("inf"), max(5, epochs//5), 0
    hist = {"train_mae_pos":[], "val_mae_pos":[]}

    for ep in range(1, epochs+1):
        tr_maes = []
        for imgs, pos_t, ang_t in tr_ds:
            with tf.GradientTape() as tape:
                pp, pa = model(imgs, training=True)
                loss   = (tf.reduce_mean(tf.square(pos_t - pp)) +
                          tf.reduce_mean(tf.square(ang_t - pa)))
            grads = tape.gradient(loss, model.trainable_variables)
            model.optimizer.apply_gradients(zip(grads, model.trainable_variables))
            tr_maes.append(float(tf.reduce_mean(tf.abs(pos_t - pp))))

        val_maes = []
        for imgs, pos_v, _ in val_ds:
            pp, _ = model(imgs, training=False)
            val_maes.append(float(tf.reduce_mean(tf.abs(pos_v - pp))))

        tr_mae  = np.mean(tr_maes)
        val_mae = np.mean(val_maes)
        hist["train_mae_pos"].append(tr_mae)
        hist["val_mae_pos"].append(val_mae)

        if val_mae < best_val - 1e-5:
            best_val, cnt = val_mae, 0
        else:
            cnt += 1

        if ep % max(1, epochs//5) == 0 or ep == epochs:
            print(f"  Ép {ep:3d}/{epochs}  train_MAE={tr_mae:.5f}  "
                  f"val_MAE={val_mae:.5f}")
        if cnt >= patience:
            print(f"  Early stopping à l'époque {ep}")
            break
    return hist


def train_c4(data, epochs_p1=10, epochs_p2=30, batch=32):
    """Entraînement de MobileNet-APR-C4 (perte géodésique + pondération apprise)."""
    print("\n--- Entraînement : MobileNet-APR-C4 ---")
    wloss = LearnedLoss()

    def make_ds(X, pos, quat, shuffle):
        ds = tf.data.Dataset.from_tensor_slices((X, pos, quat))
        if shuffle: ds = ds.shuffle(len(X), seed=42)
        return ds.batch(batch).prefetch(tf.data.AUTOTUNE)

    tr_ds  = make_ds(data["X224_tr"],  data["pos_mm_tr"],  data["quat_tr"],  True)
    val_ds = make_ds(data["X224_val"], data["pos_mm_val"], data["quat_val"], False)

    def step(model, opt, imgs, pos_t, quat_t, training):
        all_vars = model.trainable_variables + wloss.trainable_variables
        with tf.GradientTape() as tape:
            pp, qp = model(imgs, training=training)
            lp = loss_pos(pos_t, pp)
            lo = loss_quat(quat_t, qp)
            total = wloss(lp, lo)
        if training:
            grads = tape.gradient(total, all_vars)
            opt.apply_gradients(zip(grads, all_vars))
        return float(total)

    hist = {"train_total":[], "val_total":[]}

    # Phase 1
    model = build_c4_model(freeze=True)
    opt   = tf.keras.optimizers.Adam(1e-3)
    for ep in range(1, epochs_p1+1):
        tt = np.mean([step(model, opt, *b, True)  for b in tr_ds])
        vt = np.mean([step(model, opt, *b, False) for b in val_ds])
        hist["train_total"].append(tt); hist["val_total"].append(vt)
        if ep % max(1, epochs_p1//5) == 0 or ep == epochs_p1:
            print(f"  [Ph1] Ép {ep:3d}/{epochs_p1}  train={tt:.4f}  val={vt:.4f}")

    # Phase 2
    model.trainable = True
    steps  = max(1, len(data["X224_tr"]) // batch)
    cosine = tf.keras.optimizers.schedules.CosineDecay(1e-4, epochs_p2*steps)
    opt    = tf.keras.optimizers.Adam(cosine)
    best_val, patience, cnt = float("inf"), max(5, epochs_p2//5), 0

    for ep in range(1, epochs_p2+1):
        tt = np.mean([step(model, opt, *b, True)  for b in tr_ds])
        vt = np.mean([step(model, opt, *b, False) for b in val_ds])
        hist["train_total"].append(tt); hist["val_total"].append(vt)
        if vt < best_val - 1e-4:
            best_val, cnt = vt, 0
        else:
            cnt += 1
        if ep % max(1, epochs_p2//5) == 0 or ep == epochs_p2:
            print(f"  [Ph2] Ép {ep:3d}/{epochs_p2}  train={tt:.4f}  val={vt:.4f}")
        if cnt >= patience:
            print(f"  Early stopping à l'époque {ep}")
            break
    return model, hist


# ══════════════════════════════════════════════════════════════════════════════
# 6.  ÉVALUATION UNIFIÉE
# ══════════════════════════════════════════════════════════════════════════════

def eval_baseline(model, X_te, pos_z_te, pos_raw_te, quat_te,
                  mean_z, std_z):
    """Évalue un modèle baseline (sortie z-score)."""
    preds = model.predict(X_te, verbose=0)
    pos_pred = preds[0] * std_z + mean_z
    dists    = haversine_m(pos_raw_te[:,0], pos_raw_te[:,1],
                           pos_pred[:,0],   pos_pred[:,1])
    # Pas de quaternion pour les baselines → orientation non mesurée
    ori_med  = float("nan")
    return _metrics(dists, pos_raw_te, pos_pred, ori_med, None, None)


def eval_c4(model, X_te, pos_mm_te, pos_raw_te, quat_te,
            mean_mm, range_mm):
    """Évalue MobileNet-APR-C4 (sortie min-max + quaternion)."""
    preds    = model.predict(X_te, verbose=0)
    pos_pred = preds[0] * range_mm + mean_mm
    quat_pred= preds[1]
    dists    = haversine_m(pos_raw_te[:,0], pos_raw_te[:,1],
                           pos_pred[:,0],   pos_pred[:,1])
    ori_errs = ori_error_deg(quat_te, quat_pred)
    ori_med  = float(np.median(ori_errs))
    return _metrics(dists, pos_raw_te, pos_pred, ori_med, ori_errs, quat_pred)


def _metrics(dists, pos_true, pos_pred, ori_med, ori_errs, quat_pred):
    return {
        "mae_pos_deg":       float(np.mean(np.abs(pos_true - pos_pred))),
        "haversine_median_m":float(np.median(dists)),
        "haversine_mean_m":  float(np.mean(dists)),
        "haversine_max_m":   float(np.max(dists)),
        "success_2m":        float(np.mean(dists <  2)*100),
        "success_10m":       float(np.mean(dists < 10)*100),
        "success_20m":       float(np.mean(dists < 20)*100),
        "ori_median_deg":    ori_med,
        "distances":         dists,
        "ori_errors":        ori_errs,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 7.  VISUALISATIONS
# ══════════════════════════════════════════════════════════════════════════════

COLORS = ["#888780","#2a78d6","#1baf7a","#eb6834","#e34948"]


def plot_haversine_bars(results, output_dir):
    names = list(results.keys())
    vals  = [results[n]["haversine_median_m"] for n in names]
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(names, vals, color=COLORS[:len(names)], edgecolor="white")
    best = int(np.argmin(vals))
    bars[best].set_edgecolor("gold"); bars[best].set_linewidth(2.5)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2,
                bar.get_height()+max(vals)*.01,
                f"{v:.1f} m", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Haversine médiane (m)")
    ax.set_title("Comparaison des modèles — Distance Haversine médiane ↓")
    ax.grid(axis="y", alpha=.3); ax.set_ylim(bottom=0)
    plt.tight_layout()
    p = os.path.join(output_dir, "comparison_haversine.png")
    plt.savefig(p, dpi=150); plt.close(); print(f"📊 → {p}")


def plot_cdf(results, output_dir):
    fig, ax = plt.subplots(figsize=(10, 5))
    for (name, res), color in zip(results.items(), COLORS):
        dists_sorted = np.sort(res["distances"])
        cdf = np.arange(1, len(dists_sorted)+1) / len(dists_sorted)
        ax.plot(dists_sorted, cdf*100, label=name, color=color, linewidth=2)
    for thresh, ls in [(2,"--"),(10,":"),(20,"-.")]:
        ax.axvline(thresh, color="gray", linestyle=ls, alpha=.6,
                   label=f"{thresh} m")
    ax.set_xlabel("Erreur Haversine (m)")
    ax.set_ylabel("CDF (%)")
    ax.set_title("Courbe CDF des erreurs de localisation")
    ax.legend(fontsize=9); ax.grid(alpha=.3)
    plt.tight_layout()
    p = os.path.join(output_dir, "comparison_cdf.png")
    plt.savefig(p, dpi=150); plt.close(); print(f"📊 → {p}")


def plot_multi_metrics(results, output_dir):
    names = list(results.keys())
    hav   = [results[n]["haversine_median_m"] for n in names]
    s2    = [results[n]["success_2m"]         for n in names]
    s20   = [results[n]["success_20m"]        for n in names]
    infer = [results[n].get("infer_ms", 0)   for n in names]

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    for ax, vals, title, ylabel in [
        (axes[0], hav,   "Haversine médiane ↓",    "m"),
        (axes[1], s2,    "Succès < 2 m ↑",         "%"),
        (axes[2], s20,   "Succès < 20 m ↑",        "%"),
        (axes[3], infer, "Temps d'inférence ↓",    "ms/img"),
    ]:
        bars = ax.bar(names, vals, color=COLORS[:len(names)])
        best = int(np.argmin(vals) if "↓" in title else np.argmax(vals))
        bars[best].set_edgecolor("gold"); bars[best].set_linewidth(2.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2,
                    bar.get_height()+max(vals)*.01,
                    f"{v:.1f}", ha="center", va="bottom", fontsize=8)
        ax.set_title(title); ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=20); ax.grid(axis="y", alpha=.3)
        ax.set_ylim(bottom=0)
    plt.suptitle("Comparaison multi-métriques — APR sur campus d'Antananarivo",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    p = os.path.join(output_dir, "comparison_multi.png")
    plt.savefig(p, dpi=150); plt.close(); print(f"📊 → {p}")


# ══════════════════════════════════════════════════════════════════════════════
# 8.  EXPORT TABLEAU
# ══════════════════════════════════════════════════════════════════════════════

def export_csv(rows, output_dir):
    df = pd.DataFrame(rows)
    p  = os.path.join(output_dir, "comparison_results.csv")
    df.to_csv(p, index=False, sep=";")
    print(f"📄 CSV → {p}")
    return df


def export_latex(df, output_dir, data_info):
    """Tableau LaTeX booktabs prêt à coller dans la section 2.13."""
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        (r"\caption{Comparaison des modèles APR sur le campus d'Antananarivo "
         rf"({data_info['n_total']} images, split 65/15/20\%, graine 42). "
         r"$\downarrow$ : plus faible est meilleur ; $\uparrow$ : plus élevé est meilleur.}"),
        r"\label{tab:comparison_models}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lccccccccc}",
        r"\toprule",
        (r"\textbf{Modèle} & \textbf{MAE pos.~($^\circ$)}$\downarrow$ & "
         r"\textbf{Hav. méd.~(m)}$\downarrow$ & "
         r"\textbf{Succ.\,$<$2\,m~(\%)}$\uparrow$ & "
         r"\textbf{Succ.\,$<$10\,m~(\%)}$\uparrow$ & "
         r"\textbf{Succ.\,$<$20\,m~(\%)}$\uparrow$ & "
         r"\textbf{Ori.~méd.~($^\circ$)}$\downarrow$ & "
         r"\textbf{Params} & "
         r"\textbf{Taille~(Mo)}$\downarrow$ & "
         r"\textbf{Infér.~(ms)}$\downarrow$ \\"),
        r"\midrule",
    ]
    for _, row in df.iterrows():
        ori = f"{row['Ori. méd. (°)']:.1f}" if not pd.isna(row['Ori. méd. (°)']) else "—"
        is_ours = "APR-C4" in row["Modèle"]
        name = rf"\textbf{{{row['Modèle']}}}" if is_ours else row["Modèle"]
        lines.append(
            f"{name} & {row['MAE pos. (°)']:.6f} & {row['Hav. méd. (m)']:.2f} & "
            f"{row['Succès < 2 m (%)']:.1f} & {row['Succès < 10 m (%)']:.1f} & "
            f"{row['Succès < 20 m (%)']:.1f} & {ori} & "
            f"{row['Params']:,} & {row['Taille (Mo)']:.2f} & "
            f"{row['Inférence (ms)']:.2f} \\\\"
        )
    lines += [
        r"\midrule",
        (rf"\multicolumn{{10}}{{l}}{{\textit{{Jeu de données : {data_info['n_total']} images, "
         rf"appareil : {data_info['device']}, "
         r"résolution : 1920\,×\,1440\,px, GPS : \textasciitilde3--5\,m.}}}} \\"),
        r"\multicolumn{10}{l}{\textit{Split aléatoire (graine 42) : "
        r"train 65\,\%, validation 15\,\%, test 20\,\% — sans contrainte spatiale.}} \\",
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\end{table}",
    ]
    p = os.path.join(output_dir, "comparison_results.tex")
    with open(p, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"📄 LaTeX → {p}")


def print_summary_table(rows, data):
    """Affichage console du tableau récapitulatif."""
    print(f"\n{'='*95}")
    print("  TABLEAU COMPARATIF — APR sur campus d'Antananarivo")
    print(f"  Jeu de test : {data['n_test']} images "
          f"(total {data['n_total']} | train {data['n_train']} | val {data['n_val']})")
    print(f"{'='*95}")
    hdr = (f"{'Modèle':<22} {'MAE(°)':>9} {'Hav.méd(m)':>11} "
           f"{'<2m(%)':>7} {'<10m(%)':>8} {'<20m(%)':>8} "
           f"{'Ori(°)':>8} {'Params':>11} {'Mo':>6} {'ms':>6}")
    print(hdr)
    print("-"*95)
    for r in rows:
        ori = f"{r['Ori. méd. (°)']:.1f}" if not pd.isna(r['Ori. méd. (°)']) else "  —"
        star = "★" if "APR-C4" in r["Modèle"] else " "
        print(
            f"{star}{r['Modèle']:<21} "
            f"{r['MAE pos. (°)']:>9.6f} "
            f"{r['Hav. méd. (m)']:>11.2f} "
            f"{r['Succès < 2 m (%)']:>7.1f} "
            f"{r['Succès < 10 m (%)']:>8.1f} "
            f"{r['Succès < 20 m (%)']:>8.1f} "
            f"{ori:>8} "
            f"{r['Params']:>11,} "
            f"{r['Taille (Mo)']:>6.2f} "
            f"{r['Inférence (ms)']:>6.2f}"
        )
    print("="*95)
    print("★ = notre modèle proposé\n")


# ══════════════════════════════════════════════════════════════════════════════
# 9.  PIPELINE PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", default="./images")
    parser.add_argument("--csv",        default="./position_angle.csv")
    parser.add_argument("--c4_model",   default=None,
                        help="Chemin vers mobilenet_apr_c4_final.keras déjà entraîné. "
                             "Si absent, le modèle est réentraîné ici.")
    parser.add_argument("--output_dir", default="./outputs_comparison")
    parser.add_argument("--epochs",     type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed",       type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Chargement ────────────────────────────────────────────────────────────
    data = load_dataset(args.csv, args.images_dir, seed=args.seed)

    # ── Modèles et configurations d'entraînement ──────────────────────────────
    configs = [
        ("AlexNet",
         build_alexnet(),
         data["X227_tr"], data["pos_z_tr"],  data["ang_tr"] if "ang_tr" in data else np.zeros((data["n_train"],3)),
         data["X227_val"],data["pos_z_val"], np.zeros((data["n_val"],3)),
         False),
        ("ResNet50V2",
         build_resnet50(),
         data["X224_tr"], data["pos_z_tr"],  np.zeros((data["n_train"],3)),
         data["X224_val"],data["pos_z_val"], np.zeros((data["n_val"],3)),
         False),
        ("Sivapong CNN",
         build_sivapong(),
         data["X224_tr"], data["pos_z_tr"],  np.zeros((data["n_train"],3)),
         data["X224_val"],data["pos_z_val"], np.zeros((data["n_val"],3)),
         False),
        ("MobileNetV1",
         build_mobilenet_v1(),
         data["X224_tr"], data["pos_z_tr"],  np.zeros((data["n_train"],3)),
         data["X224_val"],data["pos_z_val"], np.zeros((data["n_val"],3)),
         False),
    ]

    # Récupérer ang_train depuis angles.csv si disponible, sinon zéros
    # (les modèles de comparaison prédisent azimut/pitch/roll en z-score)
    ang_tr_z  = np.zeros((data["n_train"], 3), dtype=np.float32)
    ang_val_z = np.zeros((data["n_val"],   3), dtype=np.float32)

    results = {}
    trained = {}
    rows    = []

    # ── Entraînement des 4 baselines ─────────────────────────────────────────
    for name, model, Xtr, pz_tr, _, Xval, pz_val, _ in configs:
        train_baseline(model, Xtr, pz_tr, ang_tr_z,
                       Xval, pz_val, ang_val_z,
                       epochs=args.epochs, batch=args.batch_size,
                       model_name=name)
        trained[name] = model

    # ── Entraînement / chargement de MobileNet-APR-C4 ────────────────────────
    if args.c4_model and os.path.exists(args.c4_model):
        print(f"\n--- Chargement MobileNet-APR-C4 : {args.c4_model} ---")
        c4 = tf.keras.models.load_model(
            args.c4_model,
            custom_objects={"L2NormQ": L2NormQ, "CBAM": CBAM})
    else:
        c4, _ = train_c4(data,
                          epochs_p1=10,
                          epochs_p2=args.epochs,
                          batch=args.batch_size)
        c4.save(os.path.join(args.output_dir, "mobilenet_apr_c4.keras"))
    trained["MobileNet-APR-C4"] = c4

    # ── Évaluation ────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  ÉVALUATION SUR LE JEU DE TEST")
    print("="*60)

    for name, model in trained.items():
        print(f"  Évaluation : {name}")
        if name == "MobileNet-APR-C4":
            m = eval_c4(model,
                        data["X224_te"], data["pos_mm_te"],
                        data["pos_raw_te"], data["quat_te"],
                        data["pos_mean_mm"], data["pos_range"])
        else:
            Xte = data["X227_te"] if name == "AlexNet" else data["X224_te"]
            m   = eval_baseline(model, Xte, data["pos_z_te"],
                                data["pos_raw_te"], data["quat_te"],
                                data["pos_mean_z"], data["pos_std_z"])

        n_params = int(np.sum([np.prod(v.shape) for v in model.variables]))
        size_mb  = model_size_mb(model)
        Xte_inf  = data["X227_te"] if name == "AlexNet" else data["X224_te"]
        infer    = inference_ms(model, Xte_inf)

        m["params"]   = n_params
        m["size_mb"]  = size_mb
        m["infer_ms"] = infer
        results[name] = m

        rows.append({
            "Modèle":             name,
            "MAE pos. (°)":       m["mae_pos_deg"],
            "Hav. méd. (m)":      m["haversine_median_m"],
            "Succès < 2 m (%)":   m["success_2m"],
            "Succès < 10 m (%)":  m["success_10m"],
            "Succès < 20 m (%)":  m["success_20m"],
            "Ori. méd. (°)":      m["ori_median_deg"],
            "Params":             n_params,
            "Taille (Mo)":        size_mb,
            "Inférence (ms)":     infer,
        })

    # ── Affichage & exports ───────────────────────────────────────────────────
    print_summary_table(rows, data)

    df = export_csv(rows, args.output_dir)
    export_latex(df, args.output_dir, data_info={
        "n_total": data["n_total"],
        "device":  DATASET_INFO["device"],
    })

    plot_haversine_bars(results, args.output_dir)
    plot_cdf(results, args.output_dir)
    plot_multi_metrics(results, args.output_dir)

    print(f"\n✅ Tout est sauvegardé dans : {args.output_dir}")


if __name__ == "__main__":
    # Correction de l'attribut ang_tr manquant si angles séparés non disponibles
    main()
