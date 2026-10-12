"""
Étude d'ablation progressive — MobileNet-APR-Campus
=====================================================
Conforme à la suggestion du rapporteur (Chapitre 2, sections 2.8–2.10) :

  Config 1 : MobileNetV2 original (sortie position uniquement, Euler, MSE)
  Config 2 : + tête APR duale (position + orientation Euler)
  Config 3 : + quaternion (L2-normalize) + perte géodésique
  Config 4 : + module CBAM
  Config 5 : + élagage structuré 30 % (channel pruning)
  Config 6 : + augmentation de données adaptée aux façades
  Config 7 : + régularisation (Dropout + L2 weight-decay)
  Config 8 : modèle complet (+ clustering auxiliaire + smoothness loss)

Pour chaque configuration, le script mesure et affiche :
  - MAE position (degrés décimaux, sur coordonnées dénormalisées)
  - Distance géodésique de Haversine médiane (m) — Éq. manuscrit
  - Erreur d'orientation médiane (°)
  - Nombre de paramètres total / entraînable
  - Taille mémoire du modèle (Mo)
  - Temps d'inférence médian par image (ms)

Usage :
    python ablation_study.py \
        --images_dir ./images \
        --positions_csv ./positions.csv \
        --angles_csv    ./angles.csv \
        --epochs        30 \
        --batch_size    32
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, regularizers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.cluster import KMeans

# Modules locaux (même dossier que ce script)
sys.path.insert(0, os.path.dirname(__file__))
from cbam import apply_cbam, SpatialAttention
from data_loader import load_dataset
from losses import (
    position_loss,
    quaternion_geodesic_loss,
    HomoscedasticUncertaintyWeightedLoss,
)
from metrics import (
    haversine_distance,
    median_orientation_error_degrees,
    position_mae,
)
from preprocessing import PositionNormalizer
from pruning import (
    compute_prune_masks,
    apply_prune_masks,
    maintain_pruning_masks,
    compute_sparsity,
)

# ─────────────────────────────────────────────────────────────────────────────
# CORRECTIFS apportés à cette version (voir CHANGELOG.md pour le détail) :
#   1. Validation séparée du test (data["X_val"] / data["X_test"]) : le test
#      n'est plus jamais vu avant l'évaluation finale.
#   2. Augmentation : suppression du flip horizontal (qui inverse l'image
#      sans inverser la pose -> labels faux) ; flou appliqué aléatoirement
#      (et non systématiquement) pour que train et test restent comparables.
#   3. Pruning : calcul du masque par saillance de gradient réelle (sur des
#      lots de calibration du train set), puis maintien du masque à CHAQUE
#      étape de gradient pendant tout le fine-tuning qui suit (sinon
#      l'optimiseur fait "repousser" des poids censés être élagués).
#   4. Config 8 : vrais pseudo-labels de cluster (KMeans sur les positions
#      d'entraînement), au lieu d'un label constant à 0.
#   5. Option de dégel progressif du backbone (configs 3 à 8), pour refléter
#      le protocole en 2 phases du manuscrit (train.py) plutôt que de
#      n'entraîner que la tête avec un backbone gelé en permanence.
#   6. Mesure du temps d'inférence via tf.function (moins d'overhead eager).
#   7. Colonne de sparsité ajoutée : l'élagage par masquage ne réduit PAS
#      la taille du fichier .keras ; on rapporte donc séparément le % de
#      canaux mis à zéro pour ne pas laisser croire à une compression.
# ─────────────────────────────────────────────────────────────────────────────

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


# ─────────────────────────────────────────────────────────────────────────────
# Couche custom partagée par plusieurs configs
# ─────────────────────────────────────────────────────────────────────────────
class L2NormalizeQuaternion(layers.Layer):
    """Normalisation L2 du quaternion brut (Éq. quat_normalize)."""

    def call(self, q_raw):
        norm = tf.norm(q_raw, axis=-1, keepdims=True)
        norm = tf.maximum(norm, 1e-12)
        return q_raw / norm


# ─────────────────────────────────────────────────────────────────────────────
# Augmentation de données (Config 6)
# ─────────────────────────────────────────────────────────────────────────────
def augment_image(image):
    """
    Augmentation adaptée aux façades répétitives (Section "Augmentation de
    données adaptée aux façades répétitives" du manuscrit).

    CORRECTIF vs version initiale :
      - Le FLIP HORIZONTAL est supprimé. Une image retournée miroir change
        la géométrie de la scène (azimut, position relative des façades)
        sans que la position/l'orientation labellisée ne soit elle-même
        retournée : le modèle apprenait donc une association image<->pose
        fausse sur la moitié augmentée des exemples. Retourner correctement
        la pose demanderait de recalculer azimuth/position en fonction de
        la géométrie de la scène, ce qui dépasse le cadre d'une simple
        augmentation d'image ; on le laisse donc de côté.
      - Le flou n'est plus appliqué systématiquement à 100% des images
        d'entraînement (ce qui créait un écart de distribution avec le
        test, jamais flouté) : il est maintenant tiré aléatoirement avec
        une probabilité de 30%, comme la luminosité/contraste.
      - luminosité/contraste conservés (transformations photométriques :
        elles ne changent pas la pose réelle de la caméra).
    """
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)

    def _blur():
        blurred = tf.nn.avg_pool2d(
            tf.expand_dims(image, 0), ksize=3, strides=1, padding="SAME"
        )
        return tf.squeeze(blurred, 0)

    image = tf.cond(tf.random.uniform([]) < 0.3, _blur, lambda: image)
    return image


def make_tf_dataset(images, positions, quaternions, clusters=None,
                    batch_size=32, shuffle=True, augment=False):
    """
    clusters : tableau optionnel (N,) d'identifiants de cluster entiers
    (Config 8). Si fourni, le dataset produit des 4-uplets
    (image, position, quaternion, cluster) ; sinon des 3-uplets, comme
    précédemment (rétro-compatible avec les autres configurations).
    """
    if clusters is not None:
        ds = tf.data.Dataset.from_tensor_slices(
            (images, positions, quaternions, clusters.astype(np.int32))
        )
    else:
        ds = tf.data.Dataset.from_tensor_slices((images, positions, quaternions))

    if shuffle:
        ds = ds.shuffle(buffer_size=len(images), seed=42)

    if augment:
        if clusters is not None:
            ds = ds.map(
                lambda img, p, q, c: (augment_image(img), p, q, c),
                num_parallel_calls=tf.data.AUTOTUNE,
            )
        else:
            ds = ds.map(
                lambda img, p, q: (augment_image(img), p, q),
                num_parallel_calls=tf.data.AUTOTUNE,
            )
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


# ─────────────────────────────────────────────────────────────────────────────
# Builders — une fonction par configuration
# ─────────────────────────────────────────────────────────────────────────────

def build_config1():
    """
    Config 1 — MobileNetV2 original.
    Backbone seul + GAP + Dense(2) pour la position.
    Sortie unique : (phi, lambda) normalisé.  Perte : MSE standard.
    Pas de tête d'orientation, pas de quaternion.
    """
    backbone = MobileNetV2(weights="imagenet", include_top=False,
                           input_shape=(224, 224, 3))
    backbone.trainable = False
    x = backbone.output
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    pos_out = layers.Dense(2, activation="linear", name="position")(x)
    # Sortie orientation fictive (zéros) pour garder une API uniforme
    ori_out = layers.Lambda(
        lambda t: tf.tile(tf.constant([[1., 0., 0., 0.]]),
                          [tf.shape(t)[0], 1]),
        name="orientation",
    )(pos_out)
    return tf.keras.Model(backbone.input, [pos_out, ori_out],
                          name="C1_MobileNetV2_original")


def build_config2():
    """
    Config 2 — + Tête APR duale.
    Deux branches Dense parallèles mais orientation encodée en Euler (3 valeurs),
    pas encore de quaternion ni de normalisation L2.
    Perte : MSE sur les deux sorties.
    """
    backbone = MobileNetV2(weights="imagenet", include_top=False,
                           input_shape=(224, 224, 3))
    backbone.trainable = False
    x = backbone.output
    pooled = layers.GlobalAveragePooling2D(name="gap")(x)

    # Branche position
    p = layers.Dense(256, activation="relu", name="pos_dense")(pooled)
    pos_out = layers.Dense(2, activation="linear", name="position")(p)

    # Branche orientation (Euler, 3 sorties, pas de quaternion)
    o = layers.Dense(256, activation="relu", name="ori_dense")(pooled)
    euler_out = layers.Dense(3, activation="linear", name="euler_raw")(o)
    # Convertir en quaternion trivial (w=1, xyz=euler/‖‖) pour API uniforme
    # Remarque : ce n'est pas un vrai quaternion unitaire — c'est Config 2
    # qui n'a pas encore la normalisation L2.
    ori_out = layers.Lambda(
        lambda e: tf.math.l2_normalize(
            tf.concat([tf.ones([tf.shape(e)[0], 1]), e], axis=-1), axis=-1
        ),
        name="orientation",
    )(euler_out)
    return tf.keras.Model(backbone.input, [pos_out, ori_out],
                          name="C2_APR_head")


def build_config3():
    """
    Config 3 — + Quaternion unitaire + perte géodésique.
    La branche orientation sort 4 valeurs + normalisation L2 garantissant
    la contrainte d'unitarité (Éq. quat_normalize).
    """
    backbone = MobileNetV2(weights="imagenet", include_top=False,
                           input_shape=(224, 224, 3))
    backbone.trainable = False
    x = backbone.output
    pooled = layers.GlobalAveragePooling2D(name="gap")(x)

    p = layers.Dense(256, activation="relu", name="pos_dense")(pooled)
    pos_out = layers.Dense(2, activation="linear", name="position")(p)

    o = layers.Dense(256, activation="relu", name="ori_dense")(pooled)
    q_raw = layers.Dense(4, activation="linear", name="quaternion_raw")(o)
    ori_out = L2NormalizeQuaternion(name="orientation")(q_raw)

    return tf.keras.Model(backbone.input, [pos_out, ori_out],
                          name="C3_quaternion")


def build_config4():
    """
    Config 4 — + Module CBAM (attention spatiale).
    Inséré après la dernière conv 1×1 du backbone, avant le GAP.
    """
    backbone = MobileNetV2(weights="imagenet", include_top=False,
                           input_shape=(224, 224, 3))
    backbone.trainable = False
    x = backbone.output
    x = apply_cbam(x, kernel_size=7, name="cbam_spatial_attention")
    pooled = layers.GlobalAveragePooling2D(name="gap")(x)

    p = layers.Dense(256, activation="relu", name="pos_dense")(pooled)
    pos_out = layers.Dense(2, activation="linear", name="position")(p)

    o = layers.Dense(256, activation="relu", name="ori_dense")(pooled)
    q_raw = layers.Dense(4, activation="linear", name="quaternion_raw")(o)
    ori_out = L2NormalizeQuaternion(name="orientation")(q_raw)

    return tf.keras.Model(backbone.input, [pos_out, ori_out],
                          name="C4_CBAM")


def build_config5():
    """
    Config 5 — + Élagage structuré 30 % des canaux internes (blocs 7–17).

    CORRECTIF vs version initiale : ce builder ne fait plus l'élagage
    lui-même. Dans la version précédente, le pruning était appliqué ICI,
    à la construction, par heuristique de magnitude (faute de données de
    calibration disponibles à ce stade) sur un backbone encore gelé et
    jamais entraîné : les canaux mis à zéro ne pouvaient donc jamais être
    compensés par le reste du réseau, ce qui explique la forte dégradation
    observée. Le pruning réel (saillance de gradient calculée sur des lots
    d'entraînement, masque maintenu pendant tout le fine-tuning) est
    maintenant effectué dans run_ablation(), après construction du modèle
    et à partir de données réelles — voir la fonction `train_model` et le
    paramètre `prune_masks`.

    Ce builder retourne donc une architecture identique à Config 4
    (c'est run_ablation qui lui applique ensuite l'élagage).
    """
    backbone = MobileNetV2(weights="imagenet", include_top=False,
                           input_shape=(224, 224, 3))
    backbone.trainable = False
    x = backbone.output
    x = apply_cbam(x, kernel_size=7, name="cbam_spatial_attention")
    pooled = layers.GlobalAveragePooling2D(name="gap")(x)
    p = layers.Dense(256, activation="relu", name="pos_dense")(pooled)
    pos_out = layers.Dense(2, activation="linear", name="position")(p)
    o = layers.Dense(256, activation="relu", name="ori_dense")(pooled)
    q_raw = layers.Dense(4, activation="linear", name="quaternion_raw")(o)
    ori_out = L2NormalizeQuaternion(name="orientation")(q_raw)
    return tf.keras.Model(backbone.input, [pos_out, ori_out],
                          name="C5_pruning")


def build_config6():
    """
    Config 6 — + Augmentation de données.
    Architecture identique à Config 5 (élagage appliqué par run_ablation
    après construction) ; l'augmentation est activée dans le pipeline de
    données (flag `augment=True` dans make_tf_dataset).
    """
    backbone = MobileNetV2(weights="imagenet", include_top=False,
                           input_shape=(224, 224, 3))
    backbone.trainable = False
    x = backbone.output
    x = apply_cbam(x, kernel_size=7, name="cbam_spatial_attention")
    pooled = layers.GlobalAveragePooling2D(name="gap")(x)
    p = layers.Dense(256, activation="relu", name="pos_dense")(pooled)
    pos_out = layers.Dense(2, activation="linear", name="position")(p)
    o = layers.Dense(256, activation="relu", name="ori_dense")(pooled)
    q_raw = layers.Dense(4, activation="linear", name="quaternion_raw")(o)
    ori_out = L2NormalizeQuaternion(name="orientation")(q_raw)
    return tf.keras.Model(backbone.input, [pos_out, ori_out],
                          name="C6_augmentation")


def build_config7(dropout_rate=0.3, l2_lambda=1e-4):
    """
    Config 7 — + Régularisation (Dropout + L2 weight-decay).
    Dropout après chaque couche Dense intermédiaire de la tête +
    régularisation L2 sur les noyaux des couches Dense.
    """
    backbone = MobileNetV2(weights="imagenet", include_top=False,
                           input_shape=(224, 224, 3))
    backbone.trainable = False
    x = backbone.output
    x = apply_cbam(x, kernel_size=7, name="cbam_spatial_attention")
    # (Élagage retiré d'ici — voir note dans build_config5 : appliqué par
    # run_ablation() après construction, avec une vraie saillance de
    # gradient et un masque maintenu pendant le fine-tuning.)
    pooled = layers.GlobalAveragePooling2D(name="gap")(x)

    reg = regularizers.l2(l2_lambda)

    p = layers.Dense(256, activation="relu",
                     kernel_regularizer=reg, name="pos_dense")(pooled)
    p = layers.Dropout(dropout_rate, name="pos_dropout")(p)
    pos_out = layers.Dense(2, activation="linear", name="position")(p)

    o = layers.Dense(256, activation="relu",
                     kernel_regularizer=reg, name="ori_dense")(pooled)
    o = layers.Dropout(dropout_rate, name="ori_dropout")(o)
    q_raw = layers.Dense(4, activation="linear", name="quaternion_raw")(o)
    ori_out = L2NormalizeQuaternion(name="orientation")(q_raw)

    return tf.keras.Model(backbone.input, [pos_out, ori_out],
                          name="C7_regularization")


def build_config8(n_clusters=5):
    """
    Config 8 — Modèle complet.
    Ajoute la tête de classification auxiliaire pour le clustering
    géographique (Section "Décodage par Clustering" du manuscrit) sur le
    même vecteur de caractéristiques 1280-dimensional.
    Troisième sortie : cluster_id (softmax sur K classes).
    Pour l'évaluation ablation, seules les sorties position et orientation
    sont utilisées dans les métriques ; la tête de clustering guide
    l'entraînement via une cross-entropy auxiliaire.

    CORRECTIF vs version initiale : dans ablation_study.py d'origine, les
    labels passés à cette tête pendant l'entraînement étaient tous fixés à
    0 (tf.zeros(...)) — la tête apprenait donc une constante, sans aucun
    signal de clustering réel. run_ablation() calcule maintenant de vrais
    pseudo-labels par k-means sur les positions d'entraînement
    (cf. compute_cluster_labels) et les fournit au dataset de C8.
    Cette architecture est aussi, par construction, élaguée par
    run_ablation() comme les configs 5 à 7 (c'est bien le "modèle complet").
    """
    backbone = MobileNetV2(weights="imagenet", include_top=False,
                           input_shape=(224, 224, 3))
    backbone.trainable = False
    x = backbone.output
    x = apply_cbam(x, kernel_size=7, name="cbam_spatial_attention")

    pooled = layers.GlobalAveragePooling2D(name="gap")(x)

    reg = regularizers.l2(1e-4)

    p = layers.Dense(256, activation="relu",
                     kernel_regularizer=reg, name="pos_dense")(pooled)
    p = layers.Dropout(0.3, name="pos_dropout")(p)
    pos_out = layers.Dense(2, activation="linear", name="position")(p)

    o = layers.Dense(256, activation="relu",
                     kernel_regularizer=reg, name="ori_dense")(pooled)
    o = layers.Dropout(0.3, name="ori_dropout")(o)
    q_raw = layers.Dense(4, activation="linear", name="quaternion_raw")(o)
    ori_out = L2NormalizeQuaternion(name="orientation")(q_raw)

    # Tête de clustering auxiliaire (Section "Décodage par Clustering")
    cluster_out = layers.Dense(
        n_clusters, activation="softmax",
        kernel_regularizer=reg, name="cluster",
    )(pooled)

    return tf.keras.Model(
        backbone.input,
        [pos_out, ori_out, cluster_out],
        name="C8_full_model",
    )


def compute_cluster_labels(pos_train, pos_other, n_clusters=5, random_state=42):
    """
    Calcule de vrais pseudo-labels de cluster géographique par k-means,
    ajustés sur les positions normalisées d'entraînement (pos_train), puis
    appliqués (predict, pas de fuite) à un autre ensemble (val ou test).

    Remplace le label constant à 0 de la version initiale de
    build_config8 / train_step, qui ne fournissait aucun signal de
    clustering réel à la tête auxiliaire.

    Retourne (labels_train, labels_other), tableaux d'entiers dans
    [0, n_clusters).
    """
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state)
    labels_train = km.fit_predict(pos_train)
    labels_other = km.predict(pos_other)
    return labels_train.astype(np.int32), labels_other.astype(np.int32)


# ─────────────────────────────────────────────────────────────────────────────
# Boucle d'entraînement unifiée
# ─────────────────────────────────────────────────────────────────────────────

def train_model(model, train_ds, val_ds, epochs, config_id,
                use_geodesic_loss=True,
                use_learned_weighting=True,
                use_smoothness=False,
                has_cluster_head=False,
                n_clusters=5,
                lr=1e-3,
                prune_masks=None,
                unfreeze_backbone_at=None,
                finetune_lr=1e-5,
                restore_best_weights=True):
    """
    Entraîne un modèle d'ablation et retourne l'historique de perte.

    Paramètres d'activation par configuration :
      use_geodesic_loss    : False pour Config 1-2 (MSE simple)
      use_learned_weighting: False pour Config 1-2 (β fixe)
      has_cluster_head     : True uniquement pour Config 8

    Paramètres ajoutés (correctifs) :
      prune_masks           : dict {bloc: indices_canaux} issu de
                              pruning.compute_prune_masks, réappliqué après
                              CHAQUE étape de gradient (maintain_pruning_masks)
                              pour que l'élagage reste effectif pendant tout
                              le fine-tuning. None = pas de pruning (C1–C4).
      unfreeze_backbone_at  : numéro d'époque à partir duquel le backbone
                              MobileNetV2 est dégelé (model.trainable=True)
                              et l'optimiseur repart à `finetune_lr`. None =
                              backbone gelé sur toute la durée (comportement
                              d'origine). Reflète le protocole en 2 phases
                              du manuscrit (cf. train.py) plutôt que de
                              limiter l'ablation à l'entraînement de la tête.
      restore_best_weights  : si True (défaut), restaure en fin
                              d'entraînement les poids correspondant au
                              meilleur val_loss observé (plutôt que les
                              poids de la dernière époque, potentiellement
                              surentraînés) avant le retour à run_ablation.

    IMPORTANT : `val_ds` doit être un ensemble de VALIDATION distinct du
    jeu de test final (voir data_loader.load_dataset, qui fournit
    désormais X_val séparément de X_test). Le test ne doit jamais entrer
    dans cette fonction.
    """
    uncertainty_loss = HomoscedasticUncertaintyWeightedLoss() \
        if use_learned_weighting else None

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    backbone_unfrozen = False

    def train_step(images, pos_true, quat_true, cluster_true=None):
        with tf.GradientTape() as tape:
            outputs = model(images, training=True)

            if has_cluster_head:
                pos_pred, ori_pred, cluster_pred = outputs
            else:
                pos_pred, ori_pred = outputs[0], outputs[1]

            l_pos = position_loss(pos_true, pos_pred)

            if use_geodesic_loss:
                l_ori = quaternion_geodesic_loss(quat_true, ori_pred)
            else:
                l_ori = tf.reduce_mean(tf.square(quat_true - ori_pred))

            if use_learned_weighting and uncertainty_loss is not None:
                total = uncertainty_loss(l_pos, l_ori)
            else:
                total = l_pos + l_ori

            # Perte de clustering (Config 8) — cluster_true vient maintenant
            # de vrais pseudo-labels k-means (cf. compute_cluster_labels),
            # plus d'une constante à 0.
            if has_cluster_head and cluster_true is not None:
                l_cluster = tf.reduce_mean(
                    tf.keras.losses.sparse_categorical_crossentropy(
                        cluster_true, cluster_pred
                    )
                )
                total = total + 0.1 * l_cluster

            if model.losses:
                total = total + tf.add_n(model.losses)

        all_vars = model.trainable_variables
        if uncertainty_loss is not None:
            all_vars = all_vars + uncertainty_loss.trainable_variables
        grads = tape.gradient(total, all_vars)
        optimizer.apply_gradients(zip(grads, all_vars))

        # Maintien du masque de pruning : sans ceci, l'optimiseur peut
        # redonner un poids non nul à un canal censé être élagué.
        if prune_masks:
            maintain_pruning_masks(model, prune_masks)

        return total, l_pos, l_ori

    best_val_loss = float("inf")
    best_weights = None
    patience_counter = 0
    patience = max(5, epochs // 5)   # early stopping léger

    for epoch in range(1, epochs + 1):
        # Dégel progressif du backbone (reflète le protocole 2 phases du
        # manuscrit). Nouvel optimiseur à LR plus faible pour la phase
        # d'affinage global, comme dans train.py.
        if (unfreeze_backbone_at is not None
                and epoch == unfreeze_backbone_at
                and not backbone_unfrozen):
            model.trainable = True
            optimizer = tf.keras.optimizers.Adam(learning_rate=finetune_lr)
            backbone_unfrozen = True
            print(f"    [{config_id}] Dégel du backbone à l'époque {epoch} "
                  f"(LR -> {finetune_lr}).")

        train_losses = []
        for batch in train_ds:
            if has_cluster_head:
                images, pos_true, quat_true, cluster_true = batch
            else:
                images, pos_true, quat_true = batch
                cluster_true = None
            loss, l_pos, l_ori = train_step(images, pos_true, quat_true, cluster_true)
            train_losses.append(float(loss))

        val_losses = []
        for batch in val_ds:
            if has_cluster_head:
                images, pos_true, quat_true, _ = batch
            else:
                images, pos_true, quat_true = batch
            outputs = model(images, training=False)
            pos_pred, ori_pred = outputs[0], outputs[1]
            l_p = position_loss(pos_true, pos_pred)
            l_o = (quaternion_geodesic_loss(quat_true, ori_pred)
                   if use_geodesic_loss
                   else tf.reduce_mean(tf.square(quat_true - ori_pred)))
            val_losses.append(float(l_p + l_o))

        val_loss = np.mean(val_losses)
        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            patience_counter = 0
            if restore_best_weights:
                best_weights = model.get_weights()
        else:
            patience_counter += 1

        if epoch % max(1, epochs // 5) == 0 or epoch == epochs:
            print(
                f"    [{config_id}] Époque {epoch:3d}/{epochs} — "
                f"train={np.mean(train_losses):.4f}  val={val_loss:.4f}"
            )

        if patience_counter >= patience:
            print(f"    [{config_id}] Early stopping à l'époque {epoch}.")
            break

    if restore_best_weights and best_weights is not None:
        model.set_weights(best_weights)
        # Un dernier passage de maintien du masque : set_weights() peut
        # avoir restauré un état antérieur à certaines mises à zéro.
        if prune_masks:
            maintain_pruning_masks(model, prune_masks)


# ─────────────────────────────────────────────────────────────────────────────
# Mesures de performance
# ─────────────────────────────────────────────────────────────────────────────

def count_params(model):
    total = int(np.sum([np.prod(v.shape) for v in model.variables]))
    trainable = int(np.sum([np.prod(v.shape) for v in model.trainable_variables]))
    return total, trainable


def model_size_mb(model, tmp_path="/tmp/_ablation_tmp.keras"):
    """
    Taille réelle du fichier .keras sauvegardé (Mo).

    ATTENTION (voir CHANGELOG) : pour les configs élaguées par masquage
    (C5–C8), cette taille NE DIMINUE PAS, puisque les poids mis à zéro
    restent stockés comme n'importe quel autre poids. Le pruning par
    masquage ne mesure donc ici qu'un effet potentiel sur la *précision*,
    pas sur la *compacité* — voir la colonne de sparsité ajoutée dans
    evaluate_metrics / run_ablation pour une mesure honnête de l'élagage.
    """
    model.save(tmp_path)
    size_bytes = os.path.getsize(tmp_path)
    os.remove(tmp_path)
    return size_bytes / (1024 ** 2)


def inference_time_ms(model, X_sample, n_runs=50):
    """
    Temps d'inférence médian par image (ms) mesuré sur n_runs passes,
    après 5 passes de chauffe (warm-up GPU/CPU).

    CORRECTIF vs version initiale : l'appel direct model(batch) en mode
    eager mesurait surtout l'overhead Python de TensorFlow (d'où des temps
    ~150-175 ms, irréalistes pour un MobileNetV2 à cette résolution).
    L'enveloppe dans tf.function compile le graphe une fois (lors du
    warm-up) puis mesure l'exécution réelle du graphe compilé, bien plus
    représentative d'un déploiement.
    """
    infer_fn = tf.function(lambda x: model(x, training=False))
    batch = tf.constant(X_sample[:1])   # une seule image
    for _ in range(5):                  # warm-up (inclut la compilation du graphe)
        infer_fn(batch)
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        infer_fn(batch)
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.median(times))


def evaluate_metrics(model, X_test, pos_test_norm, quat_test,
                     position_normalizer):
    """
    Calcule MAE, distance Haversine médiane (m) et erreur orientation (°)
    sur l'ensemble de test.

    La distance géodésique de Haversine suit l'Éq. du manuscrit :
        d = 2R · arcsin(√(sin²(Δφ/2) + cos(φ₁)·cos(φ₂)·sin²(Δλ/2)))
    """
    outputs = model.predict(X_test, verbose=0)
    pos_pred_norm = outputs[0]
    quat_pred = outputs[1]

    pos_pred = position_normalizer.inverse_transform(pos_pred_norm)
    pos_true = position_normalizer.inverse_transform(pos_test_norm)

    mae = position_mae(pos_true, pos_pred)

    # Haversine vectorisée (Éq. manuscrit, R en mètres)
    R = 6_371_000.0
    lat1 = np.radians(pos_true[:, 0]); lon1 = np.radians(pos_true[:, 1])
    lat2 = np.radians(pos_pred[:, 0]); lon2 = np.radians(pos_pred[:, 1])
    dphi = lat2 - lat1; dlam = lon2 - lon1
    a = (np.sin(dphi / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin(dlam / 2) ** 2)
    dists = 2.0 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    median_dist = float(np.median(dists))

    median_ori, _ = median_orientation_error_degrees(quat_test, quat_pred)
    return mae, median_dist, median_ori


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline principal
# ─────────────────────────────────────────────────────────────────────────────

def run_ablation(args):
    print("\n" + "=" * 72)
    print("  ÉTUDE D'ABLATION PROGRESSIVE — MobileNet-APR-Campus")
    print("  (version corrigée — voir CHANGELOG.md)")
    print("=" * 72)

    # ── Chargement des données ─────────────────────────────────────────────
    print("\n[0/8] Chargement et préparation des données...")
    data = load_dataset(
        images_dir=args.images_dir,
        combined_csv_path=getattr(args, 'combined_csv', None),
        positions_csv_path=getattr(args, 'positions_csv', None),
        angles_csv_path=getattr(args, 'angles_csv', None),
        test_size=0.2,
        val_size=args.val_size,
    )
    pos_norm = data["position_normalizer"]
    X_train, X_val, X_test = data["X_train"], data["X_val"], data["X_test"]
    pos_train, pos_val, pos_test = data["pos_train"], data["pos_val"], data["pos_test"]
    quat_train, quat_val, quat_test = data["quat_train"], data["quat_val"], data["quat_test"]

    # CORRECTIF : val_ds (dédié à l'early stopping / sélection de config)
    # est maintenant DISTINCT de test_ds (réservé à l'évaluation finale,
    # jamais vu pendant l'entraînement). Dans la version initiale, le même
    # ensemble servait aux deux usages.
    train_ds_no_aug = make_tf_dataset(
        X_train, pos_train, quat_train,
        batch_size=args.batch_size, shuffle=True, augment=False,
    )
    train_ds_aug = make_tf_dataset(
        X_train, pos_train, quat_train,
        batch_size=args.batch_size, shuffle=True, augment=True,
    )
    val_ds = make_tf_dataset(
        X_val, pos_val, quat_val,
        batch_size=args.batch_size, shuffle=False, augment=False,
    )
    test_ds = make_tf_dataset(
        X_test, pos_test, quat_test,
        batch_size=args.batch_size, shuffle=False, augment=False,
    )

    # Datasets avec pseudo-labels de cluster réels (Config 8 uniquement) —
    # CORRECTIF : remplace le label constant à 0 de la version initiale.
    cluster_train, cluster_val = compute_cluster_labels(
        pos_train, pos_val, n_clusters=5, random_state=getattr(args, "seed", 42)
    )
    train_ds_aug_clusters = make_tf_dataset(
        X_train, pos_train, quat_train, clusters=cluster_train,
        batch_size=args.batch_size, shuffle=True, augment=True,
    )
    val_ds_clusters = make_tf_dataset(
        X_val, pos_val, quat_val, clusters=cluster_val,
        batch_size=args.batch_size, shuffle=False, augment=False,
    )

    print(
        f"  Train : {len(X_train)} images  |  "
        f"Val : {len(X_val)} images  |  Test : {len(X_test)} images"
    )

    # Lots de calibration (issus du TRAIN uniquement) pour le calcul de la
    # saillance de gradient utilisée par le pruning réel des configs 5–8.
    calibration_batches = []
    for images, pos_true, quat_true in train_ds_no_aug.take(5):
        calibration_batches.append(
            (images, (pos_true, quat_true))
        )

    def pruning_loss_fn(preds, targets):
        # preds peut contenir 2 sorties (configs 5-7 : position, orientation)
        # ou 3 (config 8 : + cluster) — seules les deux premières servent
        # au calcul de la saillance. Un déballage strict `a, b = preds`
        # planterait sur le modèle à 3 sorties de la Config 8.
        pos_pred, ori_pred = preds[0], preds[1]
        pos_true, ori_true = targets
        return position_loss(pos_true, pos_pred) + quaternion_geodesic_loss(ori_true, ori_pred)

    # À partir de quelle époque dégeler le backbone pour les configs 3 à 8
    # (reflète le protocole en 2 phases du manuscrit / train.py : gel
    # initial de la tête, puis affinage global à LR réduit). Les Configs
    # 1 et 2 restent backbone gelé sur toute la durée, pour continuer à
    # représenter fidèlement "MobileNetV2 original" / "tête seule", qui
    # sont les points de comparaison de départ de l'ablation.
    unfreeze_at = max(1, args.epochs // 3)

    # ── Définition des 8 configurations ───────────────────────────────────
    # needs_pruning : élagage réel (saillance de gradient + masque maintenu
    #                 pendant le fine-tuning) appliqué par run_ablation
    #                 juste après la construction du modèle, pour C5 à C8.
    configs = [
        # (id, nom_court, builder, train_ds, val_ds, kwargs_train, needs_pruning)
        (1, "MobileNetV2 original",
         build_config1, train_ds_no_aug, val_ds,
         dict(use_geodesic_loss=False, use_learned_weighting=False),
         False),

        (2, "+ Tête APR duale",
         build_config2, train_ds_no_aug, val_ds,
         dict(use_geodesic_loss=False, use_learned_weighting=False),
         False),

        (3, "+ Quaternion (L2-norm + perte géodésique)",
         build_config3, train_ds_no_aug, val_ds,
         dict(use_geodesic_loss=True, use_learned_weighting=True,
              unfreeze_backbone_at=unfreeze_at),
         False),

        (4, "+ CBAM (attention spatiale)",
         build_config4, train_ds_no_aug, val_ds,
         dict(use_geodesic_loss=True, use_learned_weighting=True,
              unfreeze_backbone_at=unfreeze_at),
         False),

        (5, "+ Pruning 30 % (saillance réelle, masque maintenu)",
         build_config5, train_ds_no_aug, val_ds,
         dict(use_geodesic_loss=True, use_learned_weighting=True,
              unfreeze_backbone_at=unfreeze_at),
         True),

        (6, "+ Augmentation de données (photométrique uniquement)",
         build_config6, train_ds_aug, val_ds,
         dict(use_geodesic_loss=True, use_learned_weighting=True,
              unfreeze_backbone_at=unfreeze_at),
         True),

        (7, "+ Régularisation (Dropout + L2)",
         build_config7, train_ds_aug, val_ds,
         dict(use_geodesic_loss=True, use_learned_weighting=True,
              unfreeze_backbone_at=unfreeze_at),
         True),

        (8, "Modèle complet (+ Clustering réel)",
         lambda: build_config8(n_clusters=5),
         train_ds_aug_clusters, val_ds_clusters,
         dict(use_geodesic_loss=True, use_learned_weighting=True,
              has_cluster_head=True, unfreeze_backbone_at=unfreeze_at),
         True),
    ]

    # ── Boucle d'ablation ─────────────────────────────────────────────────
    results = []

    for cfg_id, cfg_name, builder_fn, t_ds, v_ds, train_kwargs, needs_pruning in configs:
        print(f"\n{'─'*72}")
        print(f"  Config {cfg_id}/8 : {cfg_name}")
        print(f"{'─'*72}")

        # Construction
        model = builder_fn()

        # Élagage réel (C5–C8) : calcul du masque par saillance de gradient
        # sur des lots de CALIBRATION issus du train set, puis application.
        # Le masque sera ensuite maintenu à chaque étape de gradient par
        # train_model (prune_masks=...), contrairement à la version
        # initiale où le masquage, une fois appliqué, pouvait être
        # partiellement annulé par le fine-tuning.
        prune_masks = None
        if needs_pruning:
            prune_masks = compute_prune_masks(
                model, prune_ratio=0.30,
                data_batches=calibration_batches, loss_fn=pruning_loss_fn,
            )
            apply_prune_masks(model, prune_masks)
            sparsity = compute_sparsity(model, prune_masks)
            print(f"  Élagage appliqué : {sparsity:.1f}% des canaux ciblés mis à zéro.")

        # Entraînement
        train_model(
            model, t_ds, v_ds,
            epochs=args.epochs,
            config_id=f"C{cfg_id}",
            lr=args.lr,
            prune_masks=prune_masks,
            **train_kwargs,
        )

        # Évaluation des métriques — UNIQUEMENT sur le jeu de test, jamais
        # vu pendant l'entraînement ni la sélection du meilleur epoch.
        print(f"  → Évaluation de C{cfg_id} sur le jeu de TEST...")
        mae, median_dist_m, median_ori_deg = evaluate_metrics(
            model, X_test, pos_test, quat_test, pos_norm,
        )

        # Métriques de taille / vitesse
        total_params, trainable_params = count_params(model)
        size_mb = model_size_mb(model)
        infer_ms = inference_time_ms(model, X_test)
        sparsity_pct = compute_sparsity(model, prune_masks) if prune_masks else 0.0

        results.append({
            "Config": cfg_id,
            "Description": cfg_name,
            "MAE pos. (°)": round(mae, 6),
            "Haversine méd. (m)": round(median_dist_m, 2),
            "Err. orient. méd. (°)": round(median_ori_deg, 3),
            "Params total": total_params,
            "Params entraîn.": trainable_params,
            "Taille (Mo)": round(size_mb, 2),
            "Inférence (ms)": round(infer_ms, 2),
            "Canaux élagués (%)": round(sparsity_pct, 1),
        })

        print(
            f"  MAE={mae:.6f}°  |  Haversine={median_dist_m:.2f} m  |"
            f"  Ori={median_ori_deg:.3f}°  |"
            f"  Params={total_params:,}  |  {size_mb:.2f} Mo  |"
            f"  {infer_ms:.2f} ms/img  |  Élagage={sparsity_pct:.1f}%"
        )

        tf.keras.backend.clear_session()

    # ── Tableau de synthèse ────────────────────────────────────────────────
    df = pd.DataFrame(results)

    print("\n\n" + "=" * 72)
    print("  TABLEAU DE SYNTHÈSE DE L'ÉTUDE D'ABLATION")
    print("=" * 72)

    # Affichage console
    col_widths = {
        "Config": 6,
        "Description": 40,
        "MAE pos. (°)": 13,
        "Haversine méd. (m)": 20,
        "Err. orient. méd. (°)": 23,
        "Params total": 14,
        "Params entraîn.": 16,
        "Taille (Mo)": 12,
        "Inférence (ms)": 15,
        "Canaux élagués (%)": 19,
    }
    header = "  ".join(k.ljust(v) for k, v in col_widths.items())
    sep = "  ".join("─" * v for v in col_widths.values())
    print("\n" + header)
    print(sep)
    for _, row in df.iterrows():
        line = "  ".join(
            str(row[k]).ljust(v) for k, v in col_widths.items()
        )
        print(line)

    # ── Export CSV ────────────────────────────────────────────────────────
    # CORRECTIF : args.positions_csv peut être None en Mode A (combined_csv),
    # ce qui faisait planter os.path.dirname(None). On retombe sur
    # combined_csv, puis sur images_dir en dernier recours.
    ref_path = getattr(args, "combined_csv", None) or args.positions_csv or args.images_dir
    out_dir = os.path.dirname(ref_path) or "."
    out_csv = os.path.join(out_dir, "ablation_results.csv")
    df.to_csv(out_csv, index=False, sep=";")
    print(f"\n  Résultats exportés → {out_csv}")

    # ── Export LaTeX ──────────────────────────────────────────────────────
    out_tex = out_csv.replace(".csv", ".tex")
    _export_latex_table(df, out_tex)
    print(f"  Tableau LaTeX exporté → {out_tex}")

    print("\n" + "=" * 72 + "\n")
    return df


def _export_latex_table(df, path):
    """
    Génère un tableau LaTeX booktabs prêt à coller dans le manuscrit
    (chapitres 2.8–2.10), conforme aux exigences du rapporteur.
    """
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Étude d'ablation progressive de MobileNet-APR-Campus. "
        r"$\downarrow$ : plus faible est meilleur.}",
        r"\label{tab:ablation}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{clccccccccc}",
        r"\toprule",
        r"\textbf{C\#} & \textbf{Configuration} & "
        r"\textbf{MAE pos.~($^\circ$)}$\downarrow$ & "
        r"\textbf{Haversine méd.~(m)}$\downarrow$ & "
        r"\textbf{Ori. méd.~($^\circ$)}$\downarrow$ & "
        r"\textbf{Params tot.} & "
        r"\textbf{Params entr.} & "
        r"\textbf{Taille (Mo)}$\downarrow$ & "
        r"\textbf{Inférence (ms)}$\downarrow$ & "
        r"\textbf{Canaux élagués (\%)} \\",
        r"\midrule",
    ]
    for _, row in df.iterrows():
        cfg_id = int(row["Config"])
        desc = row["Description"].replace("+", r"\texttt{+}")
        if cfg_id == 8:
            desc = r"\textbf{" + desc + r"}"
        line = (
            f"{cfg_id} & {desc} & "
            f"{row['MAE pos. (°)']:.6f} & "
            f"{row['Haversine méd. (m)']:.2f} & "
            f"{row['Err. orient. méd. (°)']:.3f} & "
            f"{row['Params total']:,} & "
            f"{row['Params entraîn.']:,} & "
            f"{row['Taille (Mo)']:.2f} & "
            f"{row['Inférence (ms)']:.2f} & "
            f"{row.get('Canaux élagués (%)', 0.0):.1f} \\\\"
        )
        lines.append(line)
    lines += [
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\end{table}",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Point d'entrée
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Étude d'ablation progressive — MobileNet-APR-Campus"
    )
    parser.add_argument("--images_dir",    type=str, default="./")
    parser.add_argument("--combined_csv",  type=str, default=None,
                        help="Fichier CSV unique (image_name,lat,lon,azimuth,pitch,roll). "
                             "Si fourni, --positions_csv et --angles_csv sont ignorés.")
    parser.add_argument("--positions_csv", type=str, default="./positions.csv")
    parser.add_argument("--angles_csv",    type=str, default="./angles.csv")
    parser.add_argument("--epochs",        type=int, default=30,
                        help="Nombre d'époques par configuration (défaut 30)")
    parser.add_argument("--batch_size",    type=int, default=32)
    parser.add_argument("--lr",            type=float, default=1e-3,
                        help="Taux d'apprentissage initial (défaut 1e-3)")
    parser.add_argument("--val_size",      type=float, default=0.15,
                        help="Fraction du jeu TOTAL réservée à la validation "
                             "(early stopping), distincte du test (défaut 0.15)")
    parser.add_argument("--seed",          type=int, default=42,
                        help="Graine aléatoire (split des données, k-means)")
    args = parser.parse_args()
    run_ablation(args)


if __name__ == "__main__":
    main()
