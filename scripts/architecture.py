"""
Architecture MobileNet-APR-Campus.

Assemble :
  - le backbone MobileNetV2 pré-entraîné ImageNet (Section "MobileNet V2
    comme Extracteur de Caractéristiques")
  - le module d'attention spatiale CBAM-léger, inséré après la dernière
    convolution 1x1 (1280 filtres) et avant le Global Average Pooling
    (Section "Module d'attention sur les régions discriminantes")
  - la tête de régression duale avec normalisation quaternionique L2
    (Section "Tête de régression duale avec normalisation quaternionique")

Architecture de la tête (conforme au manuscrit) :
  Branche position    : Dense 1280->256 (ReLU) puis Dense 256->2  -> (phi~, lambda~)
  Branche orientation  : Dense 1280->256 (ReLU) puis Dense 256->4 (q_raw)
                         suivie d'une normalisation L2 -> q_hat (||q_hat||=1)
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV2

from cbam import apply_cbam


class L2NormalizeQuaternion(layers.Layer):
    """
    Couche de normalisation L2 du quaternion brut, garantissant la
    contrainte d'unitarité indépendamment des prédictions brutes du réseau
    (Éq. quat_normalize) :

        q_hat = q_raw / ||q_raw||_2
    """

    def call(self, q_raw):
        # epsilon très faible : on ne veut intervenir que dans le cas
        # degenere ||q_raw|| == 0 exactement, pas re-mettre a l'echelle des
        # quaternions bruts deja non nuls (cf. tf.math.l2_normalize qui
        # calcule x / max(norm(x), epsilon)).
        norm = tf.norm(q_raw, axis=-1, keepdims=True)
        norm = tf.maximum(norm, 1e-12)
        return q_raw / norm


def build_mobilenet_apr_campus(
    input_shape=(224, 224, 3),
    weights="imagenet",
    cbam_kernel_size=7,
    dense_units=256,
    dropout_rate=0.0,
    freeze_backbone=False,
):
    """
    Construit le modèle MobileNet-APR-Campus.

    Paramètres
    ----------
    input_shape      : forme des images d'entrée, (224, 224, 3) par défaut
                        conformément au manuscrit.
    weights          : poids de départ du backbone ("imagenet" ou None).
    cbam_kernel_size  : taille du noyau de la convolution spatiale CBAM.
    dense_units       : largeur des couches denses intermédiaires des deux
                        branches (256 dans le manuscrit).
    dropout_rate      : dropout optionnel appliqué après chaque couche
                        dense intermédiaire (0.0 = désactivé, conforme au
                        manuscrit qui ne mentionne pas de dropout dans la
                        tête ; paramètre laissé disponible pour
                        l'expérimentation).
    freeze_backbone   : si True, gèle les poids du backbone MobileNetV2
                        (utilisé en Phase 1 de l'entraînement par transfert).

    Retourne
    --------
    tf.keras.Model avec deux sorties nommées :
      - "position"    : (batch, 2)  -> (phi_tilde, lambda_tilde)
      - "orientation" : (batch, 4)  -> quaternion unitaire (qw, qx, qy, qz)
    """
    backbone = MobileNetV2(
        weights=weights,
        include_top=False,
        input_shape=input_shape,
    )
    backbone.trainable = not freeze_backbone

    # Carte de caractéristiques (H', W', 1280) en sortie du backbone,
    # correspondant à la couche "Conv_1" (1x1, 1280 filtres) de MobileNetV2.
    features = backbone.output

    # Module d'attention CBAM-léger (attention spatiale uniquement),
    # inséré après le bloc d'expansion global, avant le GAP.
    features = apply_cbam(features, kernel_size=cbam_kernel_size, name="cbam_spatial_attention")

    pooled = layers.GlobalAveragePooling2D(name="global_average_pool")(features)

    # --- Branche position --------------------------------------------
    pos_branch = layers.Dense(dense_units, activation="relu", name="position_dense_256")(pooled)
    if dropout_rate > 0:
        pos_branch = layers.Dropout(dropout_rate, name="position_dropout")(pos_branch)
    position_output = layers.Dense(2, activation="linear", name="position")(pos_branch)

    # --- Branche orientation -------------------------------------------
    ori_branch = layers.Dense(dense_units, activation="relu", name="orientation_dense_256")(pooled)
    if dropout_rate > 0:
        ori_branch = layers.Dropout(dropout_rate, name="orientation_dropout")(ori_branch)
    quaternion_raw = layers.Dense(4, activation="linear", name="quaternion_raw")(ori_branch)
    orientation_output = L2NormalizeQuaternion(name="orientation")(quaternion_raw)

    model = Model(
        inputs=backbone.input,
        outputs=[position_output, orientation_output],
        name="MobileNet_APR_Campus",
    )
    return model
