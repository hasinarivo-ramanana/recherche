"""
Fonctions de perte pour MobileNet-APR-Campus.

Conforme à la Section "Fonction de Perte Composée" du manuscrit :

  - Perte de position (Éq. loss_pos) : erreur L1 (MAE), robuste aux valeurs
    aberrantes liées aux erreurs GNSS ponctuelles.

        L_pos = ||phi~ - phi~_hat||_1 + ||lambda~ - lambda~_hat||_1

  - Perte d'orientation quaternionique (Éq. loss_ori), tenant compte de
    l'équivalence q ~ -q (même rotation) :

        L_ori = 1 - |q_hat . q_ref|^2

  - Pondération apprise par incertitude homoscédastique de Kendall & Gal
    (Éq. loss_learned) :

        L(theta, s_x, s_q) = (1 / (2 e^{s_x})) * L_pos
                            + (1 / (2 e^{s_q})) * L_ori
                            + s_x + s_q

    où s_x, s_q sont deux scalaires appris par rétropropagation jouant le
    rôle de logarithmes des variances.
"""

import tensorflow as tf
from tensorflow.keras import layers


def position_loss(y_true, y_pred):
    """
    Perte L1 (MAE) sur la position normalisée (phi_tilde, lambda_tilde).
    Conforme à l'Éq. loss_pos : somme des erreurs absolues par composante,
    moyennée sur le lot.
    """
    return tf.reduce_mean(tf.reduce_sum(tf.abs(y_true - y_pred), axis=-1))


def quaternion_geodesic_loss(y_true, y_pred):
    """
    Perte d'orientation quaternionique géodésique (Éq. loss_ori) :

        L_ori = 1 - |q_hat . q_ref|^2

    Le carré de la valeur absolue du produit scalaire rend la perte
    invariante au signe du quaternion (q et -q représentent la même
    rotation). Minimale (0) lorsque les quaternions sont colinéaires
    (orientations identiques ou antipodales), maximale (1) lorsqu'ils sont
    orthogonaux.
    """
    dot_product = tf.reduce_sum(y_true * y_pred, axis=-1)
    return tf.reduce_mean(1.0 - tf.square(dot_product))


class HomoscedasticUncertaintyWeightedLoss(layers.Layer):
    """
    Pondération apprise de la perte composée par incertitude
    homoscédastique (Kendall & Gal), conforme à l'Éq. loss_learned :

        L(theta, s_x, s_q) = (1 / (2 e^{s_x})) * L_pos
                            + (1 / (2 e^{s_q})) * L_ori
                            + s_x + s_q

    s_x et s_q sont des scalaires entraînables (logarithmes des variances),
    initialisés à 0 (variance initiale = 1, c.-à-d. poids initial 0.5 pour
    chaque terme), appris conjointement avec les poids du réseau par
    rétropropagation standard.

    Utilisation : cette couche est insérée dans le graphe Keras et calcule
    la perte totale à partir des deux pertes individuelles (déjà calculées),
    de sorte que s_x et s_q apparaissent comme des variables entraînables
    du modèle et reçoivent un gradient à chaque étape.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.s_x = self.add_weight(
            name="log_variance_position",
            shape=(),
            initializer="zeros",
            trainable=True,
        )
        self.s_q = self.add_weight(
            name="log_variance_orientation",
            shape=(),
            initializer="zeros",
            trainable=True,
        )

    def call(self, l_pos, l_ori):
        precision_pos = tf.exp(-self.s_x)
        precision_ori = tf.exp(-self.s_q)
        total = (
            0.5 * precision_pos * l_pos
            + 0.5 * precision_ori * l_ori
            + self.s_x
            + self.s_q
        )
        return total

    def get_log_variances(self):
        """Retourne (s_x, s_q) pour suivi/diagnostic pendant l'entraînement."""
        return float(self.s_x.numpy()), float(self.s_q.numpy())


class MobileNetAPRLoss(tf.keras.losses.Loss):
    """
    Wrapper combinant position_loss et quaternion_geodesic_loss avec un
    poids fixe beta (Éq. loss_total), pour les phases d'entraînement où la
    pondération apprise n'est pas utilisée. Fournie comme alternative plus
    simple à HomoscedasticUncertaintyWeightedLoss.
    """

    def __init__(self, beta=1.0, name="mobilenet_apr_loss"):
        super().__init__(name=name)
        self.beta = beta

    def call(self, y_true, y_pred):
        # Non utilisée directement : la perte composée nécessite deux
        # sorties séparées (position, orientation). Voir
        # `compute_total_loss` ci-dessous pour l'utilisation pratique avec
        # un modèle à sorties multiples.
        raise NotImplementedError(
            "Utiliser compute_total_loss(pos_true, pos_pred, ori_true, ori_pred, beta) "
            "pour un modele a sorties multiples."
        )


def compute_total_loss(pos_true, pos_pred, ori_true, ori_pred, beta=1.0):
    """
    Calcule la perte composée a poids fixe (Éq. loss_total) :

        L(theta) = L_pos(theta) + beta * L_ori(theta)
    """
    l_pos = position_loss(pos_true, pos_pred)
    l_ori = quaternion_geodesic_loss(ori_true, ori_pred)
    return l_pos + beta * l_ori, l_pos, l_ori
