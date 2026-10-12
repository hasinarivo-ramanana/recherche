"""
Module d'attention CBAM-léger (Convolutional Block Attention Module).

Conforme à la Section "Module d'attention sur les régions discriminantes
(CBAM-léger)" du manuscrit : seule l'attention spatiale est retenue
(version allégée), insérée après le bloc d'expansion global (avant le
Global Average Pooling).

    M_s = sigmoid( conv_7x7( [AvgPool_channel(F) ; MaxPool_channel(F)] ) )
    F'  = F * M_s
"""

import tensorflow as tf
from tensorflow.keras import layers


class SpatialAttention(layers.Layer):
    """
    Attention spatiale CBAM-léger.

    Pour une carte de caractéristiques F de forme (H, W, C), calcule un
    pooling moyenne et un pooling max le long de l'axe des canaux, les
    concatène, applique une convolution 7x7 + sigmoïde pour produire la
    carte d'attention M_s de forme (H, W, 1), puis pondère F par M_s.
    """

    def __init__(self, kernel_size=7, **kwargs):
        super().__init__(**kwargs)
        self.kernel_size = kernel_size
        self.conv = layers.Conv2D(
            filters=1,
            kernel_size=kernel_size,
            padding="same",
            activation="sigmoid",
            use_bias=False,
            name="cbam_spatial_conv7x7",
        )

    def build(self, input_shape):
        self.conv.build((input_shape[0], input_shape[1], input_shape[2], 2))
        super().build(input_shape)

    def call(self, inputs):
        avg_pool = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        max_pool = tf.reduce_max(inputs, axis=-1, keepdims=True)
        concat = tf.concat([avg_pool, max_pool], axis=-1)
        attention_map = self.conv(concat)  # (H, W, 1), valeurs dans (0, 1)
        return inputs * attention_map

    def get_config(self):
        config = super().get_config()
        config.update({"kernel_size": self.kernel_size})
        return config


def apply_cbam(feature_map, kernel_size=7, name="cbam"):
    """Applique le module d'attention spatiale CBAM-léger à une carte de
    caractéristiques. Point d'entrée fonctionnel pratique pour l'API
    fonctionnelle de Keras."""
    return SpatialAttention(kernel_size=kernel_size, name=name)(feature_map)
