"""
Élagage structuré par sensibilité de canal (Channel Pruning).

Conforme à la Section "Élagage structuré par sensibilité de canal" du
manuscrit :

  - Le score de saillance d'un canal c est la norme L1 des gradients de la
    perte par rapport aux activations de ce canal, accumulée sur N exemples :

        s_c = sum_n | dL_n / da_c^(l) |                       (eq:pruning_score)

  - L'élagage est appliqué uniquement aux blocs de la moitié supérieure du
    réseau (blocs 7 à 17 de MobileNet V2, c.-à-d. block_6 à block_16 dans la
    nomenclature Keras, en comptant expanded_conv comme bloc 1).
  - Taux d'élagage cible : 30% des canaux internes (couches d'expansion
    "expand"), via un seuil adaptatif tau_l = p_l-ème percentile des scores
    de saillance de la couche l (p_l = 30).

Implémentation : Keras ne supporte pas nativement le retrait dynamique de
filtres d'une couche existante. La stratégie utilisée ici est l'élagage par
masquage structuré : les canaux de plus faible saillance sont mis à zéro de
façon permanente (poids ET biais), ce qui revient à les supprimer
fonctionnellement (sortie nulle, gradient nul lors du fine-tuning si l'on
gèle leur masque) tout en conservant une architecture Keras standard,
sérialisable et compatible avec le reste du pipeline (CBAM, tête duale).
Cela correspond exactement à la sémantique du score de saillance défini dans
le manuscrit : les canaux peu saillants sont "supprimés" (mis à zéro), les
autres conservés.
"""

import re
import numpy as np
import tensorflow as tf


# Blocs de la "moitié supérieure" du réseau visés par l'élagage
# (blocs 7 à 17 du manuscrit == block_6 à block_16 dans la nomenclature Keras,
# expanded_conv comptant comme bloc 1).
PRUNABLE_BLOCK_RANGE = list(range(6, 17))  # block_6 ... block_16


def get_prunable_expand_layers(model, block_range=PRUNABLE_BLOCK_RANGE):
    """
    Retourne la liste des couches Conv2D "expand" (1x1, expansion des
    canaux internes) des blocs ciblés par l'élagage.
    """
    pattern = re.compile(r"^block_(\d+)_expand$")
    layers = []
    for layer in model.layers:
        m = pattern.match(layer.name)
        if m and int(m.group(1)) in block_range:
            layers.append(layer)
    return layers


def compute_channel_saliency(model, layer, data_batches, loss_fn):
    """
    Calcule le score de saillance s_c (Éq. pruning_score) de chaque canal de
    sortie de `layer`, accumulé sur plusieurs lots de données.

    Paramètres
    ----------
    model       : tf.keras.Model complet (utilisé pour le calcul du gradient)
    layer       : couche Conv2D dont on évalue les canaux de sortie
    data_batches: itérable de tuples (images, targets) représentatifs
                  de l'ensemble d'entraînement
    loss_fn     : fonction (model, images, targets) -> perte scalaire,
                  utilisée pour le calcul de dL/da_c

    Retourne
    --------
    np.ndarray de forme (C,) : score de saillance par canal.
    """
    saliency = None
    for images, targets in data_batches:
        with tf.GradientTape() as tape:
            activation_holder = {}

            # On intercepte l'activation de la couche cible via un modèle
            # auxiliaire à deux sorties (sortie originale + activation
            # intermédiaire), ce qui permet d'obtenir dL/da_c proprement.
            intermediate_model = tf.keras.Model(
                inputs=model.input,
                outputs=[model.output, layer.output]
                if not isinstance(model.output, (list, tuple))
                else list(model.output) + [layer.output],
            )
            outputs = intermediate_model(images, training=True)
            *preds, activation = outputs if isinstance(outputs, (list, tuple)) else (outputs,)
            tape.watch(activation)
            loss = loss_fn(preds if len(preds) > 1 else preds[0], targets)

        grads = tape.gradient(loss, activation)
        if grads is None:
            continue
        # |dL/da_c| sommé sur le lot, la hauteur et la largeur spatiales
        batch_saliency = tf.reduce_sum(tf.abs(grads), axis=[0, 1, 2]).numpy()
        saliency = batch_saliency if saliency is None else saliency + batch_saliency

    if saliency is None:
        # Repli : pas de gradient calculable (ex. pas de données fournies) ->
        # score uniforme, aucun canal n'est privilégié.
        saliency = np.ones((layer.filters,), dtype=np.float32)
    return saliency


def channels_to_prune(saliency_scores, prune_ratio=0.30):
    """
    À partir des scores de saillance, retourne les indices des canaux à
    élaguer : ceux dont le score est inférieur au seuil adaptatif
    tau_l = (prune_ratio * 100)-ème percentile des scores de la couche.
    """
    threshold = np.percentile(saliency_scores, prune_ratio * 100)
    indices = np.where(saliency_scores <= threshold)[0]
    return indices


def compute_prune_masks(model, prune_ratio=0.30, data_batches=None, loss_fn=None,
                        block_range=PRUNABLE_BLOCK_RANGE):
    """
    Calcule (sans les appliquer) les indices de canaux à élaguer pour
    chaque bloc ciblé. Sépare le CALCUL du masque de son APPLICATION, afin
    de pouvoir réappliquer le même masque à chaque étape d'entraînement
    (voir maintain_pruning_masks ci-dessous) : sans cela, un fine-tuning
    ultérieur (optimiseur Adam, momentum, L2) peut faire "repousser" des
    poids pourtant mis à zéro, ce qui annule discrètement l'élagage.

    Retourne un dict {nom_de_bloc: indices_np_array}.
    """
    prunable_layers = get_prunable_expand_layers(model, block_range)
    masks = {}
    for expand_layer in prunable_layers:
        if data_batches is not None and loss_fn is not None:
            saliency = compute_channel_saliency(model, expand_layer, data_batches, loss_fn)
        else:
            # Heuristique de repli : magnitude L1 des poids du filtre,
            # utilisée classiquement en l'absence de données de calibration.
            weights = expand_layer.get_weights()[0]  # (1, 1, C_in, C_out)
            saliency = np.sum(np.abs(weights), axis=(0, 1, 2))
        prune_idx = channels_to_prune(saliency, prune_ratio)
        block_name = expand_layer.name.replace("_expand", "")
        masks[block_name] = prune_idx
    return masks


def apply_prune_masks(model, masks):
    """
    Applique (met à zéro) les canaux désignés par `masks` sur les couches
    expand / depthwise / project de chaque bloc concerné.

    Appeler cette fonction UNE FOIS après compute_prune_masks pour élaguer
    le modèle, puis À NOUVEAU après chaque mise à jour de gradient pendant
    le fine-tuning (cf. maintain_pruning_masks), pour que l'élagage reste
    effectif tout au long de l'entraînement.
    """
    report = {}
    for block_name, prune_idx in masks.items():
        if len(prune_idx) == 0:
            report[block_name] = 0
            continue
        expand_layer = model.get_layer(f"{block_name}_expand")
        depthwise_layer = model.get_layer(f"{block_name}_depthwise")
        project_layer = model.get_layer(f"{block_name}_project")

        _zero_out_channels(expand_layer, prune_idx, axis="output")
        _zero_out_depthwise_channels(depthwise_layer, prune_idx)
        _zero_out_channels(project_layer, prune_idx, axis="input")
        report[block_name] = len(prune_idx)
    return report


def maintain_pruning_masks(model, masks):
    """
    À appeler APRÈS chaque optimizer.apply_gradients() pendant une phase
    d'entraînement où le modèle reste élagué (ex. fine-tuning post-pruning,
    ou tout entraînement d'une config construite à partir d'un modèle déjà
    élagué). Ne fait rien si masks est None ou vide.

    Léger (quelques affectations numpy par étape) ; nécessaire car
    l'élagage par masquage n'est pas une contrainte structurelle de
    l'architecture Keras — rien n'empêche l'optimiseur de redonner un
    poids non nul à un canal élagué sauf à le remettre à zéro explicitement.
    """
    if not masks:
        return
    apply_prune_masks(model, masks)


def apply_structured_pruning(model, prune_ratio=0.30, data_batches=None, loss_fn=None):
    """
    Calcule ET applique l'élagage structuré par sensibilité de canal sur
    les couches "expand" des blocs ciblés (blocs 7 à 17 du manuscrit).
    Conservée pour compatibilité ascendante (ex. train.py) ; en interne,
    délègue à compute_prune_masks + apply_prune_masks.

    IMPORTANT : cette fonction seule ne maintient PAS le masque pendant un
    entraînement ultérieur. Pour un fine-tuning post-pruning, récupérer le
    dict de masques via compute_prune_masks et appeler
    maintain_pruning_masks(model, masks) après chaque étape de gradient
    (voir ablation_study.py et train.py pour un exemple d'utilisation).

    Retourne un dictionnaire {nom_de_couche: nombre_de_canaux_élagués} pour
    le rapport d'élagage.
    """
    masks = compute_prune_masks(model, prune_ratio, data_batches, loss_fn)
    return apply_prune_masks(model, masks)


def compute_sparsity(model, masks):
    """
    Fraction de canaux effectivement élagués (mis à zéro) sur l'ensemble
    des canaux visés par `masks`, toutes couches confondues. Utile pour
    rapporter honnêtement que l'élagage par masquage NE réduit PAS le
    nombre de paramètres non nuls stockés (contrairement à un vrai
    pruning structurel suivi d'une reconstruction), seulement leur
    contribution fonctionnelle.
    """
    if not masks:
        return 0.0
    total_channels, pruned_channels = 0, 0
    for block_name, prune_idx in masks.items():
        expand_layer = model.get_layer(f"{block_name}_expand")
        total_channels += expand_layer.filters
        pruned_channels += len(prune_idx)
    return 100.0 * pruned_channels / max(1, total_channels)


def _zero_out_channels(conv_layer, channel_indices, axis):
    """Met à zéro les poids (et biais le cas échéant) d'une couche Conv2D
    1x1, soit sur l'axe des canaux de sortie (axis='output'), soit sur
    l'axe des canaux d'entrée (axis='input')."""
    if len(channel_indices) == 0:
        return
    weights = conv_layer.get_weights()
    kernel = weights[0]  # (kh, kw, C_in, C_out)
    if axis == "output":
        kernel[..., channel_indices] = 0.0
        if len(weights) > 1:  # biais présent
            bias = weights[1]
            bias[channel_indices] = 0.0
            conv_layer.set_weights([kernel, bias])
            return
    elif axis == "input":
        kernel[:, :, channel_indices, :] = 0.0
    conv_layer.set_weights([kernel] + weights[1:])


def _zero_out_depthwise_channels(depthwise_layer, channel_indices):
    """Met à zéro les canaux ciblés d'une couche DepthwiseConv2D (un filtre
    par canal d'entrée)."""
    if len(channel_indices) == 0:
        return
    weights = depthwise_layer.get_weights()
    kernel = weights[0]  # (kh, kw, C_in, 1) pour depth_multiplier=1
    kernel[:, :, channel_indices, :] = 0.0
    depthwise_layer.set_weights([kernel] + weights[1:])
