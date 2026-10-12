"""
Script principal d'entraînement de MobileNet-APR-Campus.

Reproduit le protocole d'entraînement en trois phases décrit dans la
Section "Protocole d'Entraînement et Évaluation" du manuscrit :

  Phase 1 - Gel du backbone et entraînement de la tête (10 époques)
  Phase 2 - Affinage global (30 époques), LR/10, cosine annealing
  Phase 3 - Élagage structuré (30% des canaux, blocs 7-17) puis
            ré-entraînement (20 époques), LR très faible

La perte composée utilise la pondération apprise par incertitude
homoscédastique (Kendall & Gal), cf. losses.py.

Usage :
    python train.py --images_dir ./ --positions_csv ./positions.csv \
                     --angles_csv ./angles.csv
"""

import argparse
import os

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from architecture import build_mobilenet_apr_campus
from data_loader import load_dataset
from losses import position_loss, quaternion_geodesic_loss, HomoscedasticUncertaintyWeightedLoss
from metrics import evaluate_model
from pruning import compute_prune_masks, apply_prune_masks, maintain_pruning_masks


# ---------------------------------------------------------------------------
# Boucle d'entraînement personnalisée
#
# Une train_step custom est nécessaire car la perte composée (Éq.
# loss_learned) dépend de DEUX sorties du modèle (position, orientation) ET
# de deux variables apprises supplémentaires (s_x, s_q) qui ne font pas
# partie du modèle Keras lui-même (elles vivent dans la couche
# HomoscedasticUncertaintyWeightedLoss). model.compile()/model.fit() ne
# permettent pas nativement d'optimiser conjointement des variables
# "externes" au modèle de cette façon avec une perte couplée comme l'Éq.
# loss_learned ; une boucle explicite donne un contrôle total et reste
# fidèle à la formulation du manuscrit.
# ---------------------------------------------------------------------------
class Trainer:
    def __init__(self, model, optimizer, uncertainty_loss, prune_masks=None):
        """
        prune_masks : dict {bloc: indices_canaux}, optionnel (cf.
        pruning.compute_prune_masks). Lorsqu'il est fourni, le masque est
        réappliqué après CHAQUE étape de gradient (CORRECTIF : sans cela,
        la Phase 3 d'affinage post-élagage peut redonner un poids non nul
        aux canaux censés être élagués, annulant discrètement le pruning).
        """
        self.model = model
        self.optimizer = optimizer
        self.uncertainty_loss = uncertainty_loss
        self.prune_masks = prune_masks
        # Les variables entraînables de la perte (s_x, s_q) doivent être
        # optimisées conjointement avec celles du modèle.
        self.trainable_vars = lambda: (
            self.model.trainable_variables + self.uncertainty_loss.trainable_variables
        )

    def train_step(self, images, pos_true, quat_true):
        # NOTE : plus décoré par @tf.function au niveau de la méthode, car
        # le maintien du masque de pruning (maintain_pruning_masks) modifie
        # les poids en numpy pur (model.get_weights/set_weights), ce qui
        # n'est pas traçable dans un graphe tf.function. L'essentiel du
        # coût (forward + backward) reste de toute façon sur le graphe
        # interne de self.model.
        with tf.GradientTape() as tape:
            pos_pred, quat_pred = self.model(images, training=True)
            l_pos = position_loss(pos_true, pos_pred)
            l_ori = quaternion_geodesic_loss(quat_true, quat_pred)
            total_loss = self.uncertainty_loss(l_pos, l_ori)

        variables = self.trainable_vars()
        grads = tape.gradient(total_loss, variables)
        self.optimizer.apply_gradients(zip(grads, variables))

        if self.prune_masks:
            maintain_pruning_masks(self.model, self.prune_masks)

        return total_loss, l_pos, l_ori

    @tf.function
    def test_step(self, images, pos_true, quat_true):
        pos_pred, quat_pred = self.model(images, training=False)
        l_pos = position_loss(pos_true, pos_pred)
        l_ori = quaternion_geodesic_loss(quat_true, quat_pred)
        total_loss = self.uncertainty_loss(l_pos, l_ori)
        return total_loss, l_pos, l_ori

    def run_epoch(self, dataset, training=True):
        step_fn = self.train_step if training else self.test_step
        total_losses, pos_losses, ori_losses = [], [], []
        for images, pos_true, quat_true in dataset:
            total_loss, l_pos, l_ori = step_fn(images, pos_true, quat_true)
            total_losses.append(float(total_loss))
            pos_losses.append(float(l_pos))
            ori_losses.append(float(l_ori))
        return float(np.mean(total_losses)), float(np.mean(pos_losses)), float(np.mean(ori_losses))


def make_tf_dataset(images, positions, quaternions, batch_size=32, shuffle=True):
    ds = tf.data.Dataset.from_tensor_slices((images, positions, quaternions))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(images), seed=42)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def run_training_loop(trainer, train_ds, val_ds, epochs, phase_name, history):
    """
    CORRECTIF : le second dataset est désormais explicitement nommé val_ds
    et doit être le jeu de VALIDATION (data["X_val"] côté main()), distinct
    du jeu de test. Dans la version initiale, c'est le jeu de test qui
    était passé ici sous le nom test_ds et utilisé comme validation
    pendant les 3 phases d'entraînement, ce qui le "contaminait" avant
    l'évaluation finale.
    """
    for epoch in range(1, epochs + 1):
        train_total, train_pos, train_ori = trainer.run_epoch(train_ds, training=True)
        val_total, val_pos, val_ori = trainer.run_epoch(val_ds, training=False)
        s_x, s_q = trainer.uncertainty_loss.get_log_variances()
        print(
            f"[{phase_name}] Époque {epoch}/{epochs} - "
            f"train_loss={train_total:.4f} (pos={train_pos:.4f}, ori={train_ori:.4f}) | "
            f"val_loss={val_total:.4f} (pos={val_pos:.4f}, ori={val_ori:.4f}) | "
            f"s_x={s_x:.3f}, s_q={s_q:.3f}"
        )
        history["train_total"].append(train_total)
        history["train_pos"].append(train_pos)
        history["train_ori"].append(train_ori)
        history["val_total"].append(val_total)
        history["val_pos"].append(val_pos)
        history["val_ori"].append(val_ori)


def main():
    parser = argparse.ArgumentParser(description="Entraînement de MobileNet-APR-Campus")
    parser.add_argument("--images_dir", type=str, default="./")
    parser.add_argument("--positions_csv", type=str, default="./positions.csv")
    parser.add_argument("--angles_csv", type=str, default="./angles.csv")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--phase1_epochs", type=int, default=10)
    parser.add_argument("--phase2_epochs", type=int, default=30)
    parser.add_argument("--phase3_epochs", type=int, default=20)
    parser.add_argument("--prune_ratio", type=float, default=0.30)
    parser.add_argument("--output_model", type=str, default="mobilenet_apr_campus.keras")
    args = parser.parse_args()

    # -----------------------------------------------------------------
    # Chargement et préparation des données
    # -----------------------------------------------------------------
    # CORRECTIF : load_dataset retourne désormais un split train/val/test à
    # trois voies (voir data_loader.py). On entraîne et on fait l'early
    # stopping sur (train, val) ; le test n'est utilisé qu'à l'évaluation
    # finale, en bas de ce script.
    data = load_dataset(
        positions_csv_path=args.positions_csv,
        angles_csv_path=args.angles_csv,
        images_dir=args.images_dir,
    )

    train_ds = make_tf_dataset(
        data["X_train"], data["pos_train"], data["quat_train"], batch_size=args.batch_size
    )
    val_ds = make_tf_dataset(
        data["X_val"], data["pos_val"], data["quat_val"], batch_size=args.batch_size, shuffle=False
    )
    test_ds = make_tf_dataset(
        data["X_test"], data["pos_test"], data["quat_test"], batch_size=args.batch_size, shuffle=False
    )

    history = {k: [] for k in ["train_total", "train_pos", "train_ori", "val_total", "val_pos", "val_ori"]}

    # -----------------------------------------------------------------
    # Phase 1 : backbone gelé, entraînement de la tête seule (10 époques)
    # -----------------------------------------------------------------
    print("\n=== Phase 1 : gel du backbone, entraînement de la tête ===")
    model = build_mobilenet_apr_campus(weights="imagenet", freeze_backbone=True)
    uncertainty_loss = HomoscedasticUncertaintyWeightedLoss()
    optimizer_phase1 = tf.keras.optimizers.Adam(learning_rate=1e-3)
    trainer = Trainer(model, optimizer_phase1, uncertainty_loss)
    run_training_loop(trainer, train_ds, val_ds, args.phase1_epochs, "Phase 1", history)

    # -----------------------------------------------------------------
    # Phase 2 : affinage global, backbone dégelé, LR/10, cosine annealing
    # (30 époques)
    # -----------------------------------------------------------------
    print("\n=== Phase 2 : affinage global (backbone dégelé) ===")
    model.trainable = True
    steps_per_epoch = max(1, len(data["X_train"]) // args.batch_size)
    cosine_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=1e-4,  # 1e-3 / 10
        decay_steps=args.phase2_epochs * steps_per_epoch,
    )
    optimizer_phase2 = tf.keras.optimizers.Adam(learning_rate=cosine_schedule)
    trainer = Trainer(model, optimizer_phase2, uncertainty_loss)
    run_training_loop(trainer, train_ds, val_ds, args.phase2_epochs, "Phase 2", history)

    # -----------------------------------------------------------------
    # Phase 3 : élagage structuré (30% des canaux, blocs 7-17) puis
    # ré-entraînement à très faible LR (20 époques)
    # -----------------------------------------------------------------
    print("\n=== Phase 3 : élagage structuré puis ré-entraînement ===")

    def loss_fn_for_pruning(preds, targets):
        pos_pred, ori_pred = preds
        pos_true, ori_true = targets
        l_pos = position_loss(pos_true, pos_pred)
        l_ori = quaternion_geodesic_loss(ori_true, ori_pred)
        return l_pos + l_ori

    # Quelques lots représentatifs de l'ensemble d'entraînement pour le
    # calcul du score de saillance (Éq. pruning_score).
    calibration_batches = []
    for images, pos_true, quat_true in train_ds.take(5):
        calibration_batches.append((images, (pos_true, quat_true)))

    # CORRECTIF : compute_prune_masks calcule le masque SANS modifier le
    # modèle ; apply_prune_masks l'applique une première fois ; le Trainer
    # reçoit ensuite ce même masque et le réapplique après CHAQUE étape de
    # gradient du ré-entraînement (prune_masks=...). Dans la version
    # initiale, apply_structured_pruning élaguait une seule fois avant la
    # boucle, et rien n'empêchait l'optimiseur de "faire repousser"
    # certains poids pendant les 20 époques de Phase 3.
    prune_masks = compute_prune_masks(
        model,
        prune_ratio=args.prune_ratio,
        data_batches=calibration_batches,
        loss_fn=loss_fn_for_pruning,
    )
    pruning_report = apply_prune_masks(model, prune_masks)
    print("Rapport d'élagage (canaux mis à zéro par bloc) :", pruning_report)

    optimizer_phase3 = tf.keras.optimizers.Adam(learning_rate=1e-5)
    trainer = Trainer(model, optimizer_phase3, uncertainty_loss, prune_masks=prune_masks)
    run_training_loop(trainer, train_ds, val_ds, args.phase3_epochs, "Phase 3", history)

    # -----------------------------------------------------------------
    # Sauvegarde et visualisation
    # -----------------------------------------------------------------
    model.save(args.output_model)
    print(f"\nModèle sauvegardé : {args.output_model}")

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(history["train_total"], label="Train")
    plt.plot(history["val_total"], label="Validation")
    plt.title("Perte totale pondérée")
    plt.xlabel("Époque")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(history["train_pos"], label="Train")
    plt.plot(history["val_pos"], label="Validation")
    plt.title("Perte de position (L1)")
    plt.xlabel("Époque")
    plt.ylabel("MAE normalisée")
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(history["train_ori"], label="Train")
    plt.plot(history["val_ori"], label="Validation")
    plt.title("Perte d'orientation (géodésique quaternion)")
    plt.xlabel("Époque")
    plt.ylabel("1 - |q.q_ref|^2")
    plt.legend()

    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=150)
    print("Courbes d'entraînement sauvegardées : training_curves.png")

    # -----------------------------------------------------------------
    # Évaluation finale (métriques conformes au manuscrit)
    # -----------------------------------------------------------------
    results = evaluate_model(
        model,
        data["X_test"],
        data["pos_test"],
        data["quat_test"],
        data["position_normalizer"],
        success_threshold_m=2.0,  # seuil du Niveau 3 (PG-PnP) du pipeline hiérarchique
    )

    # Dénormaliser les positions pour pouvoir afficher les coordonnées réelles
    pos_pred_norm, quat_pred = model.predict(data["X_test"], verbose=0)
    pos_pred_latlon = data["position_normalizer"].inverse_transform(pos_pred_norm)
    pos_true_latlon = data["position_normalizer"].inverse_transform(data["pos_test"])

    # Distance de Haversine individuelle par image (Éq. manuscrit) :
    #   d(y_hat_i, y_i) = 2R * arcsin( sqrt( sin²(Δφ/2)
    #                      + cos(φ1)*cos(φ2)*sin²(Δλ/2) ) )
    R = 6371000.0  # rayon terrestre en mètres
    lat1 = np.radians(pos_true_latlon[:, 0])
    lon1 = np.radians(pos_true_latlon[:, 1])
    lat2 = np.radians(pos_pred_latlon[:, 0])
    lon2 = np.radians(pos_pred_latlon[:, 1])
    delta_phi    = lat2 - lat1
    delta_lambda = lon2 - lon1
    a = (np.sin(delta_phi / 2.0) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin(delta_lambda / 2.0) ** 2)
    haversine_distances_m = 2.0 * R * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))

    print("\n" + "=" * 72)
    print("  RÉSULTATS D'ÉVALUATION FINALE — MobileNet-APR-Campus")
    print("=" * 72)

    # Tableau ligne par ligne
    header = (f"{'Image':<20} {'Lat réf':>12} {'Lon réf':>12} "
              f"{'Lat préd':>12} {'Lon préd':>12} {'d_Haversine (m)':>17}")
    print("\n" + header)
    print("-" * len(header))
    for i, d_m in enumerate(haversine_distances_m):
        # Récupérer le nom de l'image (index dans le split test)
        img_name = f"img_test_{i+1:03d}"
        print(
            f"{img_name:<20} "
            f"{pos_true_latlon[i,0]:>12.8f} "
            f"{pos_true_latlon[i,1]:>12.8f} "
            f"{pos_pred_latlon[i,0]:>12.8f} "
            f"{pos_pred_latlon[i,1]:>12.8f} "
            f"{d_m:>17.3f}"
        )

    # Récapitulatif statistique
    print("-" * len(header))
    print(f"\n  Formule appliquée : d = 2R·arcsin(√(sin²(Δφ/2) + cos(φ₁)·cos(φ₂)·sin²(Δλ/2)))")
    print(f"  R = {R:.0f} m\n")
    print(f"  Distance Haversine minimale   : {haversine_distances_m.min():.3f} m")
    print(f"  Distance Haversine maximale   : {haversine_distances_m.max():.3f} m")
    print(f"  Distance Haversine moyenne    : {haversine_distances_m.mean():.3f} m")
    print(f"  Distance Haversine médiane    : {np.median(haversine_distances_m):.3f} m")
    print(f"  Taux de succès (< 2 m)        : {results['success_rate']*100:.1f}%")
    print(f"  MAE position (degrés décim.)  : {results['position_mae_degrees']:.6f}°")
    print(f"  Erreur orientation médiane    : {results['median_orientation_error_deg']:.3f}°")
    print("=" * 72)

    model.summary()

    model.summary()


if __name__ == "__main__":
    main()
