"""
Métriques d'évaluation pour MobileNet-APR-Campus.

Conforme à la Section "Comparaison avec les Méthodes APR de Référence" et
aux équations commentées (eq:error_ori) du manuscrit :

  - Distance géodésique de Haversine entre la position prédite et la
    position de référence (en mètres), calculée sur les coordonnées
    DÉNORMALISÉES (degrés décimaux réels, pas les valeurs normalisées
    phi_tilde/lambda_tilde utilisées pendant l'entraînement).

  - Erreur absolue moyenne (MAE) sur (latitude, longitude), conforme à
    L_perf du manuscrit.

  - Erreur d'orientation médiane (Éq. error_ori), en degrés :

        e_ori = 2 * arccos(|q_hat . q_ref|)
"""

import numpy as np

EARTH_RADIUS_KM = 6371.0


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Distance géodésique de Haversine (en mètres) entre deux points donnés
    en degrés décimaux. Conforme à la formule du manuscrit :

        d = 2R * arcsin( sqrt( sin^2(delta_phi/2)
                                + cos(phi1)*cos(phi2)*sin^2(delta_lambda/2) ) )

    Accepte des scalaires ou des tableaux numpy (calcul vectorisé).
    """
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)

    delta_phi = lat2_rad - lat1_rad
    delta_lambda = lon2_rad - lon1_rad

    a = (
        np.sin(delta_phi / 2.0) ** 2
        + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(delta_lambda / 2.0) ** 2
    )
    c = 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    distance_km = EARTH_RADIUS_KM * c
    return distance_km * 1000.0  # conversion en mètres


def position_mae(y_true_latlon, y_pred_latlon):
    """
    Erreur absolue moyenne (MAE) sur les coordonnées dénormalisées
    (latitude, longitude), conforme à L_perf du manuscrit :

        L_perf = (1/N) * sum_i ( |lat_hat_i - lat_i| + |lon_hat_i - lon_i| )
    """
    y_true_latlon = np.asarray(y_true_latlon)
    y_pred_latlon = np.asarray(y_pred_latlon)
    return float(np.mean(np.sum(np.abs(y_pred_latlon - y_true_latlon), axis=-1)))


def median_position_error_meters(y_true_latlon, y_pred_latlon):
    """
    Erreur de position médiane (en mètres), calculée via la distance de
    Haversine entre chaque paire (prédiction, référence) dénormalisée.
    """
    y_true_latlon = np.asarray(y_true_latlon)
    y_pred_latlon = np.asarray(y_pred_latlon)
    distances = haversine_distance(
        y_true_latlon[:, 0], y_true_latlon[:, 1],
        y_pred_latlon[:, 0], y_pred_latlon[:, 1],
    )
    return float(np.median(distances)), distances


def median_orientation_error_degrees(q_true, q_pred):
    """
    Erreur d'orientation médiane (Éq. error_ori), en degrés :

        e_ori = 2 * arccos(|q_hat . q_ref|)

    L'usage de la valeur absolue rend la mesure invariante au signe du
    quaternion (q et -q représentent la même rotation).
    """
    q_true = np.asarray(q_true)
    q_pred = np.asarray(q_pred)
    dot = np.sum(q_true * q_pred, axis=-1)
    dot = np.clip(np.abs(dot), 0.0, 1.0)  # évite les erreurs numériques hors [0,1]
    angles_rad = 2.0 * np.arccos(dot)
    angles_deg = np.degrees(angles_rad)
    return float(np.median(angles_deg)), angles_deg


def success_rate(distances_m, threshold_m):
    """
    Taux de succès : proportion de prédictions dont l'erreur de position
    est inférieure au seuil donné (ex. <2 m pour le niveau 3, <10 m pour le
    niveau 2 du pipeline hiérarchique du manuscrit).
    """
    distances_m = np.asarray(distances_m)
    return float(np.mean(distances_m < threshold_m))


def evaluate_model(model, X_test, pos_test_normalized, quat_test, position_normalizer, success_threshold_m=2.0):
    """
    Évalue le modèle complet sur un ensemble de test et retourne un
    dictionnaire de métriques conformes au manuscrit (MAE, distance de
    Haversine médiane, taux de succès, erreur d'orientation médiane).
    """
    pos_pred_norm, quat_pred = model.predict(X_test, verbose=0)

    pos_pred_latlon = position_normalizer.inverse_transform(pos_pred_norm)
    pos_true_latlon = position_normalizer.inverse_transform(pos_test_normalized)

    mae = position_mae(pos_true_latlon, pos_pred_latlon)
    median_dist_m, all_distances = median_position_error_meters(pos_true_latlon, pos_pred_latlon)
    rate = success_rate(all_distances, success_threshold_m)
    median_ori_deg, all_ori_errors = median_orientation_error_degrees(quat_test, quat_pred)

    return {
        "position_mae_degrees": mae,
        "median_position_error_m": median_dist_m,
        "success_rate": rate,
        "success_threshold_m": success_threshold_m,
        "median_orientation_error_deg": median_ori_deg,
        "all_position_errors_m": all_distances,
        "all_orientation_errors_deg": all_ori_errors,
    }
