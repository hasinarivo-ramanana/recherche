# -*- coding: utf-8 -*-
"""
geo_utils.py
============
Fonctions geodesiques, de projection locale et de manipulation de
quaternions/rotations utilisees par les trois niveaux de localisation.
"""

import math
import numpy as np

EARTH_RADIUS_M = 6371000.0


# ----------------------------------------------------------------------
# Distances geodesiques
# ----------------------------------------------------------------------
def haversine(lat1, lon1, lat2, lon2):
    """Distance en metres entre deux points (lat, lon) en degres decimaux."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (math.sin(dphi / 2) ** 2
         + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2)
    c = 2 * math.asin(min(1.0, math.sqrt(a)))
    return EARTH_RADIUS_M * c


def azimuth_deg(lat1, lon1, lat2, lon2):
    """Azimut (cap, Nord=0 deg, sens horaire) du point 1 vers le point 2."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dlambda = math.radians(lon2 - lon1)
    x = math.sin(dlambda) * math.cos(phi2)
    y = (math.cos(phi1) * math.sin(phi2)
         - math.sin(phi1) * math.cos(phi2) * math.cos(dlambda))
    theta = math.atan2(x, y)
    return (math.degrees(theta) + 360.0) % 360.0


# ----------------------------------------------------------------------
# Projection locale (equirectangulaire) lat/lon <-> plan ENU (metres)
# Approximation valable a l'echelle d'un campus (quelques km).
# ----------------------------------------------------------------------
def latlon_to_local_xy(lat, lon, lat0, lon0):
    """
    Convertit (lat, lon) en coordonnees planes locales (x=Est, y=Nord), en
    metres, centrees sur l'origine (lat0, lon0).
    """
    lat_rad0 = math.radians(lat0)
    dlat = math.radians(lat - lat0)
    dlon = math.radians(lon - lon0)
    x = dlon * math.cos(lat_rad0) * EARTH_RADIUS_M
    y = dlat * EARTH_RADIUS_M
    return x, y


def local_xy_to_latlon(x, y, lat0, lon0):
    """Inverse de latlon_to_local_xy."""
    lat_rad0 = math.radians(lat0)
    dlat = y / EARTH_RADIUS_M
    dlon = x / (EARTH_RADIUS_M * math.cos(lat_rad0))
    lat = lat0 + math.degrees(dlat)
    lon = lon0 + math.degrees(dlon)
    return lat, lon


# ----------------------------------------------------------------------
# Quaternions / rotations (convention COLMAP & convention these : qw,qx,qy,qz)
# ----------------------------------------------------------------------
def quat_normalize(q):
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / n


def quat_to_rotmat(q):
    """q = (qw, qx, qy, qz) -> matrice de rotation 3x3."""
    qw, qx, qy, qz = quat_normalize(q)
    R = np.array([
        [1 - 2 * (qy ** 2 + qz ** 2), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx ** 2 + qz ** 2), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx ** 2 + qy ** 2)],
    ])
    return R


def rotmat_to_quat(R):
    """Matrice de rotation 3x3 -> quaternion (qw, qx, qy, qz)."""
    tr = np.trace(R)
    if tr > 0:
        S = math.sqrt(tr + 1.0) * 2
        qw = 0.25 * S
        qx = (R[2, 1] - R[1, 2]) / S
        qy = (R[0, 2] - R[2, 0]) / S
        qz = (R[1, 0] - R[0, 1]) / S
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        S = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        qw = (R[2, 1] - R[1, 2]) / S
        qx = 0.25 * S
        qy = (R[0, 1] + R[1, 0]) / S
        qz = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        qw = (R[0, 2] - R[2, 0]) / S
        qx = (R[0, 1] + R[1, 0]) / S
        qy = 0.25 * S
        qz = (R[1, 2] + R[2, 1]) / S
    else:
        S = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        qw = (R[1, 0] - R[0, 1]) / S
        qx = (R[0, 2] + R[2, 0]) / S
        qy = (R[1, 2] + R[2, 1]) / S
        qz = 0.25 * S
    return quat_normalize([qw, qx, qy, qz])


def quat_to_euler_deg(q):
    """
    Quaternion (qw,qx,qy,qz) -> angles d'Euler (yaw psi, pitch theta, roll phi_r)
    en degres, convention ZYX (cf. section 4 de la these).
    """
    qw, qx, qy, qz = quat_normalize(q)
    phi_r = math.atan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx ** 2 + qy ** 2))
    sin_theta = 2 * (qw * qy - qz * qx)
    sin_theta = max(-1.0, min(1.0, sin_theta))
    theta = math.asin(sin_theta)
    psi = math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy ** 2 + qz ** 2))
    return math.degrees(psi), math.degrees(theta), math.degrees(phi_r)


def camera_center(R, t):
    """Centre optique C = -R^T . t (convention monde->camera : X_cam = R.X_world + t)."""
    return -R.T @ t


# ----------------------------------------------------------------------
# Alignement de similarite (Umeyama) : COLMAP (arbitraire) -> ENU (metres)
# ----------------------------------------------------------------------
def umeyama_alignment(src, dst):
    """
    Calcule la transformation de similarite (R, s, t) telle que :
        dst_i ~= s * R @ src_i + t
    src, dst : arrays (N,3), N >= 3 points correspondants (non colineaires).
    Retourne (R (3,3), s (float), t (3,)).
    Reference : Umeyama, "Least-squares estimation of transformation
    parameters between two point patterns", PAMI 1991.
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    assert src.shape == dst.shape and src.shape[0] >= 3

    n, dim = src.shape
    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src_c = src - mu_src
    dst_c = dst - mu_dst

    var_src = (src_c ** 2).sum() / n

    cov = (dst_c.T @ src_c) / n
    U, D, Vt = np.linalg.svd(cov)

    S = np.eye(dim)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[-1, -1] = -1

    R = U @ S @ Vt
    s = np.trace(np.diag(D) @ S) / var_src if var_src > 1e-12 else 1.0
    t = mu_dst - s * R @ mu_src
    return R, s, t


def apply_similarity(R, s, t, points):
    """dst = s*R@src + t, points: (N,3) ou (3,)."""
    points = np.asarray(points, dtype=np.float64)
    if points.ndim == 1:
        return s * (R @ points) + t
    return (s * (R @ points.T).T) + t


def invert_similarity(R, s, t, points_dst):
    """src = R^T @ ((dst - t) / s)."""
    points_dst = np.asarray(points_dst, dtype=np.float64)
    if points_dst.ndim == 1:
        return R.T @ ((points_dst - t) / s)
    return (R.T @ ((points_dst - t) / s).T).T
