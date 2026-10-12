# -*- coding: utf-8 -*-
"""
colmap_io.py
============
Lecture des fichiers de sortie COLMAP :
  - points3D.txt   (nuage de points 3D + tracks)
  - images.txt     (poses des images d'entrainement + observations 2D)
  - cameras.txt    (intrinseques, optionnel)
  - database.db    (keypoints + descripteurs SIFT, format SQLite)

Formats de reference : https://colmap.github.io/format.html
"""

import os
import sqlite3
import struct
import numpy as np


# ----------------------------------------------------------------------
# points3D.txt
# ----------------------------------------------------------------------
def read_points3D_text(path):
    """
    Retourne un dict : point3D_id -> {
        'xyz': np.array([X, Y, Z]),
        'rgb': (r, g, b),
        'error': float,
        'track': [(image_id, point2D_idx), ...]
    }
    """
    points = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            point3d_id = int(parts[0])
            xyz = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
            rgb = (int(parts[4]), int(parts[5]), int(parts[6]))
            error = float(parts[7])
            track = []
            for i in range(8, len(parts), 2):
                track.append((int(parts[i]), int(parts[i + 1])))
            points[point3d_id] = {
                "xyz": xyz, "rgb": rgb, "error": error, "track": track
            }
    return points


# ----------------------------------------------------------------------
# images.txt
# ----------------------------------------------------------------------
def read_images_text(path):
    """
    Retourne un dict : image_id -> {
        'qvec': np.array([qw,qx,qy,qz]),
        'tvec': np.array([tx,ty,tz]),
        'camera_id': int,
        'name': str,
        'points2D': [(x, y, point3D_id_or_-1), ...]
    }
    Le format images.txt alterne deux lignes par image :
      ligne 1: IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
      ligne 2: liste de (X, Y, POINT3D_ID) pour chaque keypoint
    """
    images = {}
    with open(path, "r", encoding="utf-8") as f:
        lines = [l for l in f if not l.startswith("#") and l.strip()]
    i = 0
    while i < len(lines):
        parts = lines[i].split()
        image_id = int(parts[0])
        qvec = np.array([float(parts[1]), float(parts[2]),
                          float(parts[3]), float(parts[4])])
        tvec = np.array([float(parts[5]), float(parts[6]), float(parts[7])])
        camera_id = int(parts[8])
        name = parts[9]

        points2D = []
        if i + 1 < len(lines):
            p2d_parts = lines[i + 1].split()
            for k in range(0, len(p2d_parts), 3):
                x = float(p2d_parts[k])
                y = float(p2d_parts[k + 1])
                pid = int(p2d_parts[k + 2])
                points2D.append((x, y, pid))
        images[image_id] = {
            "qvec": qvec, "tvec": tvec, "camera_id": camera_id,
            "name": name, "points2D": points2D
        }
        i += 2
    return images


# ----------------------------------------------------------------------
# cameras.txt (optionnel)
# ----------------------------------------------------------------------
def read_cameras_text(path):
    """
    Retourne un dict : camera_id -> {
        'model': str, 'width': int, 'height': int, 'params': [float,...]
    }
    """
    cameras = {}
    if not path or not os.path.exists(path):
        return cameras
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            camera_id = int(parts[0])
            model = parts[1]
            width = int(parts[2])
            height = int(parts[3])
            params = [float(p) for p in parts[4:]]
            cameras[camera_id] = {
                "model": model, "width": width, "height": height, "params": params
            }
    return cameras


def camera_to_K(camera):
    """Construit la matrice intrinseque K a partir d'une entree cameras.txt."""
    model = camera["model"]
    p = camera["params"]
    if model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "SIMPLE_RADIAL_FISHEYE"):
        f, cx, cy = p[0], p[1], p[2]
        fx = fy = f
    elif model in ("PINHOLE", "OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV"):
        fx, fy, cx, cy = p[0], p[1], p[2], p[3]
    elif model in ("RADIAL", "RADIAL_FISHEYE"):
        f, cx, cy = p[0], p[1], p[2]
        fx = fy = f
    else:
        # repli generique : on suppose les 4 premiers parametres = fx,fy,cx,cy
        fx, fy, cx, cy = p[0], p[0], p[1], p[2]
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    return K


# ----------------------------------------------------------------------
# database.db (SQLite) - keypoints & descripteurs SIFT
# ----------------------------------------------------------------------
def _blob_to_array(blob, dtype, cols):
    if blob is None:
        return np.zeros((0, cols), dtype=dtype)
    arr = np.frombuffer(blob, dtype=dtype)
    return arr.reshape(-1, cols)


def read_keypoints_and_descriptors(db_path, image_id):
    """
    Lit les keypoints (Nx2 ou Nx4 -> on garde x,y) et descripteurs (Nx128, uint8)
    d'une image donnee depuis database.db (schema standard COLMAP).
    Retourne (keypoints_xy (N,2) float32, descriptors (N,128) uint8) ou (None, None)
    si l'image n'existe pas dans la base.
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT rows, cols, data FROM keypoints WHERE image_id = ?", (image_id,))
    row = cur.fetchone()
    if row is None:
        conn.close()
        return None, None
    kp_rows, kp_cols, kp_blob = row
    keypoints = _blob_to_array(kp_blob, np.float32, kp_cols)
    keypoints_xy = keypoints[:, :2].copy()

    cur.execute("SELECT rows, cols, data FROM descriptors WHERE image_id = ?", (image_id,))
    row = cur.fetchone()
    conn.close()
    if row is None:
        return keypoints_xy, None
    d_rows, d_cols, d_blob = row
    descriptors = _blob_to_array(d_blob, np.uint8, d_cols)
    return keypoints_xy, descriptors


def list_images_in_database(db_path):
    """Retourne un dict image_id -> name depuis la table `images` du database.db."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT image_id, name FROM images")
    rows = cur.fetchall()
    conn.close()
    return {r[0]: r[1] for r in rows}
