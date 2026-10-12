package com.thesis.arlocalizer.model

import kotlin.math.*

/**
 * Quaternion unitaire q = qw + qx*i + qy*j + qz*k
 * Cf. section 4.4 du manuscrit (representation d'orientation).
 */
data class Quaternion(val qw: Double, val qx: Double, val qy: Double, val qz: Double) {

    fun normalized(): Quaternion {
        val n = sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
        return if (n < 1e-12) Quaternion(1.0, 0.0, 0.0, 0.0) else Quaternion(qw / n, qx / n, qy / n, qz / n)
    }

    /** Produit scalaire dans R^4, utilise pour la perte d'orientation (Eq. loss_ori) et pour l'angle geodesique. */
    fun dot(other: Quaternion): Double = qw * other.qw + qx * other.qx + qy * other.qy + qz * other.qz

    /** Erreur angulaire en degres entre deux quaternions (equivalence q ~ -q). */
    fun angularErrorDeg(reference: Quaternion): Double {
        val d = abs(this.normalized().dot(reference.normalized())).coerceIn(-1.0, 1.0)
        return Math.toDegrees(2.0 * acos(d))
    }

    /** Conversion quaternion -> angles d'Euler ZYX en degres (lacet, tangage, roulis), pour l'export CSV. */
    fun toEulerZYXDeg(): Triple<Double, Double, Double> {
        val q = normalized()
        val roll = atan2(2 * (q.qw * q.qx + q.qy * q.qz), 1 - 2 * (q.qx * q.qx + q.qy * q.qy))
        val sinp = 2 * (q.qw * q.qy - q.qz * q.qx)
        val pitch = if (abs(sinp) >= 1) (PI / 2) * sign(sinp) else asin(sinp)
        val yaw = atan2(2 * (q.qw * q.qz + q.qx * q.qy), 1 - 2 * (q.qy * q.qy + q.qz * q.qz))
        return Triple(Math.toDegrees(yaw), Math.toDegrees(pitch), Math.toDegrees(roll))
    }

    /** Conversion en matrice de rotation 3x3 (row-major, 9 doubles). */
    fun toRotationMatrix(): DoubleArray {
        val q = normalized()
        val (w, x, y, z) = listOf(q.qw, q.qx, q.qy, q.qz)
        return doubleArrayOf(
            1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
            2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
            2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)
        )
    }

    companion object {
        /**
         * Conversion Euler (lacet psi, tangage theta, roulis phi_r) -> quaternion,
         * convention ZYX identique a celle utilisee par l'application Android de collecte
         * (Eqs. 4.qw - 4.qz du manuscrit).
         */
        fun fromEulerZYX(yaw: Double, pitch: Double, roll: Double): Quaternion {
            val cy = cos(yaw / 2); val sy = sin(yaw / 2)
            val cp = cos(pitch / 2); val sp = sin(pitch / 2)
            val cr = cos(roll / 2); val sr = sin(roll / 2)
            val qw = cr * cp * cy + sr * sp * sy
            val qx = sr * cp * cy - cr * sp * sy
            val qy = cr * sp * cy + sr * cp * sy
            val qz = cr * cp * sy - sr * sp * cy
            return Quaternion(qw, qx, qy, qz).normalized()
        }

        /**
         * Reconstruction d'orientation Android (formule exacte issue des runs precedents) :
         * R = Rz(-azimuth) . Rx(-pitch) . Ry(roll)
         * azimuth, pitch, roll en radians, capteurs Android (SensorManager.getOrientation).
         */
        fun fromAndroidOrientation(azimuthRad: Double, pitchRad: Double, rollRad: Double): Quaternion {
            fun rotZ(a: Double) = arrayOf(
                doubleArrayOf(cos(a), -sin(a), 0.0),
                doubleArrayOf(sin(a), cos(a), 0.0),
                doubleArrayOf(0.0, 0.0, 1.0)
            )
            fun rotX(a: Double) = arrayOf(
                doubleArrayOf(1.0, 0.0, 0.0),
                doubleArrayOf(0.0, cos(a), -sin(a)),
                doubleArrayOf(0.0, sin(a), cos(a))
            )
            fun rotY(a: Double) = arrayOf(
                doubleArrayOf(cos(a), 0.0, sin(a)),
                doubleArrayOf(0.0, 1.0, 0.0),
                doubleArrayOf(-sin(a), 0.0, cos(a))
            )
            fun matmul(a: Array<DoubleArray>, b: Array<DoubleArray>): Array<DoubleArray> {
                val r = Array(3) { DoubleArray(3) }
                for (i in 0..2) for (j in 0..2) for (k in 0..2) r[i][j] += a[i][k] * b[k][j]
                return r
            }
            val rz = rotZ(-azimuthRad)
            val rx = rotX(-pitchRad)
            val ry = rotY(rollRad)
            val r = matmul(matmul(rz, rx), ry)
            return fromRotationMatrix(r)
        }

        private fun fromRotationMatrix(m: Array<DoubleArray>): Quaternion {
            val trace = m[0][0] + m[1][1] + m[2][2]
            return if (trace > 0) {
                val s = 0.5 / sqrt(trace + 1.0)
                Quaternion(
                    0.25 / s,
                    (m[2][1] - m[1][2]) * s,
                    (m[0][2] - m[2][0]) * s,
                    (m[1][0] - m[0][1]) * s
                ).normalized()
            } else {
                // Cas degeneres omis pour la brievete (rares en usage pieton normal)
                val qw = sqrt(1.0 + trace) / 2.0
                Quaternion(qw, (m[2][1] - m[1][2]) / (4 * qw), (m[0][2] - m[2][0]) / (4 * qw), (m[1][0] - m[0][1]) / (4 * qw)).normalized()
            }
        }
    }
}

private operator fun <T> List<T>.component4(): T = this[3]

/** Pose geographique : position WGS-84 + orientation. Sortie native du Niveau 1 (APR). */
data class GeoPose(
    val latitude: Double,
    val longitude: Double,
    val orientation: Quaternion,
    /** Niveau de la cascade qui a produit cette pose : 1 (APR), 2 (YOLO+SIG) ou 3 (PG-PnP). */
    val level: Int,
    /** Ecart-type de position en metres, utilise comme prior par les niveaux suivants et par PoseAnchor. */
    val sigmaPosMeters: Double,
    val timestampMs: Long = System.currentTimeMillis()
)

/** Pose rigide SE(3) dans un repere metrique local (ENU), utilisee par PG-PnP et PoseAnchor. */
data class Pose3D(
    val rotation: DoubleArray, // 3x3 row-major
    val translation: DoubleArray // (x,y,z) en metres
) {
    /**
     * Convention utilisee dans PoseAnchorManager : Pose3D represente la pose de la camera EXPRIMEE
     * dans un repere cible (transform local->cible), i.e. X_cible = R * X_local + t.
     * (A distinguer de la convention "extrinseques" utilisee par PgPnpEngine, monde->camera.)
     */

    /** Inverse de la transformation : X_local = R^T * (X_cible - t). */
    fun inverse(): Pose3D {
        val rt = doubleArrayOf(rotation[0], rotation[3], rotation[6], rotation[1], rotation[4], rotation[7], rotation[2], rotation[5], rotation[8])
        val t = doubleArrayOf(
            -(rt[0] * translation[0] + rt[1] * translation[1] + rt[2] * translation[2]),
            -(rt[3] * translation[0] + rt[4] * translation[1] + rt[5] * translation[2]),
            -(rt[6] * translation[0] + rt[7] * translation[1] + rt[8] * translation[2])
        )
        return Pose3D(rt, t)
    }

    /** Composition this ∘ other : applique d'abord `other` (local->intermediaire), puis `this` (intermediaire->cible). */
    fun compose(other: Pose3D): Pose3D {
        val r = DoubleArray(9)
        for (i in 0..2) for (j in 0..2) {
            var s = 0.0
            for (k in 0..2) s += rotation[i * 3 + k] * other.rotation[k * 3 + j]
            r[i * 3 + j] = s
        }
        val t = DoubleArray(3)
        for (i in 0..2) {
            var s = translation[i]
            for (k in 0..2) s += rotation[i * 3 + k] * other.translation[k]
            t[i] = s
        }
        return Pose3D(r, t)
    }

    /** Applique la transformation a un point : X_cible = R * X_local + t. */
    fun apply(point: DoubleArray): DoubleArray = doubleArrayOf(
        rotation[0] * point[0] + rotation[1] * point[1] + rotation[2] * point[2] + translation[0],
        rotation[3] * point[0] + rotation[4] * point[1] + rotation[5] * point[2] + translation[1],
        rotation[6] * point[0] + rotation[7] * point[1] + rotation[8] * point[2] + translation[2]
    )

    companion object {
        fun identity() = Pose3D(
            doubleArrayOf(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
            doubleArrayOf(0.0, 0.0, 0.0)
        )
    }
}
