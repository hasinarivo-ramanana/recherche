package com.thesis.arlocalizer.util

import kotlin.math.*

/**
 * Constantes WGS-84 et conversions geographiques utilisees dans tout le pipeline :
 * - Haversine (distance geodesique, Chapitres 1 et 2)
 * - ECEF / ENU (repere local metrique, Chapitre 4 "PoseAnchor")
 */
object GeoUtils {

    const val EARTH_RADIUS_M = 6_371_000.0
    private const val WGS84_A = 6_378_137.0          // demi grand axe (m)
    private const val WGS84_E2 = 0.00669438           // excentricite^2

    /** Distance de Haversine entre deux points WGS-84 (m). Eq. distance geodesique du manuscrit. */
    fun haversine(lat1: Double, lon1: Double, lat2: Double, lon2: Double): Double {
        val p1 = Math.toRadians(lat1); val p2 = Math.toRadians(lat2)
        val dPhi = Math.toRadians(lat2 - lat1)
        val dLambda = Math.toRadians(lon2 - lon1)
        val a = sin(dPhi / 2).pow(2) + cos(p1) * cos(p2) * sin(dLambda / 2).pow(2)
        return 2 * EARTH_RADIUS_M * asin(sqrt(a))
    }

    /** Azimut geographique (radians, 0 = Nord, sens horaire) du point 1 vers le point 2 (approx. plane, courtes distances). */
    fun azimuth(lat1: Double, lon1: Double, lat2: Double, lon2: Double): Double {
        val dLat = lat2 - lat1
        val dLon = lon2 - lon1
        // atan2(dLon, dLat) donne un azimut compas approx. pour de petites distances (< qq km)
        return atan2(dLon, dLat)
    }

    /** WGS-84 (lat, lon, h) -> ECEF (X, Y, Z) en metres. Eq. 4.ecef du manuscrit. */
    fun geodeticToEcef(latDeg: Double, lonDeg: Double, h: Double = 0.0): DoubleArray {
        val phi = Math.toRadians(latDeg)
        val lambda = Math.toRadians(lonDeg)
        val n = WGS84_A / sqrt(1 - WGS84_E2 * sin(phi).pow(2))
        val x = (n + h) * cos(phi) * cos(lambda)
        val y = (n + h) * cos(phi) * sin(lambda)
        val z = (n * (1 - WGS84_E2) + h) * sin(phi)
        return doubleArrayOf(x, y, z)
    }

    /**
     * Matrice de rotation ECEF -> ENU (East-North-Up) locale, centree sur (lat0, lon0).
     * Retourne un tableau 3x3 row-major. Eq. 4.R_ENU du manuscrit.
     */
    fun ecefToEnuRotation(lat0Deg: Double, lon0Deg: Double): DoubleArray {
        val phi = Math.toRadians(lat0Deg)
        val lambda = Math.toRadians(lon0Deg)
        return doubleArrayOf(
            -sin(lambda), cos(lambda), 0.0,
            -sin(phi) * cos(lambda), -sin(phi) * sin(lambda), cos(phi),
            cos(phi) * cos(lambda), cos(phi) * sin(lambda), sin(phi)
        )
    }

    /** Convertit un point WGS-84 en coordonnees ENU locales (m) par rapport a une origine (lat0, lon0). */
    fun geodeticToEnu(latDeg: Double, lonDeg: Double, h: Double, lat0Deg: Double, lon0Deg: Double, h0: Double): DoubleArray {
        val ecef = geodeticToEcef(latDeg, lonDeg, h)
        val ecef0 = geodeticToEcef(lat0Deg, lon0Deg, h0)
        val d = doubleArrayOf(ecef[0] - ecef0[0], ecef[1] - ecef0[1], ecef[2] - ecef0[2])
        val r = ecefToEnuRotation(lat0Deg, lon0Deg)
        return matVec(r, d)
    }

    /** Approximation plane locale (mètres) autour d'une origine — suffisante à l'échelle d'un campus (< 1 km). */
    fun geodeticToLocalMeters(latDeg: Double, lonDeg: Double, lat0Deg: Double, lon0Deg: Double): DoubleArray {
        val dLat = Math.toRadians(latDeg - lat0Deg)
        val dLon = Math.toRadians(lonDeg - lon0Deg)
        val north = dLat * EARTH_RADIUS_M
        val east = dLon * EARTH_RADIUS_M * cos(Math.toRadians(lat0Deg))
        return doubleArrayOf(east, north)
    }

    fun matVec(m: DoubleArray, v: DoubleArray): DoubleArray {
        require(m.size == 9 && v.size == 3)
        return doubleArrayOf(
            m[0] * v[0] + m[1] * v[1] + m[2] * v[2],
            m[3] * v[0] + m[4] * v[1] + m[5] * v[2],
            m[6] * v[0] + m[7] * v[1] + m[8] * v[2]
        )
    }

    fun matTranspose(m: DoubleArray): DoubleArray = doubleArrayOf(
        m[0], m[3], m[6],
        m[1], m[4], m[7],
        m[2], m[5], m[8]
    )

    fun matMul(a: DoubleArray, b: DoubleArray): DoubleArray {
        val r = DoubleArray(9)
        for (i in 0..2) for (j in 0..2) {
            var s = 0.0
            for (k in 0..2) s += a[i * 3 + k] * b[k * 3 + j]
            r[i * 3 + j] = s
        }
        return r
    }
}
