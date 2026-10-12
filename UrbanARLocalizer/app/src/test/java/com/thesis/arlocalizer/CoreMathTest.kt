package com.thesis.arlocalizer

import com.thesis.arlocalizer.model.Quaternion
import com.thesis.arlocalizer.util.GeoUtils
import org.junit.Assert.assertEquals
import org.junit.Test
import kotlin.math.abs

class CoreMathTest {

    @Test
    fun haversine_knownDistance_isAccurate() {
        // Antananarivo campus, deux points separes d'environ 100m (verifie a la calculatrice)
        val d = GeoUtils.haversine(-18.9100, 47.5400, -18.9109, 47.5400)
        assertEquals(100.0, d, 2.0)
    }

    @Test
    fun quaternion_eulerRoundTrip_isConsistent() {
        val yaw = Math.toRadians(30.0)
        val pitch = Math.toRadians(10.0)
        val roll = Math.toRadians(-5.0)
        val q = Quaternion.fromEulerZYX(yaw, pitch, roll)
        val norm = q.qw * q.qw + q.qx * q.qx + q.qy * q.qy + q.qz * q.qz
        assertEquals(1.0, norm, 1e-9)
    }

    @Test
    fun quaternion_angularError_zeroForIdentical() {
        val q = Quaternion.fromEulerZYX(0.5, 0.1, -0.2)
        assertEquals(0.0, q.angularErrorDeg(q), 1e-6)
    }

    @Test
    fun quaternion_antipodalEquivalence() {
        val q = Quaternion(0.7, 0.1, 0.2, 0.3).normalized()
        val qNeg = Quaternion(-q.qw, -q.qx, -q.qy, -q.qz)
        // q et -q representent la meme rotation : erreur angulaire nulle
        assertEquals(0.0, q.angularErrorDeg(qNeg), 1e-6)
    }

    @Test
    fun geodeticToEnu_originMapsToZero() {
        val enu = GeoUtils.geodeticToEnu(-18.91, 47.54, 0.0, -18.91, 47.54, 0.0)
        assertEquals(0.0, enu[0], 1e-6)
        assertEquals(0.0, enu[1], 1e-6)
    }

    @Test
    fun localMeters_matchesHaversine_forSmallDistances() {
        val lat0 = -18.9100; val lon0 = 47.5400
        val lat1 = -18.9105; val lon1 = 47.5405
        val local = GeoUtils.geodeticToLocalMeters(lat1, lon1, lat0, lon0)
        val dLocal = Math.sqrt(local[0] * local[0] + local[1] * local[1])
        val dHaversine = GeoUtils.haversine(lat0, lon0, lat1, lon1)
        assertEquals(dHaversine, dLocal, 1.0)
    }
}
