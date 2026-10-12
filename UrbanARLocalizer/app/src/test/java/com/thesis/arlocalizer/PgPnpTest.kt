package com.thesis.arlocalizer

import com.thesis.arlocalizer.level3_pgpnp.PgPnpEngine
import com.thesis.arlocalizer.level3_pgpnp.PoseMath
import com.thesis.arlocalizer.model.Correspondence2D3D
import com.thesis.arlocalizer.model.Pose3D
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class PgPnpTest {

    @Test
    fun rodrigues_zeroVector_isIdentity() {
        val r = PoseMath.rodrigues(doubleArrayOf(0.0, 0.0, 0.0))
        assertEquals(1.0, r[0], 1e-9); assertEquals(1.0, r[4], 1e-9); assertEquals(1.0, r[8], 1e-9)
        assertEquals(0.0, r[1], 1e-9)
    }

    @Test
    fun geographicFilter_removesDistantPoints() {
        val engine = PgPnpEngine(filterRadiusM = 60.0)
        val near = Correspondence2D3D(doubleArrayOf(100.0, 100.0), doubleArrayOf(10.0, 10.0, 0.0))
        val far = Correspondence2D3D(doubleArrayOf(200.0, 200.0), doubleArrayOf(500.0, 500.0, 0.0))
        val filtered = engine.geographicFilter(listOf(near, far), aprPositionEnu = doubleArrayOf(0.0, 0.0, 0.0))
        assertEquals(1, filtered.size)
        assertEquals(near, filtered[0])
    }

    @Test
    fun estimatePose_syntheticScene_convergesNearGroundTruth() {
        // Scene synthetique : camera a l'origine regardant droit devant (Z), points 3D devant elle.
        val fx = 1000.0; val fy = 1000.0; val cx = 320.0; val cy = 240.0
        val trueT = doubleArrayOf(0.3, -0.1, 0.0)
        val trueR = doubleArrayOf(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)

        val points3D = listOf(
            doubleArrayOf(-1.0, -1.0, 5.0), doubleArrayOf(1.0, -1.0, 5.0),
            doubleArrayOf(1.0, 1.0, 5.0), doubleArrayOf(-1.0, 1.0, 5.0),
            doubleArrayOf(0.0, 0.0, 6.0), doubleArrayOf(-0.5, 0.5, 4.0),
            doubleArrayOf(0.5, -0.5, 7.0), doubleArrayOf(2.0, 0.0, 5.0)
        )
        val correspondences = points3D.map { x ->
            val proj = PoseMath.project(fx, fy, cx, cy, trueR, trueT, x)
            Correspondence2D3D(proj, x)
        }

        val engine = PgPnpEngine(fx = fx, fy = fy, cx = cx, cy = cy, filterRadiusM = 1000.0, minInliers = 4, ransacIterations = 30)
        // Prior legerement decale (simule une erreur APR)
        val priorPose = Pose3D(trueR, doubleArrayOf(trueT[0] + 2.0, trueT[1] - 1.5, trueT[2]))

        val result = engine.estimatePose(correspondences, priorPose)
        assertTrue("La pose ne doit pas etre nulle", result.pose != null)
        val err = Math.sqrt(
            Math.pow(result.pose!!.translation[0] - trueT[0], 2.0) +
            Math.pow(result.pose.translation[1] - trueT[1], 2.0) +
            Math.pow(result.pose.translation[2] - trueT[2], 2.0)
        )
        assertTrue("Erreur de position trop grande: $err", err < 0.5)
    }
}
