package com.thesis.arlocalizer.level3_pgpnp

import com.thesis.arlocalizer.model.Correspondence2D3D
import com.thesis.arlocalizer.model.Pose3D
import org.opencv.calib3d.Calib3d
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfDouble
import org.opencv.core.MatOfPoint2f
import org.opencv.core.MatOfPoint3f
import org.opencv.core.Point
import org.opencv.core.Point3
import android.util.Log

/**
 * Solveur PnP via OpenCV (Calib3d.solvePnP, flag SOLVEPNP_EPNP) -- EXACTEMENT le meme algorithme
 * que cv2.solvePnP(..., flags=cv2.SOLVEPNP_EPNP) utilise dans le pipeline Python de reference
 * (level3_colmap_pnp.py). Remplace l'implementation Kotlin pure (Jacobi/EPnP a la main, session
 * precedente) : l'app embarque deja OpenCV (features2d/SIFT, cf. OpenCvSiftExtractor.kt), et
 * l'AAR OpenCV complet contient tout le module calib3d sans dependance Gradle supplementaire --
 * aucune raison de reimplementer a la main un algorithme deja present, teste et optimise (natif C++).
 *
 * Ce solveur est GLOBAL (aucune pose initiale requise), contrairement a l'ancien raffinement
 * Gauss-Newton local partant de la rotation APR, qui restait piege des lors que celle-ci etait
 * trop imprecise (cause racine confirmee par le diagnostic [DIAG-ROTATION] de la session
 * precedente : aucune convention fixe -- roll, transposee, identite -- ne dominait
 * systematiquement, signe d'un bruit reel sur l'orientation).
 */
object EPnPSolver {

    private const val TAG = "Level3Refiner"

    /**
     * @param correspondences correspondances 2D-3D (point2D en pixels, point3D en repere monde/ENU)
     * @param fx,fy,cx,cy intrinseques camera reelles (pas de distorsion geree ici, coherent avec
     *        PoseMath.project qui n'en applique pas non plus)
     * @return la pose (R,t extrinseque : X_cam = R*X_world + t), ou null si <4 points ou echec
     *         de convergence d'OpenCV (systeme degenere : points colineaires/coplanaires, etc.)
     */
    fun solve(correspondences: List<Correspondence2D3D>, fx: Double, fy: Double, cx: Double, cy: Double): Pose3D? {
        val n = correspondences.size
        if (n < 4) return null

        return try {
            val objectPoints = MatOfPoint3f().apply {
                fromList(correspondences.map { Point3(it.point3D[0], it.point3D[1], it.point3D[2]) })
            }
            val imagePoints = MatOfPoint2f().apply {
                fromList(correspondences.map { Point(it.point2D[0], it.point2D[1]) })
            }

            val cameraMatrix = Mat(3, 3, CvType.CV_64F).apply {
                put(0, 0, fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0)
            }
            val distCoeffs = MatOfDouble() // vide = pas de distorsion (idem PoseMath.project)

            val rvec = Mat()
            val tvec = Mat()

            // SOLVEPNP_EPNP : fonctionne des 4 points, pas de pose initiale requise (solveur global)
            val success = Calib3d.solvePnP(
                objectPoints, imagePoints, cameraMatrix, distCoeffs,
                rvec, tvec, false, Calib3d.SOLVEPNP_EPNP
            )
            if (!success) return null

            val rotMat = Mat()
            Calib3d.Rodrigues(rvec, rotMat)

            val r = DoubleArray(9); rotMat.get(0, 0, r)   // lecture row-major, coherent avec PoseMath
            val t = DoubleArray(3); tvec.get(0, 0, t)

            objectPoints.release(); imagePoints.release()
            cameraMatrix.release(); rvec.release(); tvec.release(); rotMat.release()

            Pose3D(r, t)
        } catch (e: Throwable) {
            // Cause frequente : points quasi-colineaires/coplanaires dans l'echantillon RANSAC
            // (systeme mal conditionne) -- OpenCV leve parfois une CvException plutot que de
            // retourner false. On journalise et on renvoie null (le candidat est simplement rejete,
            // RANSAC continue avec le tirage suivant).
            Log.w(TAG, "[EPnP] echec solvePnP sur cet echantillon (probablement degenere) : ${e.message}")
            null
        }
    }
}