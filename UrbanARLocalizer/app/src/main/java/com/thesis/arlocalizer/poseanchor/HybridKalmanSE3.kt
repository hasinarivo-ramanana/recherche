package com.thesis.arlocalizer.poseanchor

import com.thesis.arlocalizer.level3_pgpnp.PoseMath
import com.thesis.arlocalizer.model.Pose3D
import kotlin.math.*

/**
 * Filtre de Kalman simplifie sur SE(3) pour la correction de derive VIO (mecanisme PoseAnchor,
 * Section "Correction de la derive VIO par recalage periodique", Eqs. xi_update_simple - P_update).
 *
 * Etat : transformation courante T_n = (R_n, t_n) et covariance scalaire-diagonale P_n (6x6,
 * 3 termes position + 3 termes orientation). error(.) mesure l'ecart position+rotation entre deux
 * poses ; corr(.) applique une petite correction (perturbation exponentielle) a la transformation courante.
 */
class HybridKalmanSE3(initialPose: Pose3D, initialUncertaintyM: Double = 50.0) {

    private var pose: Pose3D = initialPose
    // Covariance diagonale : [pos_x, pos_y, pos_z, rot_x, rot_y, rot_z]
    private var covarianceDiag: DoubleArray = DoubleArray(6) { if (it < 3) initialUncertaintyM.pow(2) else 0.5 }

    fun currentPose(): Pose3D = pose
    fun currentCovariance(): DoubleArray = covarianceDiag.copyOf()

    /**
     * Met a jour l'etat avec une nouvelle mesure de pose (issue de la cascade de localisation, typiquement
     * le Niveau 3 / PG-PnP) et sa covariance associee (mesureSigmaPosM, mesureSigmaRotRad).
     */
    fun update(measuredPose: Pose3D, measureSigmaPosM: Double, measureSigmaRotRad: Double = 0.1) {
        val measurementCovariance = DoubleArray(6) { if (it < 3) measureSigmaPosM.pow(2) else measureSigmaRotRad.pow(2) }

        // xi_n = error(T_{n-1}, T_mesure) : erreur de position (3) + erreur d'orientation approx. (axis-angle, 3)
        val xi = errorVector(pose, measuredPose)

        // Gain de Kalman scalaire par composante : K = P / (P + R)
        val gain = DoubleArray(6) { i -> covarianceDiag[i] / (covarianceDiag[i] + measurementCovariance[i]) }

        // Correction ponderee : T_n = T_{n-1} o corr(K * xi)
        val correction = DoubleArray(6) { i -> gain[i] * xi[i] }
        val newTranslation = doubleArrayOf(
            pose.translation[0] + correction[0],
            pose.translation[1] + correction[1],
            pose.translation[2] + correction[2]
        )
        val newRotation = PoseMath.composeRotation(pose.rotation, doubleArrayOf(correction[3], correction[4], correction[5]))
        pose = Pose3D(newRotation, newTranslation)

        // P_n = (I - K) * P_{n-1}
        covarianceDiag = DoubleArray(6) { i -> (1 - gain[i]) * covarianceDiag[i] }
    }

    /**
     * Propagation entre deux mesures (odometrie VIO d'ARCore) : la transformation courante suit le
     * deplacement relatif rapporte par ARCore, avec une legere augmentation de l'incertitude (derive).
     */
    fun predict(deltaPoseFromVio: Pose3D, driftGrowthFactor: Double = 1.02) {
        val newTranslation = doubleArrayOf(
            pose.translation[0] + deltaPoseFromVio.translation[0],
            pose.translation[1] + deltaPoseFromVio.translation[1],
            pose.translation[2] + deltaPoseFromVio.translation[2]
        )
        val newRotation = PoseMath.matMul3(pose.rotation, deltaPoseFromVio.rotation)
        pose = Pose3D(newRotation, newTranslation)
        covarianceDiag = DoubleArray(6) { i -> covarianceDiag[i] * driftGrowthFactor }
    }

    private fun errorVector(a: Pose3D, b: Pose3D): DoubleArray {
        val posErr = doubleArrayOf(
            b.translation[0] - a.translation[0],
            b.translation[1] - a.translation[1],
            b.translation[2] - a.translation[2]
        )
        // Erreur d'orientation approximee : log(R_a^T . R_b) via Rodrigues inverse (petit angle)
        val rDiff = PoseMath.matMul3(transpose3(a.rotation), b.rotation)
        val rotErr = logRotation(rDiff)
        return doubleArrayOf(posErr[0], posErr[1], posErr[2], rotErr[0], rotErr[1], rotErr[2])
    }

    private fun transpose3(m: DoubleArray) = doubleArrayOf(m[0], m[3], m[6], m[1], m[4], m[7], m[2], m[5], m[8])

    private fun logRotation(r: DoubleArray): DoubleArray {
        val trace = r[0] + r[4] + r[8]
        val cosTheta = ((trace - 1) / 2).coerceIn(-1.0, 1.0)
        val theta = acos(cosTheta)
        if (theta < 1e-8) return doubleArrayOf(0.0, 0.0, 0.0)
        val factor = theta / (2 * sin(theta))
        return doubleArrayOf(
            factor * (r[7] - r[5]),
            factor * (r[2] - r[6]),
            factor * (r[3] - r[1])
        )
    }
}
