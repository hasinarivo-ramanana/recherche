package com.thesis.arlocalizer.level2_yolosig

import com.thesis.arlocalizer.model.GeoPose
import com.thesis.arlocalizer.model.ScoredCandidate
import com.thesis.arlocalizer.model.SigLandmark
import com.thesis.arlocalizer.model.YoloDetection
import com.thesis.arlocalizer.util.GeoUtils
import kotlin.math.*

/**
 * Etape 3 (contribution principale du Chapitre 2) : score de vraisemblance multi-criteres
 * S_total(c_k) = w1*s_taille + w2*s_angle + w3*s_rapport + w4*s_cote
 * Poids empiriques (Tableau "poids") : w1=0.30, w2=0.35, w3=0.20, w4=0.15.
 */
class MultiCriteriaScorer(
    private val w1SizeTaille: Double = 0.30,
    private val w2Angle: Double = 0.35,
    private val w3Rapport: Double = 0.20,
    private val w4Cote: Double = 0.15,
    private val sigmaDistance: Double = 15.0,   // sigma_d, tolerance sur la distance estimee (m)
    private val sigmaAngleDeg: Double = 20.0,   // sigma_alpha, tolerance angulaire (deg)
    private val sigmaRapport: Double = 0.3      // sigma_R, tolerance sur le rapport inter-logos
) {

    /**
     * Contrainte 1 : taille apparente du logo -> distance estimee par projection perspective (Eq. dist_estimee),
     * comparee a la distance geodesique candidat<->position APR (Eq. score_taille).
     */
    fun scoreTaille(hPx: Float, focalPx: Double, hReelM: Double, dSigM: Double): Double {
        val dEstimee = (focalPx * hReelM) / max(hPx.toDouble(), 1.0)
        val erreur = abs(dEstimee - dSigM)
        return exp(-(erreur * erreur) / (2 * sigmaDistance * sigmaDistance))
    }

    /**
     * Contrainte 2 : position angulaire horizontale (azimut). Eqs. alpha_image / alpha_SIG / score_angle.
     * cameraHeadingRad : cap (yaw) de la camera en radians, issu de l'orientation APR ou de la boussole.
     */
    fun scoreAngle(
        detectionCenterU: Float, imageCenterU: Double, focalPx: Double, cameraHeadingRad: Double,
        aprLat: Double, aprLon: Double, candidate: SigLandmark
    ): Double {
        val alphaImageLocal = atan2((detectionCenterU - imageCenterU), focalPx)
        val alphaImageAbs = cameraHeadingRad + alphaImageLocal
        val alphaSig = GeoUtils.azimuth(aprLat, aprLon, candidate.latitude, candidate.longitude)
        val diff = angleDiff(alphaImageAbs, alphaSig)
        val sigmaRad = Math.toRadians(sigmaAngleDeg)
        return exp(-(diff * diff) / (2 * sigmaRad * sigmaRad))
    }

    private fun angleDiff(a: Double, b: Double): Double {
        var d = (a - b) % (2 * PI)
        if (d > PI) d -= 2 * PI
        if (d < -PI) d += 2 * PI
        return d
    }

    /**
     * Contrainte 3 : rapport de distances entre logos co-detectes (invariant d'echelle). Eqs. rapport_image/rapport_sig.
     * Necessite >= 2 detections simultanees ; retourne un score neutre (1.0) sinon.
     */
    fun scoreRapport(
        detections: List<YoloDetection>, primaryDetection: YoloDetection,
        candidatesByClass: Map<String, SigLandmark>
    ): Double {
        if (detections.size < 3) return 1.0 // score neutre, contrainte non discriminante
        val others = detections.filter { it != primaryDetection }.take(2)
        if (others.size < 2) return 1.0

        val d12Px = distancePx(primaryDetection, others[0])
        val d13Px = distancePx(primaryDetection, others[1])
        if (d13Px < 1e-6) return 1.0
        val rImage = d12Px / d13Px

        val c1 = candidatesByClass[primaryDetection.classe] ?: return 1.0
        val c2 = candidatesByClass[others[0].classe] ?: return 1.0
        val c3 = candidatesByClass[others[1].classe] ?: return 1.0
        val d12Geo = GeoUtils.haversine(c1.latitude, c1.longitude, c2.latitude, c2.longitude)
        val d13Geo = GeoUtils.haversine(c1.latitude, c1.longitude, c3.latitude, c3.longitude)
        if (d13Geo < 1e-6) return 1.0
        val rSig = d12Geo / d13Geo

        val erreur = rImage - rSig
        return exp(-(erreur * erreur) / (2 * sigmaRapport * sigmaRapport))
    }

    private fun distancePx(a: YoloDetection, b: YoloDetection): Double {
        val dx = (a.centerU - b.centerU).toDouble()
        val dy = (a.centerV - b.centerV).toDouble()
        return sqrt(dx * dx + dy * dy)
    }

    /**
     * Contrainte 4 : cote de la rue (gauche/droite). Eqs. produit_croise / score_cote.
     */
    fun scoreCote(
        detectionCenterU: Float, imageWidth: Float,
        aprLat: Double, aprLon: Double, cameraHeadingRad: Double, candidate: SigLandmark
    ): Double {
        val coteImage = if (detectionCenterU < imageWidth / 2f) "gauche" else "droite"

        val vxCap = cos(cameraHeadingRad); val vyCap = sin(cameraHeadingRad)
        val local = GeoUtils.geodeticToLocalMeters(candidate.latitude, candidate.longitude, aprLat, aprLon)
        val norme = sqrt(local[0] * local[0] + local[1] * local[1])
        if (norme < 1e-9) return 0.5
        val vxCand = local[0] / norme; val vyCand = local[1] / norme

        val produitCroise = vxCap * vyCand - vyCap * vxCand
        val coteSig = if (produitCroise > 0) "gauche" else "droite"

        return if (coteImage == coteSig) 1.0 else 0.1
    }

    /**
     * Score total S_total et selection du meilleur candidat (Algorithme "Selection du candidat optimal").
     */
    fun selectBestCandidate(
        aprPose: GeoPose,
        primaryDetection: YoloDetection,
        allDetections: List<YoloDetection>,
        candidates: List<SigLandmark>,
        focalPx: Double,
        imageWidth: Float,
        cameraHeadingRad: Double
    ): ScoredCandidate? {
        if (candidates.isEmpty()) return null

        val candidatesByClass = allDetections.mapNotNull { det ->
            candidates.filter { it.classe == det.classe }
                .minByOrNull { GeoUtils.haversine(aprPose.latitude, aprPose.longitude, it.latitude, it.longitude) }
                ?.let { det.classe to it }
        }.toMap()

        val scored = candidates.map { c ->
            val dSig = GeoUtils.haversine(aprPose.latitude, aprPose.longitude, c.latitude, c.longitude)
            val s1 = scoreTaille(primaryDetection.heightPx, focalPx, c.hauteurReelleM, dSig)
            val s2 = scoreAngle(primaryDetection.centerU, imageWidth / 2.0, focalPx, cameraHeadingRad, aprPose.latitude, aprPose.longitude, c)
            val s3 = scoreRapport(allDetections, primaryDetection, candidatesByClass)
            val s4 = scoreCote(primaryDetection.centerU, imageWidth, aprPose.latitude, aprPose.longitude, cameraHeadingRad, c)
            val total = w1SizeTaille * s1 + w2Angle * s2 + w3Rapport * s3 + w4Cote * s4
            ScoredCandidate(c, s1, s2, s3, s4, total)
        }
        return scored.maxByOrNull { it.total }
    }
}
