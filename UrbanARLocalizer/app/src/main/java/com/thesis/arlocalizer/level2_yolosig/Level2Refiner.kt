package com.thesis.arlocalizer.level2_yolosig

import android.graphics.Bitmap
import android.util.Log
import com.thesis.arlocalizer.model.GeoPose

/**
 * Orchestrateur du Niveau 2 complet : Detection YOLO -> Requete SIG -> Scoring -> Selection.
 * Correspond au diagramme "Architecture du pipeline d'affinement" (Fig. pipeline_affinement).
 *
 * Si aucun logo/enseigne n'est detecte (branche "NON" du diagramme de decision du chapitre 1,
 * "Des logos ou enseignes sont-ils detectes ?"), le systeme retombe sur la pose APR : la
 * "technique nuage de points" (Niveau 3 direct) prend le relais dans l'orchestrateur global.
 *
 * DIAGNOSTIC : chaque cause de repli (aucune detection, aucun candidat SIG, scoring vide) est
 * maintenant loggee explicitement (tag "Level2Refiner") -- avant, refine() retournait `null`
 * silencieusement dans les 3 cas, rendant impossible de savoir LEQUEL se produisait sans ca.
 */
class Level2Refiner(
    private val yoloDetector: YoloDetector,
    private val sigRepository: SigRepository,
    private val scorer: MultiCriteriaScorer = MultiCriteriaScorer(),
    private val focalPx: Double = 1400.0 // a calibrer depuis les metadonnees EXIF / Camera2 CameraCharacteristics
) {
    companion object {
        private const val TAG = "Level2Refiner"
    }

    data class Level2Result(
        val refinedPose: GeoPose?,
        val detections: List<com.thesis.arlocalizer.model.YoloDetection>,
        val usedFallback: Boolean
    )

    /**
     * @param aprPose pose issue du Niveau 1
     * @param cameraHeadingRad cap de la camera (yaw), en radians, Nord = 0
     * @param focalPxOverride longueur focale reelle (px), ex. issue de Camera.imageIntrinsics d'ARCore ;
     *                        si non fournie, utilise la valeur par defaut (a calibrer) du constructeur.
     */
    fun refine(bitmap: Bitmap, aprPose: GeoPose, cameraHeadingRad: Double, focalPxOverride: Double? = null): Level2Result {
        val effectiveFocalPx = focalPxOverride ?: focalPx
        val detections = yoloDetector.detect(bitmap)

        if (detections.isEmpty()) {
            Log.d(TAG, "ECHEC : aucune detection YOLO dans cette image (aucun logo/enseigne visible)")
            return Level2Result(refinedPose = null, detections = emptyList(), usedFallback = true)
        }
        Log.d(TAG, "${detections.size} detection(s) YOLO : ${detections.map { "${it.classe}(${String.format("%.2f", it.confidence)})" }}")

        val primary = detections.maxByOrNull { it.confidence }!!
        val candidates = sigRepository.queryCandidates(aprPose.latitude, aprPose.longitude, primary.classe)

        if (candidates.isEmpty()) {
            Log.d(TAG, "ECHEC : aucun candidat SIG de classe '${primary.classe}' pres de " +
                    "(${aprPose.latitude}, ${aprPose.longitude}) dans un rayon de ${SigRepository.DEFAULT_RADIUS_M} m. " +
                    "Verifiez sig_data.geojson (classe presente ? rayon suffisant ? position APR coherente avec la zone couverte ?)")
            return Level2Result(refinedPose = null, detections = detections, usedFallback = true)
        }
        Log.d(TAG, "${candidates.size} candidat(s) SIG trouve(s) pour la classe '${primary.classe}'")

        val best = scorer.selectBestCandidate(
            aprPose = aprPose,
            primaryDetection = primary,
            allDetections = detections,
            candidates = candidates,
            focalPx = effectiveFocalPx,
            imageWidth = bitmap.width.toFloat(),
            cameraHeadingRad = cameraHeadingRad
        )
        if (best == null) {
            Log.d(TAG, "ECHEC : scoring multi-criteres n'a retenu aucun candidat (cas normalement impossible si candidates.isNotEmpty())")
            return Level2Result(refinedPose = null, detections = detections, usedFallback = true)
        }
        Log.d(TAG, "SUCCES : candidat retenu '${best.landmark.id}' (score=${String.format("%.3f", best.total)})")

        val refined = GeoPose(
            latitude = best.landmark.latitude,
            longitude = best.landmark.longitude,
            orientation = aprPose.orientation, // l'orientation n'est pas affinee a ce niveau
            level = 2,
            sigmaPosMeters = 9.12 // erreur mediane mesuree, Tableau "resultats"
        )
        return Level2Result(refinedPose = refined, detections = detections, usedFallback = false)
    }
}
