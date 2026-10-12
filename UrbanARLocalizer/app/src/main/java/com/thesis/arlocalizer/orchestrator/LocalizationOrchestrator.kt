package com.thesis.arlocalizer.orchestrator

import android.graphics.Bitmap
import com.thesis.arlocalizer.level1_apr.AprEngine
import com.thesis.arlocalizer.level2_yolosig.Level2Refiner
import com.thesis.arlocalizer.level3_pgpnp.Level3Refiner
import com.thesis.arlocalizer.model.GeoPose
import com.thesis.arlocalizer.model.Pose3D
import com.thesis.arlocalizer.poseanchor.PoseAnchorManager
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

/**
 * Cascade complete de localisation (Fig. pipeline global, chapitre 1) :
 *
 *   Image requete
 *      -> Niveau 1 (APR)                          ~50 m,  toujours execute
 *      -> Niveau 2 (YOLO+SIG) si logos detectes    ~9-20 m
 *      -> Niveau 3 (PG-PnP) sinon, ou en complement <2 m
 *      -> PoseAnchor (recalage AR)
 *
 * Repli garanti a chaque etage : si un niveau echoue, la pose du niveau precedent est utilisee.
 */
class LocalizationOrchestrator(
    private val aprEngine: AprEngine,
    private val level2Refiner: Level2Refiner,
    private val level3Refiner: Level3Refiner,
    private val poseAnchorManager: PoseAnchorManager
) {

    data class CascadeResult(
        val finalPose: GeoPose,
        val level1Pose: GeoPose,
        val level2Pose: GeoPose?,
        val level3Pose: GeoPose?,
        // ---- Details additionnels par niveau, utilises pour le rapport texte detaille
        //      (ex. CascadeLogFormatter / bouton "Localiser ma position"). Purement informatif :
        //      n'affecte pas la selection de finalPose (toujours priorite 3 > 2 > 1 ci-dessous). ----
        val level1ElapsedMs: Long = 0L,
        val level2ElapsedMs: Long = 0L,
        val level2Detections: List<com.thesis.arlocalizer.model.YoloDetection> = emptyList(),
        val level2UsedFallback: Boolean = true,
        val level3ElapsedMs: Long = 0L,
        val level3InlierCount: Int? = null,
        val level3UsedFallback: Boolean? = null,
        val totalElapsedMs: Long = 0L
    )

    /**
     * Execute la cascade complete pour une image donnee et met a jour PoseAnchor.
     * @param cameraHeadingRad cap de la camera (yaw), necessaire au scoring du Niveau 2.
     * @param arCorePose pose de la camera dans le repere local ARCore au moment de la capture
     *                   (Frame.camera.pose), utilisee par PoseAnchor pour recomposer T_AR->WGS.
     *                   Si null (ARCore non encore disponible/tracking perdu), le recalage AR est
     *                   simplement ignore pour ce cycle (la cascade de localisation continue normalement).
     * @param focalPxOverride longueur focale reelle (px), ex. issue de Frame.camera.imageIntrinsics
     *                   d'ARCore ; remplace la valeur par defaut (a calibrer) utilisee par le Niveau 2.
     * @param cameraIntrinsics (fx, fy, cx, cy) reelles de l'image, ex. issues de
     *                   Frame.camera.imageIntrinsics d'ARCore ; CRITIQUE pour le Niveau 3 (PG-PnP) --
     *                   sans ces valeurs reelles, RANSAC ne validera jamais d'inlier (voir
     *                   PgPnpEngine.updateIntrinsics).
     */
    suspend fun localize(
        bitmap: Bitmap,
        cameraHeadingRad: Double,
        arCorePose: Pose3D? = null,
        focalPxOverride: Double? = null,
        cameraIntrinsics: DoubleArray? = null,
        rotationDegreesForClassifiers: Float = 90f
    ): CascadeResult =
        withContext(Dispatchers.Default) {
            val tGlobalStart = System.nanoTime()
            val bitmapForClassifiers = com.thesis.arlocalizer.util.ImageUtils.rotateBitmap(bitmap, rotationDegreesForClassifiers)

            // Niveau 1 : toujours execute, fournit la zone candidate (~100 m de rayon)
            // Utilise l'image tournee (orientation portrait, coherente avec l'entrainement APR).
            val t1Start = System.nanoTime()
            val level1Pose = aprEngine.estimatePose(bitmapForClassifiers)
            val level1ElapsedMs = (System.nanoTime() - t1Start) / 1_000_000

            // Niveau 2 : detection de logos/enseignes + requete SIG (meme image tournee, coherente
            // avec l'entrainement YOLO).
            val t2Start = System.nanoTime()
            val level2Result = level2Refiner.refine(bitmapForClassifiers, level1Pose, cameraHeadingRad, focalPxOverride)
            val level2ElapsedMs = (System.nanoTime() - t2Start) / 1_000_000
            val level2Pose = level2Result.refinedPose

            // Pose prior pour le Niveau 3 : la meilleure disponible jusqu'ici (Niveau 2 si succes, sinon Niveau 1)
            val priorForLevel3 = level2Pose ?: level1Pose

            // Niveau 3 : PG-PnP sur l'image BRUTE (non tournee) + intrinseques ARCore natives
            // correspondantes -- SIFT est invariant a la rotation, donc pas besoin de tourner ici,
            // et ca evite d'avoir a recalculer des intrinseques tournees (risque de bug geometrique).
            val t3Start = System.nanoTime()
            val level3Result = level3Refiner.refine(bitmap, priorForLevel3, cameraIntrinsics)
            val level3ElapsedMs = (System.nanoTime() - t3Start) / 1_000_000
            val level3Pose = level3Result.pose

            // Selection de la pose finale par ordre de priorite decroissante de precision : 3 > 2 > 1
            val finalPose = level3Pose ?: level2Pose ?: level1Pose

            // PoseAnchor : recalage du repere AR avec la meilleure pose disponible, si une pose ARCore
            // de reference est disponible pour ce cycle (cf. note ci-dessus).
            if (arCorePose != null) {
                poseAnchorManager.onNewLocalization(finalPose, arCorePose)
            }

            val totalElapsedMs = (System.nanoTime() - tGlobalStart) / 1_000_000

            CascadeResult(
                finalPose = finalPose,
                level1Pose = level1Pose,
                level2Pose = level2Pose,
                level3Pose = level3Pose,
                level1ElapsedMs = level1ElapsedMs,
                level2ElapsedMs = level2ElapsedMs,
                level2Detections = level2Result.detections,
                level2UsedFallback = level2Result.usedFallback,
                level3ElapsedMs = level3ElapsedMs,
                level3InlierCount = level3Result.inlierCount,
                level3UsedFallback = level3Result.fallback,
                totalElapsedMs = totalElapsedMs
            )
        }
}
