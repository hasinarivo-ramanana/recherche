package com.thesis.arlocalizer.level3_pgpnp

import android.graphics.Bitmap
import android.util.Log
import com.thesis.arlocalizer.model.*
import com.thesis.arlocalizer.util.GeoUtils

/**
 * Orchestrateur du Niveau 3 : convertit la pose APR/Niveau-2 en repere ENU local, filtre le nuage
 * COLMAP geographiquement, construit les correspondances SIFT 2D-3D et lance le PG-PnP.
 * Repli garanti sur la pose du Niveau 2 en cas d'echec (couverture COLMAP insuffisante).
 *
 * DIAGNOSTIC : chaque cause de repli est desormais loggee explicitement (tag "Level3Refiner"),
 * y compris une exception levee pendant l'extraction SIFT (ex. OpenCV non initialise/non lie au
 * projet), qui AVANT ce correctif se propageait de facon non geree -- pouvant, selon le contexte
 * d'appel, interrompre silencieusement la boucle de localisation sans aucune trace exploitable.
 */
class Level3Refiner(
    private val pointCloud: List<Point3D>,
    private val pgPnpEngine: PgPnpEngine,
    private val correspondenceBuilder: SiftCorrespondenceBuilder,
    private val point3DIndex: Map<Pair<Long, Int>, Point3D>,
    private val referenceImageIdsProvider: (centerLat: Double, centerLon: Double) -> List<Long>,
    /** Origine du repere ENU local (fixee une fois, ex : centre du campus). */
    private val enuOriginLat: Double,
    private val enuOriginLon: Double
) {
    companion object {
        private const val TAG = "Level3Refiner"
    }

    data class Level3Result(val pose: GeoPose?, val inlierCount: Int, val fallback: Boolean)

    /**
     * @param cameraIntrinsics (fx, fy, cx, cy) reelles de l'image analysee (ex. Frame.camera.imageIntrinsics
     *        d'ARCore). Si non fournies, PgPnpEngine garde ses valeurs par defaut (a eviter en
     *        production : voir PgPnpEngine.updateIntrinsics pour pourquoi c'est critique).
     */
    fun refine(queryBitmap: Bitmap, priorPose: GeoPose, cameraIntrinsics: DoubleArray? = null): Level3Result {
        if (cameraIntrinsics != null && cameraIntrinsics.size == 4) {
            pgPnpEngine.updateIntrinsics(cameraIntrinsics[0], cameraIntrinsics[1], cameraIntrinsics[2], cameraIntrinsics[3])
        } else {
            Log.w(TAG, "ATTENTION : aucune intrinseque camera reelle fournie -- PgPnpEngine utilise ses " +
                "valeurs par defaut (1920x1080 suppose), quasi certainement fausses pour l'image reelle. " +
                "Ceci peut expliquer un taux d'inliers RANSAC nul de facon systematique.")
        }

        if (point3DIndex.isEmpty()) {
            Log.w(TAG, "ECHEC : point3DIndex est VIDE (points3D.txt absent/vide/mal parse dans assets/). " +
                "Verifiez que le fichier existe et correspond au format attendu.")
        }

        // Pose prior (Niveau 1 ou 2) exprimee en ENU local
        val priorEnu = GeoUtils.geodeticToEnu(priorPose.latitude, priorPose.longitude, 0.0, enuOriginLat, enuOriginLon, 0.0)
        val rotPrior = priorPose.orientation.toRotationMatrix()
        val tPrior = PoseMath.extrinsicTranslationFromCameraCenter(rotPrior, priorEnu)   // <-- FIX
        val priorPose3D = Pose3D(rotPrior, tPrior)
        Log.d(TAG, "[PRIOR] pose (lat=${priorPose.latitude}, lon=${priorPose.longitude}, niveau=${priorPose.level}) " +
            "-> ENU local (x=${"%.2f".format(priorEnu[0])}, y=${"%.2f".format(priorEnu[1])}, z=${"%.2f".format(priorEnu[2])}) " +
            "[origine ENU : lat=$enuOriginLat, lon=$enuOriginLon]")

        val refImages = referenceImageIdsProvider(priorPose.latitude, priorPose.longitude)
        if (refImages.isEmpty()) {
            Log.d(TAG, "ECHEC : aucune image de reference pres de (${priorPose.latitude}, ${priorPose.longitude}). " +
                "Causes possibles : images.txt absent/vide, originLat/originLon de LocalizerServiceLocator " +
                "incoherents avec la position reelle testee, ou rayon de SpatialImageIndex trop restrictif.")
            return Level3Result(pose = null, inlierCount = 0, fallback = true)
        }
        Log.d(TAG, "${refImages.size} image(s) de reference trouvee(s) : $refImages")

        val rawCorrespondences = try {
            correspondenceBuilder.buildCorrespondences(queryBitmap, refImages, point3DIndex)
        } catch (e: Throwable) {
            // Cause frequente : module OpenCV non ajoute au projet / bibliotheques natives non
            // chargees (SIFT.create() ou detectAndCompute() peuvent lever UnsatisfiedLinkError ou
            // equivalent). On journalise clairement puis on se replie, plutot que de laisser
            // l'exception se propager de facon non geree hors de la cascade.
            Log.e(TAG, "ECHEC : exception pendant l'extraction/l'appariement SIFT (verifiez l'integration " +
                "OpenCV -- module ajoute ? implementation(project(\":opencv\")) decommentee ?)", e)
            return Level3Result(pose = null, inlierCount = 0, fallback = true)
        }

        if (rawCorrespondences.isEmpty()) {
            Log.d(TAG, "ECHEC : 0 correspondance 2D-3D construite (SIFT n'a matche aucun point avec les " +
                "${refImages.size} image(s) de reference -- verifiez database.db, ratio test de Lowe, " +
                "ou que l'extracteur SIFT fonctionne reellement)")
            return Level3Result(pose = null, inlierCount = 0, fallback = true)
        }
        Log.d(TAG, "${rawCorrespondences.size} correspondance(s) 2D-3D brute(s) construite(s)")

        // CORRECTIF : rayon derive de la VRAIE incertitude du prior (58.64 m pour le Niveau 1,
        // 9.12 m pour le Niveau 2) au lieu du filterRadiusM=60.0 fixe de PgPnpEngine, qui etait
        // decouple de sigmaTApr=20.0 et ne representait qu'~1 sigma reel pour le Niveau 1.
        val dynamicRadiusM = PgPnpEngine.dynamicFilterRadius(priorPose.sigmaPosMeters)

        // DIAGNOSTIC : distance de chaque point 3D brut a la position prior, AVANT filtrage
        // geographique -- permet de comprendre POURQUOI le filtre rejette (ou accepte) les
        // correspondances : sont-elles juste hors du rayon dynamique, ou est-ce que le nuage de
        // points est dans un repere totalement different (des centaines/milliers de m d'ecart
        // signaleraient un probleme d'alignement ou d'unites, pas juste un rayon trop strict).
        if (rawCorrespondences.isNotEmpty()) {
            val distances = rawCorrespondences.map { c ->
                val dx = c.point3D[0] - priorEnu[0]; val dy = c.point3D[1] - priorEnu[1]
                Math.sqrt(dx * dx + dy * dy)
            }
            Log.d(TAG, "[GEO-FILTRE] distances brutes prior<->point3D (m) : " +
                "min=${"%.1f".format(distances.min())}, max=${"%.1f".format(distances.max())}, " +
                "moyenne=${"%.1f".format(distances.average())}, rayon du filtre=${"%.1f".format(dynamicRadiusM)}")
            Log.d(TAG, "[GEO-FILTRE] detail par correspondance : " +
                rawCorrespondences.mapIndexed { i, c ->
                    "#$i:point3D(${"%.1f".format(c.point3D[0])},${"%.1f".format(c.point3D[1])},${"%.1f".format(c.point3D[2])})" +
                    ",dist=${"%.1f".format(distances[i])}m"
                }.joinToString(" | "))
        }

        val filtered = pgPnpEngine.geographicFilter(rawCorrespondences, priorEnu, radiusM = dynamicRadiusM)
        Log.d(TAG, "${filtered.size} correspondance(s) apres filtrage geographique (sur ${rawCorrespondences.size}, " +
            "rayon=${"%.1f".format(dynamicRadiusM)}m derive de sigma_prior=${priorPose.sigmaPosMeters}m)")

        // DIAGNOSTIC (temporaire) : hypothese rotation APR non-informative
        /*val intr = cameraIntrinsics?.takeIf { it.size == 4 } ?: doubleArrayOf(1920.0, 1920.0, 960.0, 540.0)
        logRotationHypothesisTest(filtered, priorEnu, rotPrior, intr[0], intr[1], intr[2], intr[3])*/

        // CORRECTIF : propage l'incertitude REELLE du prior (Niveau 1 ou 2) a PgPnpEngine, au lieu
        // du sigmaTApr=20.0 fixe -- affecte sigma_pi adaptatif, lambda de rappel APR, et le
        // garde-fou de plausibilite finale (rejet si la pose s'eloigne de >5 sigma du prior).
        val result = pgPnpEngine.estimatePose(filtered, priorPose3D, sigmaTMeters = priorPose.sigmaPosMeters)

        if (result.fallbackToLevel2 || result.pose == null) {
            Log.d(TAG, "ECHEC : PG-PnP en repli (inliers=${result.inlierCount}, " +
                "minimum requis=6). Pas assez de correspondances fiables apres RANSAC.")
            return Level3Result(pose = null, inlierCount = result.inlierCount, fallback = true)
        }
        Log.d(TAG, "SUCCES : pose PG-PnP estimee avec ${result.inlierCount} inliers")

        // Reconversion ENU -> WGS-84 (approximation plane locale, coherente avec geodeticToEnu)
        val cRefined = PoseMath.cameraCenterFromPose(result.pose.rotation, result.pose.translation)  // <-- FIX
        val (lat, lon) = enuToGeodeticApprox(cRefined[0], cRefined[1])
        val q = rotationMatrixToQuaternion(result.pose.rotation)

        val refinedPose = GeoPose(
            latitude = lat, longitude = lon, orientation = q,
            level = 3, sigmaPosMeters = 2.0 // erreur cible < 2 m (Chapitre 3)
        )
        return Level3Result(pose = refinedPose, inlierCount = result.inlierCount, fallback = false)
    }

    private fun enuToGeodeticApprox(east: Double, north: Double): Pair<Double, Double> {
        val dLat = north / GeoUtils.EARTH_RADIUS_M
        val dLon = east / (GeoUtils.EARTH_RADIUS_M * Math.cos(Math.toRadians(enuOriginLat)))
        return Pair(enuOriginLat + Math.toDegrees(dLat), enuOriginLon + Math.toDegrees(dLon))
    }

    private fun rotationMatrixToQuaternion(m: DoubleArray): Quaternion {
        val trace = m[0] + m[4] + m[8]
        return if (trace > 0) {
            val s = 0.5 / Math.sqrt(trace + 1.0)
            Quaternion(0.25 / s, (m[7] - m[5]) * s, (m[2] - m[6]) * s, (m[3] - m[1]) * s).normalized()
        } else {
            Quaternion(1.0, 0.0, 0.0, 0.0)
        }
    }

    /**
     * DIAGNOSTIC (hypothese "orientation APR non-informative") : compare, sur les MEMES
     * correspondances filtrees geographiquement et la MEME position C, l'erreur de reprojection
     * sous deux hypotheses de rotation :
     *   (a) rotation predite par l'APR (rotPrior tel quel)
     *   (b) rotation identite (camera alignee sur les axes ENU -- hypothese naive de reference)
     * Si (b) donne des erreurs drastiquement plus faibles que (a), la rotation APR est la cause
     * principale de l'echec PG-PnP (et non un probleme de position, d'intrinseques ou de
     * correspondances 2D-3D -- ces trois-la sont deja corrects puisque la position C est bonne).
     * N'affecte PAS le flux normal : purement journalisation, aucune pose n'est modifiee ici.
     */
    private fun logRotationHypothesisTest(
        correspondences: List<Correspondence2D3D>,
        cameraCenterEnu: DoubleArray,
        aprRotation: DoubleArray,
        fx: Double, fy: Double, cx: Double, cy: Double
    ) {
        if (correspondences.isEmpty()) return

        fun reprojStats(label: String, r: DoubleArray) {
            val t = PoseMath.extrinsicTranslationFromCameraCenter(r, cameraCenterEnu)
            val errors = correspondences.map { c ->
                val proj = PoseMath.project(fx, fy, cx, cy, r, t, c.point3D)
                val dx = c.point2D[0] - proj[0]; val dy = c.point2D[1] - proj[1]
                Math.sqrt(dx * dx + dy * dy)
            }
            val sorted = errors.sorted()
            Log.d(TAG, "[DIAG-ROTATION][$label] min=${"%.1f".format(errors.min())}, " +
                    "max=${"%.1f".format(errors.max())}, moyenne=${"%.1f".format(errors.average())}, " +
                    "mediane=${"%.1f".format(sorted[sorted.size / 2])}")
        }
        /** Rotation autour de l'axe optique camera (roll), en degres -> matrice 3x3. */
        fun rollRotationDeg(deg: Double): DoubleArray {
            val rad = Math.toRadians(deg)
            val c = Math.cos(rad); val s = Math.sin(rad)
            return doubleArrayOf(
                c, -s, 0.0,
                s,  c, 0.0,
                0.0, 0.0, 1.0
            )
        }

        val identity = doubleArrayOf(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)

        Log.d(TAG, "[DIAG-ROTATION] === test sur ${correspondences.size} correspondance(s), " +
                "meme position C, deux hypotheses de rotation ===")
        reprojStats("rotation APR (telle quelle)", aprRotation)
        reprojStats("rotation IDENTITE (reference)", identity)

        // Hypothese : l'orientation APR a ete predite sur l'image tournee 90 deg (portrait, pour
// coherence avec l'entrainement), mais PG-PnP travaille sur l'image BRUTE paysage --
// un decalage de roll de 90 deg (dans un sens ou l'autre) n'est jamais compense.
        val rollPlus90 = PoseMath.matMul3(rollRotationDeg(90.0), aprRotation)
        val rollMinus90 = PoseMath.matMul3(rollRotationDeg(-90.0), aprRotation)
        reprojStats("APR + roll compense +90 (image tournee -> brute)", rollPlus90)
        reprojStats("APR + roll compense -90 (image tournee -> brute)", rollMinus90)

// Hypothese complementaire : inversion monde<->camera (R au lieu de R^T, ou l'inverse)
        val rt = doubleArrayOf(aprRotation[0], aprRotation[3], aprRotation[6],
            aprRotation[1], aprRotation[4], aprRotation[7],
            aprRotation[2], aprRotation[5], aprRotation[8])
        reprojStats("APR transposee (R^T, inversion monde<->camera)", rt)
        reprojStats("APR transposee (R^T, inversion monde<->camera)", rt)
    }
}
