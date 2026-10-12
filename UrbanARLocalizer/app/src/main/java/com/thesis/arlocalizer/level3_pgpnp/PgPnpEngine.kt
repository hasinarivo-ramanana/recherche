package com.thesis.arlocalizer.level3_pgpnp

import android.util.Log
import com.thesis.arlocalizer.model.Correspondence2D3D
import com.thesis.arlocalizer.model.Pose3D
import kotlin.math.*
import kotlin.random.Random

import org.opencv.calib3d.Calib3d
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfDouble
import org.opencv.core.MatOfInt
import org.opencv.core.MatOfPoint2f
import org.opencv.core.MatOfPoint3f
import org.opencv.core.Point
import org.opencv.core.Point3

/**
 * Contribution principale du Chapitre 3 : PnP Guide par Prior APR (PG-PnP).
 *
 * Trois etapes (Fig. pgpnp_overview / algo_pgpnp) :
 *  1) Filtrage geographique des correspondances 2D-3D (Eq. filtre_geo, r = 60 m)
 *  2) RANSAC pondere par la coherence avec la pose APR (Eq. poids_ransac, sigma_pi = f*sigma_t/d_typical)
 *  3) Affinement guide par la position APR : min reprojection + lambda*||t - t_APR||^2 (Eq. pgpnp_simple / lambda)
 *
 * Repere de travail : ENU metrique local centre sur la position APR (voir GeoUtils / PoseAnchorManager
 * pour la conversion vers/depuis WGS-84).
 *
 * DIAGNOSTIC : logs detailles sous le tag "Level3Refiner" (meme tag que Level3Refiner.kt et
 * SiftCorrespondenceBuilder.kt) -- sigma_pi/d_typical adaptatifs, distribution des poids RANSAC,
 * meilleur nombre d'inliers trouve a chaque tentative (meme sous le seuil minInliers=6).
 */
class PgPnpEngine(
    private var fx: Double = 1400.0,
    private var fy: Double = 1400.0,
    private var cx: Double = 960.0,
    private var cy: Double = 540.0,
    private val filterRadiusM: Double = 60.0,      // r = 3*sigma_t, sigma_t ~= 20 m
    private val reprojThresholdPx: Double = 4.0,   // epsilon, seuil d'inlier RANSAC
    private val minInliers: Int = 4,
    private val ransacIterations: Int = 60,        // borne haute apres filtrage+ponderation (Tableau ransac_iterations)
    private val sigmaTApr: Double = 20.0           // sigma_t, ecart-type de l'erreur de position APR (m)
) {
    companion object {
        private const val TAG = "Level3Refiner"

        /**
         * CORRECTIF : rayon de filtrage geographique derive de la VRAIE incertitude de position
         * du prior (ex. priorPose.sigmaPosMeters -- 58.64 m pour le Niveau 1, 9.12 m pour le
         * Niveau 2), au lieu du filterRadiusM=60.0 fixe qui etait decouple de sigmaTApr=20.0
         * (le commentaire "r = 3*sigma_t" ne correspondait donc a AUCUN des deux : ni a 20.0
         * (3*20=60, coincidence trompeuse), ni surtout a l'incertitude reelle du Niveau 1
         * (58.64 m), pour laquelle un rayon de 60 m ne represente qu'environ 1 sigma -- excluant
         * potentiellement de bonnes correspondances des que l'erreur APR reelle depasse 60 m.
         */
        fun dynamicFilterRadius(
            sigmaPosMeters: Double,
            multiplier: Double = 3.0,
            minRadiusM: Double = 30.0,
            maxRadiusM: Double = 200.0
        ): Double = (sigmaPosMeters * multiplier).coerceIn(minRadiusM, maxRadiusM)
    }
    private val RANSAC_P_SUCCESS_CONFIDENCE = 0.99 // = config.RANSAC_P_SUCCESS du pipeline Python

    /**
     * Met a jour les intrinseques caméra AVANT chaque appel a estimatePose(), avec les vraies
     * valeurs de l'image en cours d'analyse (ex. Frame.camera.imageIntrinsics d'ARCore). Sans cette
     * mise a jour, les valeurs par defaut (1920x1080 suppose) sont quasi certainement fausses pour
     * l'image reellement capturee -- ce qui biaise systematiquement TOUTE erreur de reprojection et
     * empeche RANSAC de jamais valider un inlier, meme pour de bonnes correspondances SIFT.
     */
    fun updateIntrinsics(fx: Double, fy: Double, cx: Double, cy: Double) {
        this.fx = fx; this.fy = fy; this.cx = cx; this.cy = cy
    }

    data class PgPnpResult(
        val pose: Pose3D?,
        val inlierCount: Int,
        val totalCorrespondences: Int,
        val fallbackToLevel2: Boolean
    )

    /**
     * Etape 1 : filtre les correspondances dont le point 3D est a plus de `filterRadiusM` de la position APR
     * (Eq. filtre_geo). aprPositionEnu est la position APR (Niveau 1/2) exprimee dans le meme repere ENU
     * que les points 3D du nuage (apres application de SfmToEnuAligner si necessaire).
     */
    fun geographicFilter(
        correspondences: List<Correspondence2D3D>,
        aprPositionEnu: DoubleArray,
        radiusM: Double = filterRadiusM // CORRECTIF : appelant peut passer dynamicFilterRadius(sigmaReel)
    ): List<Correspondence2D3D> {
        return correspondences.filter { c ->
            val dx = c.point3D[0] - aprPositionEnu[0]
            val dy = c.point3D[1] - aprPositionEnu[1]
            val d = sqrt(dx * dx + dy * dy)
            d <= radiusM
        }
    }

    /**
     * Sigma_pi adaptatif : sigma_pi = f * sigma_t / d_typical (correction de l'incoherence
     * notationnelle documentee : la valeur fixe 15 px du manuscrit n'est valable qu'a une
     * distance typique donnee -- ici on la derive explicitement).
     */
    fun adaptiveSigmaPi(dTypicalM: Double, sigmaTMeters: Double = sigmaTApr): Double {
        val f = (fx + fy) / 2.0
        return f * sigmaTMeters / max(dTypicalM, 1.0)
    }

    /**
     * Etape 2 : calcule les poids RANSAC w_i (Eq. poids_ransac) a partir du residu de reprojection
     * sous la pose APR utilisee comme estimation provisoire.
     */
    fun computeRansacWeights(correspondences: List<Correspondence2D3D>, aprPose: Pose3D, sigmaPiPx: Double): DoubleArray {
        return DoubleArray(correspondences.size) { i ->
            val c = correspondences[i]
            val proj = PoseMath.project(fx, fy, cx, cy, aprPose.rotation, aprPose.translation, c.point3D)
            val dx = c.point2D[0] - proj[0]
            val dy = c.point2D[1] - proj[1]
            val r = sqrt(dx * dx + dy * dy)
            exp(-(r * r) / (2 * sigmaPiPx * sigmaPiPx))
        }
    }

    /**
     * CORRECTIF (diversite des points 3D) : un meme point 3D COLMAP est souvent observe par
     * plusieurs images de reference -> plusieurs "correspondances" differentes de notre liste
     * pointent en realite vers des coordonnees (x,y,z) quasi-identiques. Sans garde-fou, un
     * tirage de nMin=6 points pour EPnP peut piocher plusieurs fois le MEME point 3D (sous des
     * point2D differents, potentiellement de faux appariements SIFT) : le systeme lineaire
     * devient quasi degenere (points confondus au lieu de 6 points independants), ce qui peut
     * produire une pose totalement aberrante tout en "collant" par coincidence a quelques
     * doublons -> faux "succes" RANSAC (cf. saut de pose a ~2.7 km observe en logs malgre
     * 9 inliers). On arrondit a 1 cm pour regrouper les points quasi-identiques.
     */
    private fun point3DKey(p: DoubleArray): Triple<Long, Long, Long> =
        Triple(Math.round(p[0] * 100.0), Math.round(p[1] * 100.0), Math.round(p[2] * 100.0))

    /** Tirage pondere sans remise (roulette-wheel) de n_min indices parmi les correspondances,
     * en evitant les doublons de point 3D (cf. point3DKey ci-dessus). */
    private fun weightedSample(
        weights: DoubleArray,
        nMin: Int,
        rng: Random,
        pointKeys: Array<Triple<Long, Long, Long>>
    ): IntArray? {
        if (weights.size < nMin) return null
        val available = weights.indices.toMutableList()
        val chosen = mutableListOf<Int>()
        val chosenKeys = mutableSetOf<Triple<Long, Long, Long>>()
        var attemptsLeft = weights.size * 4 // borne de securite, evite une boucle infinie si peu de diversite
        while (chosen.size < nMin && available.isNotEmpty() && attemptsLeft > 0) {
            attemptsLeft--
            val w = available.map { weights[it] + 1e-6 }
            val sum = w.sum()
            var r = rng.nextDouble() * sum
            var pick = 0
            for (i in w.indices) {
                r -= w[i]
                if (r <= 0) { pick = i; break }
            }
            val idx = available[pick]
            val key = pointKeys[idx]
            if (key in chosenKeys) {
                // point 3D deja represente dans cet echantillon : on le retire du pool pour CE
                // tirage (pas de sa liste globale) et on retente, sans compter parmi les nMin choisis
                available.removeAt(pick)
                continue
            }
            chosen.add(idx)
            chosenKeys.add(key)
            available.removeAt(pick)
        }
        // Pas assez de points 3D DISTINCTS disponibles pour former un echantillon EPnP valide
        if (chosen.size < nMin) return null
        return chosen.toIntArray()
    }

    /**
     * Etapes 2 + 3 completes : RANSAC pondere puis affinement guide par la position APR.
     * @param correspondences correspondances 2D-3D deja filtrees geographiquement
     * @param aprPose pose initiale (translation = position APR en ENU, rotation = orientation APR)
     */
    fun estimatePose(
        correspondences: List<Correspondence2D3D>,
        aprPose: Pose3D,
        seed: Long = 42L,
        // CORRECTIF : incertitude reelle du prior (ex. priorPose.sigmaPosMeters -- 58.64 m pour
        // le Niveau 1, 9.12 m pour le Niveau 2) au lieu du sigmaTApr=20.0 fixe, decouple de la
        // veritable erreur mesuree du niveau precedent.
        sigmaTMeters: Double = sigmaTApr,
        // CORRECTIF : nombre minimum de points 3D DISTINCTS (pas de correspondances brutes,
        // potentiellement dupliquees via le meme track COLMAP) parmi les inliers retenus.
        minUniquePoints3D: Int = minInliers,
        // CORRECTIF : garde-fou de plausibilite -- une pose dont le centre camera s'eloigne de
        // plus de maxPoseJumpSigmaMultiplier * sigmaTMeters du prior est rejetee, MEME si elle a
        // passe le seuil minInliers (cf. saut a ~2.7 km observe en logs malgre 9 inliers).
        maxPoseJumpSigmaMultiplier: Double = 5.0
    ): PgPnpResult {
        Log.d(TAG, "[PG-PnP] intrinseques utilisees : fx=$fx, fy=$fy, cx=$cx, cy=$cy")
        Log.d(TAG, "[PG-PnP] pose prior (translation ENU) : x=${"%.2f".format(aprPose.translation[0])}, " +
            "y=${"%.2f".format(aprPose.translation[1])}, z=${"%.2f".format(aprPose.translation[2])}")

        if (correspondences.size < minInliers) {
            Log.w(TAG, "[PG-PnP] ECHEC PRECOCE : ${correspondences.size} correspondance(s) disponible(s) " +
                "< minInliers=$minInliers -- RANSAC necessite au moins $minInliers points pour meme demarrer, abandon immediat")
            return PgPnpResult(pose = null, inlierCount = 0, totalCorrespondences = correspondences.size, fallbackToLevel2 = true)
        }

        // IMPORTANT : aprPose.translation est le vecteur extrinseque t (= -R*C), PAS la position
// monde de la camera. Pour calculer une distance geographique (d_typical) entre le point 3D
// (coordonnees monde/ENU) et la camera, il faut reconvertir t -> C = -R^T*t.
        val aprCenterEnu = PoseMath.cameraCenterFromPose(aprPose.rotation, aprPose.translation)
        val dTypical = correspondences.map { c ->
            val dx = c.point3D[0] - aprCenterEnu[0]; val dy = c.point3D[1] - aprCenterEnu[1]
            sqrt(dx * dx + dy * dy)
        }.average().coerceAtLeast(5.0)
        val sigmaPi = adaptiveSigmaPi(dTypical, sigmaTMeters)
        Log.d(TAG, "[PG-PnP] d_typical=${"%.2f".format(dTypical)} m, sigma_pi adaptatif=${"%.2f".format(sigmaPi)} px " +
            "(f_moy=${(fx + fy) / 2.0}, sigma_t=$sigmaTMeters m)")

        val weights = computeRansacWeights(correspondences, aprPose, sigmaPi)
        Log.d(TAG, "[PG-PnP] poids RANSAC : min=${"%.4f".format(weights.min())}, max=${"%.4f".format(weights.max())}, " +
            "moyenne=${"%.4f".format(weights.average())} (poids uniforme si tous proches de 1/N=${"%.4f".format(1.0/weights.size)})")

        /*
        //***********Début test ransac habituel ************//
        val rng = Random(seed) // conserve pour la reproductibilite du reste du pipeline (non utilise par OpenCV ici)

// --- Duplication ponderee : approxime un tirage RANSAC pondere (Eq. poids_ransac) avec un
// echantillonneur uniforme natif OpenCV. Chaque correspondance apparait entre 1 et
// maxDuplication fois, proportionnellement a son poids de coherence APR -- les points les
// plus coherents avec le prior sont donc statistiquement plus souvent tires par
// solvePnPRansac, sans avoir a reimplementer la boucle RANSAC nous-memes.
        val maxDuplication = 5
        val expandedIdx = mutableListOf<Int>()
        for (i in correspondences.indices) {
            val copies = 1 + (weights[i] * (maxDuplication - 1)).toInt().coerceIn(0, maxDuplication - 1)
            repeat(copies) { expandedIdx.add(i) }
        }
        Log.d(TAG, "[PG-PnP] duplication ponderee : ${correspondences.size} correspondance(s) -> " +
                "${expandedIdx.size} echantillon(s) pour solvePnPRansac (maxDuplication=$maxDuplication)")

        val objectPoints = MatOfPoint3f().apply {
            fromList(expandedIdx.map { i -> Point3(correspondences[i].point3D[0], correspondences[i].point3D[1], correspondences[i].point3D[2]) })
        }
        val imagePoints = MatOfPoint2f().apply {
            fromList(expandedIdx.map { i -> Point(correspondences[i].point2D[0], correspondences[i].point2D[1]) })
        }
        val cameraMatrix = Mat(3, 3, CvType.CV_64F).apply {
            put(0, 0, fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0)
        }
        val distCoeffs = MatOfDouble() // pas de distorsion geree (coherent avec PoseMath.project)
        val rvec = Mat(); val tvec = Mat(); val inliersMat = MatOfInt()

        var bestInliers: List<Int> = emptyList()
        var bestPose = aprPose

        val ransacOk = try {
            Calib3d.solvePnPRansac(
                objectPoints, imagePoints, cameraMatrix, distCoeffs,
                rvec, tvec, false,
                ransacIterations, reprojThresholdPx.toFloat(), RANSAC_P_SUCCESS_CONFIDENCE,
                inliersMat, Calib3d.SOLVEPNP_EPNP
            )
        } catch (e: Throwable) {
            Log.w(TAG, "[PG-PnP] echec solvePnPRansac (probablement systeme degenere) : ${e.message}")
            false
        }

        if (ransacOk) {
            val rotMat = Mat()
            Calib3d.Rodrigues(rvec, rotMat)
            val r = DoubleArray(9); rotMat.get(0, 0, r)
            val t = DoubleArray(3); tvec.get(0, 0, t)
            bestPose = Pose3D(r, t)

            // Remappe les indices d'inliers (dans le tableau DUPLIQUE) vers les indices ORIGINAUX,
            // dedupliques (plusieurs copies d'une meme correspondance peuvent toutes etre inliers).
            val inlierExpandedIndices = inliersMat.toArray()
            bestInliers = inlierExpandedIndices.map { expandedIdx[it] }.distinct()

            rotMat.release()
            Log.d(TAG, "[PG-PnP] solvePnPRansac (OpenCV, SOLVEPNP_EPNP) : ${bestInliers.size} inlier(s) " +
                    "unique(s) sur ${correspondences.size} correspondance(s)")
        } else {
            Log.w(TAG, "[PG-PnP] solvePnPRansac a echoue ou n'a pas convergé -- 0 inlier retenu")
        }

        objectPoints.release(); imagePoints.release(); cameraMatrix.release()
        rvec.release(); tvec.release(); inliersMat.release()
        ///*******Fin Ransac habituel test************//
        */



        //**********Début ransac réel implémentation********//
        val rng = Random(seed)
        val pointKeys = Array(correspondences.size) { i -> point3DKey(correspondences[i].point3D) }

        val epnpSampleSize = min(6, correspondences.size)
        var bestInliers: List<Int> = emptyList()
        var bestUniquePointCount = 0
        var bestPose = aprPose
        var bestInliersEverySoOften = mutableListOf<Int>()

        repeat(ransacIterations) { iter ->
            val sampleIdx = weightedSample(weights, nMin = epnpSampleSize, rng = rng, pointKeys = pointKeys) ?: return@repeat
            val sample = sampleIdx.map { correspondences[it] }
            // EPnP (Lepetit et al. 2009, meme algorithme que cv2.SOLVEPNP_EPNP) : solveur GLOBAL,
            // aucune pose initiale requise -- remplace l'ancien raffinement Gauss-Newton local partant
            // de la rotation APR, qui restait piege des lors que cette rotation etait trop imprecise
            // (cause racine confirmee par le diagnostic [DIAG-ROTATION] : aucune convention fixe --
            // roll +/-90, transposee, identite -- ne dominait systematiquement, signe d'un bruit reel
            // sur l'orientation plutot que d'une simple erreur de convention).
            val candidatePose = EPnPSolver.solve(sample, fx, fy, cx, cy) ?: return@repeat

            val inliers = correspondences.indices.filter { i ->
                val c = correspondences[i]
                val proj = PoseMath.project(fx, fy, cx, cy, candidatePose.rotation, candidatePose.translation, c.point3D)
                val dx = c.point2D[0] - proj[0]; val dy = c.point2D[1] - proj[1]
                sqrt(dx * dx + dy * dy) <= reprojThresholdPx
            }
            // CORRECTIF : on retient la meilleure pose sur la base du nombre de points 3D
            // DISTINCTS parmi les inliers (et non du nombre brut de correspondances), pour ne pas
            // favoriser une pose qui "gagne" uniquement en accumulant des doublons du meme track.
            val uniqueCount = inliers.map { pointKeys[it] }.distinct().size
            if (inliers.size > bestInliers.size) {
                bestInliers = inliers
                bestUniquePointCount = uniqueCount
                bestPose = candidatePose
                bestInliersEverySoOften.add(inliers.size)
            }
        }
        //**********Fin ransac réel implémentation********//



        // Reprojection de TOUTES les correspondances sous la MEILLEURE pose trouvee (meme si echec) :
        // montre si on etait "proche" (erreurs de quelques px de trop) ou "loin" (erreurs enormes,
        // signe d'un probleme geometrique plus profond que juste un seuil trop strict).
        val allReprojErrors = correspondences.map { c ->
            val proj = PoseMath.project(fx, fy, cx, cy, bestPose.rotation, bestPose.translation, c.point3D)
            val dx = c.point2D[0] - proj[0]; val dy = c.point2D[1] - proj[1]
            sqrt(dx * dx + dy * dy)
        }
        Log.d(TAG, "[PG-PnP] solvePnPRansac (OpenCV) : meilleur score=${bestInliers.size} inlier(s), " +
                "seuil de reprojection=${reprojThresholdPx}px")
        Log.d(TAG, "[PG-PnP] erreurs de reprojection sous la MEILLEURE pose (toutes correspondances, px) : " +
            "min=${"%.1f".format(allReprojErrors.min())}, max=${"%.1f".format(allReprojErrors.max())}, " +
            "moyenne=${"%.1f".format(allReprojErrors.average())}, mediane=${"%.1f".format(allReprojErrors.sorted()[allReprojErrors.size/2])}")

        if (bestInliers.size < minInliers) {
            Log.w(TAG, "[PG-PnP] ECHEC : ${bestInliers.size} inlier(s) < minInliers=$minInliers. " +
                if (allReprojErrors.average() > 100) "Erreurs de reprojection tres elevees (>100px en moyenne) -- " +
                    "suggere un probleme geometrique (intrinseques, alignement SfM->ENU, ou correspondances 2D-3D incorrectes), " +
                    "pas juste un manque de donnees."
                else "Erreurs de reprojection moderees -- pourrait etre un manque de correspondances valides " +
                    "ou un seuil de reprojection (${reprojThresholdPx}px) trop strict pour votre calibration.")
            return PgPnpResult(pose = null, inlierCount = bestInliers.size, totalCorrespondences = correspondences.size, fallbackToLevel2 = true)
        }

        // CORRECTIF : rejet si les inliers ne couvrent pas assez de points 3D DISTINCTS -- un
        // nombre brut d'inliers eleve peut n'etre que quelques points 3D (meme track COLMAP)
        // vus sous plusieurs point2D, ce qui ne contraint pas correctement les 6 DDL de la pose
        // (cf. saut de pose a ~2.7 km observe malgre 9 "inliers" bruts).
        if (bestUniquePointCount < minUniquePoints3D) {
            Log.w(TAG, "[PG-PnP] ECHEC : ${bestInliers.size} inlier(s) brut(s) mais seulement " +
                "$bestUniquePointCount point(s) 3D DISTINCT(S) < minUniquePoints3D=$minUniquePoints3D -- " +
                "pose probablement sous-contrainte/degeneree (points 3D quasi-confondus dans l'echantillon).")
            return PgPnpResult(pose = null, inlierCount = bestInliers.size, totalCorrespondences = correspondences.size, fallbackToLevel2 = true)
        }

        // Etape 3 : affinement guide sur l'ensemble des inliers (Eq. pgpnp_simple + lambda adaptatif, Eq. lambda)
        val inlierCorrespondences = bestInliers.map { correspondences[it] }
        val meanReprojErrBefore = inlierCorrespondences.map { c ->
            val proj = PoseMath.project(fx, fy, cx, cy, bestPose.rotation, bestPose.translation, c.point3D)
            val dx = c.point2D[0] - proj[0]; val dy = c.point2D[1] - proj[1]
            sqrt(dx * dx + dy * dy)
        }.average()

        val lambda = (meanReprojErrBefore * meanReprojErrBefore) / (sigmaTMeters * sigmaTMeters)

        val refinedPose = gaussNewtonRefine(inlierCorrespondences, bestPose, lambda = lambda, tApr = aprPose.translation, iterations = 15)
            ?: bestPose

        // CORRECTIF (garde-fou de plausibilite) : le lambda ci-dessus est calcule a partir des
        // residus des inliers SOUS LA POSE CANDIDATE ELLE-MEME -- si cette pose est deja fausse
        // mais "coherente" avec ses propres inliers (cf. doublons de points 3D), lambda reste
        // petit et ne corrige quasiment rien. On ajoute donc une verification ABSOLUE,
        // independante des residus : le centre camera final ne doit pas s'eloigner de plus de
        // maxPoseJumpSigmaMultiplier * sigmaTMeters du centre du prior APR -- sinon, rejet
        // explicite plutot qu'un "succes" trompeur (cf. saut a ~2.7 km observe en logs).
        val refinedCenterEnu = PoseMath.cameraCenterFromPose(refinedPose.rotation, refinedPose.translation)
        val poseJumpM = sqrt(
            (refinedCenterEnu[0] - aprCenterEnu[0]).let { it * it } +
            (refinedCenterEnu[1] - aprCenterEnu[1]).let { it * it }
        )
        val maxAllowedJumpM = maxPoseJumpSigmaMultiplier * sigmaTMeters
        if (poseJumpM > maxAllowedJumpM) {
            Log.w(TAG, "[PG-PnP] ECHEC (garde-fou de plausibilite) : ${bestInliers.size} inlier(s) / " +
                "$bestUniquePointCount point(s) distinct(s), mais la pose resultante s'eloigne de " +
                "${"%.1f".format(poseJumpM)} m du prior APR (max autorise=${"%.1f".format(maxAllowedJumpM)} m = " +
                "$maxPoseJumpSigmaMultiplier x sigma_t=$sigmaTMeters m). Rejet -- solution RANSAC/EPnP " +
                "probablement degeneree malgre un nombre d'inliers en apparence suffisant.")
            return PgPnpResult(pose = null, inlierCount = bestInliers.size, totalCorrespondences = correspondences.size, fallbackToLevel2 = true)
        }

        Log.d(TAG, "[PG-PnP] SUCCES : ${bestInliers.size} inliers ($bestUniquePointCount point(s) 3D distinct(s)), " +
            "lambda=${"%.4f".format(lambda)}, saut vs prior=${"%.1f".format(poseJumpM)} m, " +
            "pose finale (ENU) x=${"%.2f".format(refinedPose.translation[0])}, y=${"%.2f".format(refinedPose.translation[1])}, " +
            "z=${"%.2f".format(refinedPose.translation[2])}")

        return PgPnpResult(pose = refinedPose, inlierCount = bestInliers.size, totalCorrespondences = correspondences.size, fallbackToLevel2 = false)
    }

    /**
     * Minimisation de Gauss-Newton de :
     *   sum_i || x_i - pi(K, R, t, X_i) ||^2   +   lambda * || t - t_APR ||^2
     * Parametrisation : perturbation (delta_w in so(3), delta_t in R^3) autour de la pose courante.
     * Jacobien numerique (differences finies), suffisant vu la petite taille du probleme (< 1000 points, 6 params).
     */
    private fun gaussNewtonRefine(
        correspondences: List<Correspondence2D3D>,
        initialPose: Pose3D,
        lambda: Double,
        tApr: DoubleArray,
        iterations: Int
    ): Pose3D? {
        var r = initialPose.rotation.copyOf()
        var t = initialPose.translation.copyOf()
        val eps = 1e-6

        repeat(iterations) {
            val h = Array(6) { DoubleArray(6) }
            val g = DoubleArray(6)

            fun residualsAt(rr: DoubleArray, tt: DoubleArray): DoubleArray {
                val res = DoubleArray(correspondences.size * 2)
                for ((i, c) in correspondences.withIndex()) {
                    val proj = PoseMath.project(fx, fy, cx, cy, rr, tt, c.point3D)
                    res[2 * i] = c.point2D[0] - proj[0]
                    res[2 * i + 1] = c.point2D[1] - proj[1]
                }
                return res
            }

            val res0 = residualsAt(r, t)

            // Jacobien numerique (6 colonnes : 3 rotation, 3 translation)
            val jac = Array(res0.size) { DoubleArray(6) }
            for (p in 0 until 3) {
                val dw = DoubleArray(3); dw[p] = eps
                val rPerturbed = PoseMath.composeRotation(r, dw)
                val resP = residualsAt(rPerturbed, t)
                for (row in res0.indices) jac[row][p] = (resP[row] - res0[row]) / eps
            }
            for (p in 0 until 3) {
                val dt = t.copyOf(); dt[p] += eps
                val resP = residualsAt(r, dt)
                for (row in res0.indices) jac[row][3 + p] = (resP[row] - res0[row]) / eps
            }

            // H = J^T J, g = J^T res  (residus negatifs car on minimise ||res||^2 avec res = obs - pred)
            for (row in res0.indices) {
                for (a in 0 until 6) {
                    g[a] += jac[row][a] * res0[row]
                    for (b in 0 until 6) h[a][b] += jac[row][a] * jac[row][b]
                }
            }

            // Terme de rappel APR (Eq. pgpnp_simple) : ajoute lambda*I sur le bloc translation
            // et lambda*(t_APR - t) au gradient correspondant.
            if (lambda > 0.0) {
                for (p in 0 until 3) {
                    h[3 + p][3 + p] += lambda
                    g[3 + p] += lambda * (tApr[p] - t[p])
                }
            }

            // Amortissement Levenberg-Marquardt leger pour la stabilite numerique
            for (a in 0 until 6) h[a][a] += 1e-6 * (h[a][a] + 1.0)

            val delta = PoseMath.solveLinearSystem(h, g) ?: return@repeat

            r = PoseMath.composeRotation(r, doubleArrayOf(delta[0], delta[1], delta[2]))
            t = doubleArrayOf(t[0] + delta[3], t[1] + delta[4], t[2] + delta[5])
        }
        return Pose3D(r, t)
    }
}
