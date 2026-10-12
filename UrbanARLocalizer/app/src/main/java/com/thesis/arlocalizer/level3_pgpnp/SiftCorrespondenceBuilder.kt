package com.thesis.arlocalizer.level3_pgpnp

import android.graphics.Bitmap
import android.util.Log
import com.thesis.arlocalizer.model.Correspondence2D3D
import com.thesis.arlocalizer.model.Point3D
import kotlin.math.sqrt

/**
 * Construit les correspondances 2D (image requete) <-> 3D (nuage COLMAP) en appariant les descripteurs
 * SIFT de l'image requete avec les descripteurs ORIGINAUX stockes dans database.db (seule approche fiable,
 * cf. section "Reading original SIFT descriptors"), puis en remontant du point-image au point-3D via la
 * table de correspondance (track) exportee lors de la reconstruction COLMAP (points3D.txt/bin -> track).
 *
 * L'extraction SIFT de l'image requete et le "ratio test" de Lowe sont deportes vers OpenCV
 * (org.opencv.features2d.SIFT + BFMatcher, KNN k=2) pour des raisons de performance native.
 * Cette classe definit l'interface et la logique d'appariement independante d'OpenCV ; l'extracteur
 * concret est injecte via `SiftExtractor` pour ne pas figer une dependance native ici.
 *
 * DIAGNOSTIC : logs detailles sous le tag "Level3Refiner" (partage avec Level3Refiner.kt et
 * PgPnpEngine.kt pour un filtrage Logcat unique) -- nombre de points-cles extraits, attrition a
 * chaque etape (ratio test de Lowe, puis lookup point3DIndex) par image de reference.
 */
interface SiftExtractor {
    /** Retourne (keypoints en pixels (u,v), descripteurs 128-D uint8) pour l'image donnee. */
    fun extract(bitmap: Bitmap): Pair<Array<DoubleArray>, Array<IntArray>>
}

class SiftCorrespondenceBuilder(
    private val siftExtractor: SiftExtractor,
    private val dbReader: ColmapDatabaseReader,
    private val loweRatio: Double = 0.75
) {
    companion object {
        private const val TAG = "Level3Refiner"
    }

    /**
     * @param referenceImageIds sous-ensemble d'images COLMAP proches de la zone APR (filtrees par le SIG/index spatial)
     * @param point3DIndex table point2D (image_id, point2D_idx) -> Point3D, construite hors-ligne depuis points3D.bin/.txt
     */
    fun buildCorrespondences(
        queryBitmap: Bitmap,
        referenceImageIds: List<Long>,
        point3DIndex: Map<Pair<Long, Int>, Point3D>
    ): List<Correspondence2D3D> {
        val (queryKp, queryDesc) = siftExtractor.extract(queryBitmap)
        Log.d(TAG, "[SIFT] ${queryDesc.size} point(s)-cle(s) extrait(s) de l'image requete (${queryBitmap.width}x${queryBitmap.height} px)")
        if (queryDesc.isEmpty()) {
            Log.w(TAG, "[SIFT] ATTENTION : 0 point-cle extrait -- verifiez l'integration OpenCV " +
                "(module ajoute au projet ? OpenCVLoader initialise ?) ou la qualite/contraste de l'image")
        }

        val correspondences = mutableListOf<Correspondence2D3D>()

        for (imgId in referenceImageIds) {
            val refDesc = dbReader.getDescriptors(imgId)
            val refKp = dbReader.getKeypoints(imgId)
            if (refDesc.isEmpty()) {
                Log.d(TAG, "[SIFT] image $imgId : 0 descripteur dans database.db (image absente de la base ou id incorrect)")
                continue
            }

            var passedRatioTest = 0
            var missingIn3DIndex = 0
            var addedToCorrespondences = 0

            for (qi in queryDesc.indices) {
                val (bestIdx, bestDist, secondDist) = twoNearestNeighbors(queryDesc[qi], refDesc)
                if (bestIdx < 0) continue
                // Ratio test de Lowe : rejette les appariements ambigus (facades similaires -> nombreux outliers)
                if (secondDist > 0 && bestDist / secondDist > loweRatio) continue
                passedRatioTest++

                val point3D = point3DIndex[imgId to bestIdx]
                if (point3D == null) {
                    missingIn3DIndex++
                    continue
                }
                addedToCorrespondences++
                correspondences.add(
                    Correspondence2D3D(
                        point2D = queryKp[qi],
                        point3D = doubleArrayOf(point3D.x, point3D.y, point3D.z)
                    )
                )
            }

            Log.d(TAG, "[SIFT] image $imgId : ${refDesc.size} descripteur(s) en base, " +
                "$passedRatioTest apparie(s) apres ratio test de Lowe (seuil=$loweRatio), " +
                "$missingIn3DIndex sans point3D correspondant (point2D_idx absent du track), " +
                "$addedToCorrespondences correspondance(s) finale(s)")
        }

        Log.d(TAG, "[SIFT] TOTAL : ${correspondences.size} correspondance(s) 2D-3D sur " +
            "${referenceImageIds.size} image(s) de reference et ${queryDesc.size} point(s)-cle(s) requete")

        return correspondences
    }

    /** Distance L2 (SIFT), recherche brute-force des 2 plus proches voisins (jeu filtre localement, taille raisonnable). */
    private fun twoNearestNeighbors(query: IntArray, refs: Array<IntArray>): Triple<Int, Double, Double> {
        var best = -1; var bestD = Double.MAX_VALUE; var secondD = Double.MAX_VALUE
        for (i in refs.indices) {
            val d = l2Distance(query, refs[i])
            if (d < bestD) {
                secondD = bestD
                bestD = d
                best = i
            } else if (d < secondD) {
                secondD = d
            }
        }
        return Triple(best, bestD, secondD)
    }

    private fun l2Distance(a: IntArray, b: IntArray): Double {
        var s = 0.0
        for (i in a.indices) { val d = (a[i] - b[i]).toDouble(); s += d * d }
        return sqrt(s)
    }
}
