package com.thesis.arlocalizer.level3_pgpnp

import android.graphics.Bitmap
import org.opencv.android.Utils
import org.opencv.core.Mat
import org.opencv.core.MatOfKeyPoint
import org.opencv.features2d.SIFT
import org.opencv.imgproc.Imgproc
import org.opencv.android.OpenCVLoader
import android.util.Log
import kotlin.math.round
import kotlin.math.sqrt

/**
 * Extraction SIFT reelle via OpenCV (SiftExtractor utilise par SiftCorrespondenceBuilder, Niveau 3).
 *
 * IMPORTANT - compatibilite avec les descripteurs de database.db : OpenCV calcule des descripteurs SIFT
 * en float, avec sa propre normalisation interne, qui ne sont PAS directement comparables (distance L2)
 * aux descripteurs uint8 stockes par COLMAP. COLMAP applique une convention de quantification precise
 * (cf. code source COLMAP, feature_extraction) :
 *   1) normalisation L2 du descripteur 128-D
 *   2) multiplication par 512
 *   3) ecretage a [0, 255]
 *   4) arrondi et conversion en uint8
 * On reproduit exactement cette meme transformation ici, pour que la recherche de plus proches voisins
 * (ColmapDatabaseReader + SiftCorrespondenceBuilder) compare des grandeurs homogenes.
 *
 * Necessite le module OpenCV Android SDK ajoute au projet (voir README, section "Integration OpenCV").
 */
class OpenCvSiftExtractor : SiftExtractor {

    companion object {
        init {
            if (!OpenCVLoader.initLocal()) {
                Log.e("OpenCvSiftExtractor", "OpenCV initialization failed via initLocal, trying System.loadLibrary...")
                try {
                    System.loadLibrary("opencv_java4")
                    Log.d("OpenCvSiftExtractor", "OpenCV loaded via System.loadLibrary")
                } catch (e: UnsatisfiedLinkError) {
                    Log.e("OpenCvSiftExtractor", "OpenCV NOT loaded: ${e.message}")
                }
            } else {
                Log.d("OpenCvSiftExtractor", "OpenCV loaded via initLocal")
            }
        }
    }

    private val sift by lazy { SIFT.create() }

    override fun extract(bitmap: Bitmap): Pair<Array<DoubleArray>, Array<IntArray>> {
        val rgbaMat = Mat()
        Utils.bitmapToMat(bitmap, rgbaMat)
        val grayMat = Mat()
        Imgproc.cvtColor(rgbaMat, grayMat, Imgproc.COLOR_RGBA2GRAY)

        val keypoints = MatOfKeyPoint()
        val descriptors = Mat()
        sift.detectAndCompute(grayMat, Mat(), keypoints, descriptors)

        val kpArray = keypoints.toArray()
        val numKp = kpArray.size

        val positions = Array(numKp) { i -> doubleArrayOf(kpArray[i].pt.x, kpArray[i].pt.y) }
        val descriptorsOut = Array(numKp) { IntArray(128) }

        if (numKp > 0) {
            val row = FloatArray(128)
            for (i in 0 until numKp) {
                descriptors.get(i, 0, row)
                descriptorsOut[i] = quantizeToColmapConvention(row)
            }
        }

        rgbaMat.release(); grayMat.release(); descriptors.release(); keypoints.release()
        return Pair(positions, descriptorsOut)
    }

    /** Reproduit exactement la quantification uint8 de COLMAP : normalisation L2, x512, clip [0,255]. */
    private fun quantizeToColmapConvention(rawDescriptor: FloatArray): IntArray {
        var sumSq = 0.0
        for (v in rawDescriptor) sumSq += (v * v).toDouble()
        val norm = sqrt(sumSq).let { if (it < 1e-12) 1.0 else it }

        return IntArray(128) { c ->
            val normalized = rawDescriptor[c] / norm
            val scaled = (normalized * 512.0).coerceIn(0.0, 255.0)
            round(scaled).toInt()
        }
    }
}
