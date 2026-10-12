package com.thesis.arlocalizer.level1_apr

import android.content.Context
import android.graphics.Bitmap
import com.thesis.arlocalizer.model.GeoPose
import com.thesis.arlocalizer.model.Quaternion
import org.json.JSONObject
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

/**
 * Niveau 1 de la cascade : MobileNet-APR-Campus (deja entraine, fourni en assets/apr_model.tflite).
 *
 * Entree  : image RGB 224x224, normalisee selon les statistiques ImageNet (Eq. norm_image).
 * Sortie  : tete duale -> (phi_tilde, lambda_tilde) in [-0.5,0.5]^2  et  q_raw in R^4 (normalise L2 en sortie).
 *
 * La denormalisation de la position necessite les parametres (moyenne, min, max) calcules sur le jeu
 * d'entrainement (fichier apr_norm_params.json, a fournir a cote du modele -- voir README).
 */
class AprEngine(context: Context, modelAssetName: String = "apr_model.tflite", normParamsAssetName: String = "apr_norm_params.json") {

    companion object {
        private const val INPUT_SIZE = 224
        private val IMAGENET_MEAN = floatArrayOf(0.485f, 0.456f, 0.406f)
        private val IMAGENET_STD = floatArrayOf(0.229f, 0.224f, 0.225f)
    }

    private val interpreter: Interpreter
    private val normParams: NormParams

    /** Index reel des sorties dans le graphe TFLite converti (le convertisseur peut reordonner les
     * sorties par rapport a l'ordre de definition Keras : on les identifie donc par leur taille -
     * 2 valeurs = position, 4 valeurs = quaternion - plutot que de figer un index en dur). */
    private val positionOutputIndex: Int
    private val quaternionOutputIndex: Int

    data class NormParams(
        val latMean: Double, val latMin: Double, val latMax: Double,
        val lonMean: Double, val lonMin: Double, val lonMax: Double
    )

    init {
        interpreter = Interpreter(loadModelFile(context, modelAssetName), Interpreter.Options().apply { setNumThreads(4) })
        val json = JSONObject(context.assets.open(normParamsAssetName).bufferedReader().use { it.readText() })
        normParams = NormParams(
            latMean = json.getDouble("lat_mean"), latMin = json.getDouble("lat_min"), latMax = json.getDouble("lat_max"),
            lonMean = json.getDouble("lon_mean"), lonMin = json.getDouble("lon_min"), lonMax = json.getDouble("lon_max")
        )

        require(interpreter.outputTensorCount == 2) {
            "Modele APR inattendu : ${interpreter.outputTensorCount} sorties (2 attendues : position + quaternion)"
        }
        val shape0 = interpreter.getOutputTensor(0).shape() // ex: [1, 2] ou [1, 4]
        val shape1 = interpreter.getOutputTensor(1).shape()
        when {
            shape0.last() == 2 && shape1.last() == 4 -> { positionOutputIndex = 0; quaternionOutputIndex = 1 }
            shape0.last() == 4 && shape1.last() == 2 -> { positionOutputIndex = 1; quaternionOutputIndex = 0 }
            else -> throw IllegalStateException(
                "Sorties du modele APR inattendues : shape0=${shape0.toList()}, shape1=${shape1.toList()} " +
                "(une sortie de taille 2 [position] et une de taille 4 [quaternion] sont attendues)"
            )
        }
    }

    private fun loadModelFile(context: Context, assetName: String): MappedByteBuffer {
        val afd = context.assets.openFd(assetName)
        FileInputStream(afd.fileDescriptor).use { input ->
            return input.channel.map(FileChannel.MapMode.READ_ONLY, afd.startOffset, afd.declaredLength)
        }
    }

    /** Redimensionne + normalise l'image selon les statistiques ImageNet (Eq. 4.norm_image). */
    private fun preprocess(bitmap: Bitmap): ByteBuffer {
        val resized = Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, true)
        val buffer = ByteBuffer.allocateDirect(4 * INPUT_SIZE * INPUT_SIZE * 3).order(ByteOrder.nativeOrder())
        val pixels = IntArray(INPUT_SIZE * INPUT_SIZE)
        resized.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)
        for (p in pixels) {
            val r = ((p shr 16) and 0xFF) / 255f
            val g = ((p shr 8) and 0xFF) / 255f
            val b = (p and 0xFF) / 255f
            buffer.putFloat((r - IMAGENET_MEAN[0]) / IMAGENET_STD[0])
            buffer.putFloat((g - IMAGENET_MEAN[1]) / IMAGENET_STD[1])
            buffer.putFloat((b - IMAGENET_MEAN[2]) / IMAGENET_STD[2])
        }
        buffer.rewind()
        return buffer
    }

    /**
     * Execute l'inference et retourne la pose geographique brute du Niveau 1.
     * sigmaPosMeters est fixe empiriquement a l'erreur mediane mesuree (~58,64 m, Tableau tab_result1).
     */
    fun estimatePose(bitmap: Bitmap): GeoPose {
        val input = preprocess(bitmap)

        // Deux sorties : position (1x2) et quaternion (1x4). L'ordre reel est detecte a l'init
        // (positionOutputIndex / quaternionOutputIndex) car le convertisseur TFLite peut reordonner
        // les sorties par rapport a l'ordre de definition du modele Keras.
        val posOut = Array(1) { FloatArray(2) }
        val quatOut = Array(1) { FloatArray(4) }
        val outputs = mapOf(positionOutputIndex to posOut, quaternionOutputIndex to quatOut)

        interpreter.runForMultipleInputsOutputs(arrayOf(input), outputs)

        val latTilde = posOut[0][0].toDouble()
        val lonTilde = posOut[0][1].toDouble()
        val lat = denormalize(latTilde, normParams.latMean, normParams.latMin, normParams.latMax)
        val lon = denormalize(lonTilde, normParams.lonMean, normParams.lonMin, normParams.lonMax)

        val q = Quaternion(
            quatOut[0][0].toDouble(), quatOut[0][1].toDouble(),
            quatOut[0][2].toDouble(), quatOut[0][3].toDouble()
        ).normalized()

        return GeoPose(latitude = lat, longitude = lon, orientation = q, level = 1, sigmaPosMeters = 58.64)
    }

    /** Inverse de l'Eq. 4.norm_position : phi = phi_tilde * (max-min) + mean */
    private fun denormalize(normalized: Double, mean: Double, min: Double, max: Double): Double =
        normalized * (max - min) + mean

    fun close() = interpreter.close()
}
