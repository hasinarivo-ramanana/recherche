package com.thesis.arlocalizer.level2_yolosig

import android.content.Context
import android.graphics.Bitmap
import com.thesis.arlocalizer.model.YoloDetection
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.math.max
import kotlin.math.min

/**
 * Niveau 2 (Etape 1) : detection des logos/enseignes par YOLO (deja entraine, assets/yolo_model.tflite).
 *
 * IMPORTANT - format de sortie reellement observe (et non le format "deja filtre par NMS" suppose
 * initialement) : le modele exporte produit un tenseur BRUT, sans NMS integre, de forme
 * [1, 4+nc, nombre_ancres] (convention "channels-first" standard des exports Ultralytics/YOLOv8/v10
 * en mode detection pure) :
 *   - canaux 0..3   : cx, cy, w, h (coordonnees de boite, en pixels de l'image d'entree du modele)
 *   - canaux 4..4+nc-1 : scores de confiance par classe (deja passes par sigmoid a l'export)
 *   - nombre_ancres : ex. 8400 pour une entree 640x640 (grilles 80x80 + 40x40 + 20x20)
 *
 * nc (nombre de classes) est deduit automatiquement de la forme du tenseur de sortie
 * (nc = canaux - 4), et DOIT correspondre a la taille de `classNames` fournie au constructeur.
 * Le decodage (transposition, seuillage, conversion cxcywh -> xyxy, NMS manuelle) est fait ici
 * car ce format brut ne contient aucun filtrage prealable.
 */
class YoloDetector(
    context: Context,
    modelAssetName: String = "yolo_model.tflite",
    private val classNames: List<String> = listOf("misa", "buvette", "bni", "bfv", "boa", "pharmacie"),
    private val confidenceThreshold: Float = 0.5f,
    private val iouThreshold: Float = 0.45f
) {
    companion object {
        private const val INPUT_SIZE = 640
    }

    private val interpreter = Interpreter(loadModelFile(context, modelAssetName), Interpreter.Options().apply { setNumThreads(4) })

    /** Nombre de canaux (4 + nc) et nombre d'ancres, lus directement sur le tenseur de sortie reel. */
    private val outputChannels: Int
    private val numAnchors: Int
    private val numClasses: Int

    init {
        val outShape = interpreter.getOutputTensor(0).shape() // attendu : [1, 4+nc, num_ancres]
        require(outShape.size == 3 && outShape[0] == 1) {
            "Forme de sortie YOLO inattendue : ${outShape.toList()} (attendu [1, 4+nc, num_ancres])"
        }
        outputChannels = outShape[1]
        numAnchors = outShape[2]
        numClasses = outputChannels - 4

        require(numClasses == classNames.size) {
            "Incoherence de classes : le modele produit $numClasses classe(s) (canaux=$outputChannels, " +
            "donc nc = canaux - 4 = $numClasses), mais classNames en contient ${classNames.size} " +
            "(${classNames}). Corrigez la liste `classNames` pour qu'elle corresponde exactement " +
            "aux classes utilisees lors de l'entrainement/export du modele (dans le meme ordre)."
        }
    }

    private fun loadModelFile(context: Context, assetName: String): MappedByteBuffer {
        val afd = context.assets.openFd(assetName)
        FileInputStream(afd.fileDescriptor).use { input ->
            return input.channel.map(FileChannel.MapMode.READ_ONLY, afd.startOffset, afd.declaredLength)
        }
    }

    private fun preprocess(bitmap: Bitmap): ByteBuffer {
        val resized = Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, true)
        val buffer = ByteBuffer.allocateDirect(4 * INPUT_SIZE * INPUT_SIZE * 3).order(ByteOrder.nativeOrder())
        val pixels = IntArray(INPUT_SIZE * INPUT_SIZE)
        resized.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)
        for (p in pixels) {
            buffer.putFloat(((p shr 16) and 0xFF) / 255f)
            buffer.putFloat(((p shr 8) and 0xFF) / 255f)
            buffer.putFloat((p and 0xFF) / 255f)
        }
        buffer.rewind()
        return buffer
    }

    /** Retourne les detections dans le repere pixel de l'image ORIGINALE (avant redimensionnement). */
    fun detect(bitmap: Bitmap): List<YoloDetection> {
        val input = preprocess(bitmap)

        // Tenseur de sortie brut, forme reelle [1, outputChannels, numAnchors] (channels-first).
        val output = Array(1) { Array(outputChannels) { FloatArray(numAnchors) } }
        interpreter.run(input, output)

        val scaleX = bitmap.width.toFloat() / INPUT_SIZE
        val scaleY = bitmap.height.toFloat() / INPUT_SIZE

        val raw = mutableListOf<YoloDetection>()
        for (a in 0 until numAnchors) {
            // Recherche de la classe la plus probable pour cette ancre (canaux 4..4+nc-1)
            var bestClassId = -1
            var bestScore = confidenceThreshold
            for (c in 0 until numClasses) {
                val score = output[0][4 + c][a]
                if (score > bestScore) {
                    bestScore = score
                    bestClassId = c
                }
            }
            if (bestClassId < 0) continue // aucune classe au-dessus du seuil pour cette ancre

            val cx = output[0][0][a] * scaleX
            val cy = output[0][1][a] * scaleY
            val w = output[0][2][a] * scaleX
            val h = output[0][3][a] * scaleY

            raw.add(
                YoloDetection(
                    classe = classNames[bestClassId],
                    u = cx - w / 2f, v = cy - h / 2f,
                    widthPx = w, heightPx = h,
                    confidence = bestScore
                )
            )
        }

        return nonMaxSuppression(raw, iouThreshold)
    }

    private fun nonMaxSuppression(detections: List<YoloDetection>, iouThresh: Float): List<YoloDetection> {
        val sorted = detections.sortedByDescending { it.confidence }.toMutableList()
        val kept = mutableListOf<YoloDetection>()
        while (sorted.isNotEmpty()) {
            val best = sorted.removeAt(0)
            kept.add(best)
            sorted.removeAll { iou(best, it) > iouThresh && it.classe == best.classe }
        }
        return kept
    }

    private fun iou(a: YoloDetection, b: YoloDetection): Float {
        val x1 = max(a.u, b.u); val y1 = max(a.v, b.v)
        val x2 = min(a.u + a.widthPx, b.u + b.widthPx)
        val y2 = min(a.v + a.heightPx, b.v + b.heightPx)
        val inter = max(0f, x2 - x1) * max(0f, y2 - y1)
        val union = a.widthPx * a.heightPx + b.widthPx * b.heightPx - inter
        return if (union <= 0f) 0f else inter / union
    }

    fun close() = interpreter.close()
}
