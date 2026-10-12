package com.thesis.arlocalizer.util

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.Rect
import android.graphics.YuvImage
import android.media.Image
import java.io.ByteArrayOutputStream

/**
 * Conversion d'une image camera ARCore (format YUV_420_888, obtenue via Frame.acquireCameraImage())
 * en Bitmap RGB exploitable par les moteurs TFLite (AprEngine, YoloDetector) et par PG-PnP.
 *
 * ARCore retourne toujours du YUV_420_888 pour acquireCameraImage() ; la conversion passe par un
 * NV21 intermediaire (format supporte nativement par YuvImage/JPEG) puis un decodage Bitmap.
 * Cette double conversion a un cout CPU non negligeable (~10-20 ms sur Redmi Note 14) mais reste
 * largement compatible avec la periode d'echantillonnage de 5 s de la cascade.
 */
object ImageUtils {

    /**
     * Fait pivoter un Bitmap de `degrees` degres (sens horaire). Utile car Frame.acquireCameraImage()
     * d'ARCore renvoie toujours l'image dans l'orientation NATIVE du capteur (generalement paysage),
     * independamment de l'orientation d'affichage du telephone -- alors que les modeles APR/YOLO ont
     * ete entraines sur des photos prises telephone tenu verticalement (portrait). Sans cette
     * rotation, ces modeles recoivent des images tournees ~90 deg par rapport a leur distribution
     * d'entrainement, ce qui degrade leur precision (contrairement a SIFT, invariant a la rotation,
     * donc ce correctif n'est PAS applique au Niveau 3 -- voir Level3Refiner/LocalizationOrchestrator).
     *
     * ATTENTION : le sens exact (90 vs -90/270) depend de l'orientation de montage du capteur de
     * votre appareil. A VERIFIER EMPIRIQUEMENT (ex. sauvegarder un Bitmap issu de cette fonction sur
     * le disque et l'ouvrir dans une galerie photo : le contenu doit apparaitre a l'endroit, comme
     * si vous regardiez la scene normalement telephone tenu vertical). Ajustez ROTATION_DEGREES dans
     * ArNavigationActivity si l'image ressort tournee dans le mauvais sens.
     */
    fun rotateBitmap(bitmap: Bitmap, degrees: Float): Bitmap {
        if (degrees == 0f) return bitmap
        val matrix = android.graphics.Matrix().apply { postRotate(degrees) }
        return Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
    }

    fun yuv420888ToBitmap(image: Image, jpegQuality: Int = 90): Bitmap {
        require(image.format == ImageFormat.YUV_420_888) { "Format image inattendu : ${image.format}" }

        val nv21 = yuv420888ToNv21(image)
        val yuvImage = YuvImage(nv21, ImageFormat.NV21, image.width, image.height, null)
        val out = ByteArrayOutputStream()
        yuvImage.compressToJpeg(Rect(0, 0, image.width, image.height), jpegQuality, out)
        val jpegBytes = out.toByteArray()
        return BitmapFactory.decodeByteArray(jpegBytes, 0, jpegBytes.size)
    }

    /** Reordonne les plans Y, U, V (avec leurs pas/stride respectifs) en un buffer NV21 (Y puis VU entrelaces). */
    private fun yuv420888ToNv21(image: Image): ByteArray {
        val width = image.width
        val height = image.height
        val ySize = width * height
        val uvSize = width * height / 4

        val nv21 = ByteArray(ySize + uvSize * 2)

        val yPlane = image.planes[0]
        val uPlane = image.planes[1]
        val vPlane = image.planes[2]

        // Plan Y : copie ligne par ligne en tenant compte du rowStride
        var pos = 0
        val yBuffer = yPlane.buffer
        val yRowStride = yPlane.rowStride
        for (row in 0 until height) {
            yBuffer.position(row * yRowStride)
            yBuffer.get(nv21, pos, width)
            pos += width
        }

        // Plans U/V : format NV21 attend l'ordre V,U entrelace, sous-echantillonne par 2
        val chromaHeight = height / 2
        val chromaWidth = width / 2
        val uRowStride = uPlane.rowStride; val uPixelStride = uPlane.pixelStride
        val vRowStride = vPlane.rowStride; val vPixelStride = vPlane.pixelStride
        val uBuffer = uPlane.buffer
        val vBuffer = vPlane.buffer

        for (row in 0 until chromaHeight) {
            for (col in 0 until chromaWidth) {
                val vIndex = row * vRowStride + col * vPixelStride
                val uIndex = row * uRowStride + col * uPixelStride
                nv21[pos++] = vBuffer.get(vIndex)
                nv21[pos++] = uBuffer.get(uIndex)
            }
        }
        return nv21
    }
}
