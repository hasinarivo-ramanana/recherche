package com.thesis.arlocalizer.orchestrator

import android.graphics.Bitmap
import com.thesis.arlocalizer.model.GeoPose
import com.thesis.arlocalizer.model.Pose3D
import com.thesis.arlocalizer.model.Quaternion
import com.thesis.arlocalizer.poseanchor.PoseAnchorManager
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.MultipartBody
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONObject
import java.io.ByteArrayOutputStream
import java.util.concurrent.TimeUnit

/**
 * Interface commune entre la cascade LOCALE (LocalizationOrchestrator, niv 1/2/3 en Kotlin,
 * conservee comme repli hors-ligne si besoin) et la cascade DEPORTEE (ce fichier), qui delegue
 * tout le calcul niv1/2/3 au serveur Python (deja fiabilise, notamment le niveau 3 dont
 * l'implementation Kotlin presentait une erreur de plusieurs centaines de metres).
 *
 * ArNavigationActivity n'a besoin de connaitre QUE cette interface : basculer entre local et
 * distant se fait en changeant uniquement la construction de `orchestrator` dans onCreate().
 */
interface ILocalizationOrchestrator {
    suspend fun localize(
        bitmap: Bitmap,
        cameraHeadingRad: Double,
        arCorePose: Pose3D? = null,
        focalPxOverride: Double? = null,
        cameraIntrinsics: DoubleArray? = null,
        rotationDegreesForClassifiers: Float = 90f
    ): LocalizationOrchestrator.CascadeResult
}

/**
 * Cascade deportee : POST multipart de la photo (JPEG) + cap vers <baseUrl>/localize,
 * reconstruit un CascadeResult a partir de la reponse JSON pour rester compatible avec
 * tout le code existant d'ArNavigationActivity (applyLocalizationResult, updateArScene,
 * updateGuidance, CascadeLogFormatter, PoseAnchorManager...).
 *
 * @param baseUrl ex. "http://192.168.1.42:8765" (meme hote/port que le WebSocket /ws de
 *                collab -- un seul serveur, cf. server.py).
 */
class RemoteLocalizationOrchestrator(
    private val baseUrl: String,
    private val poseAnchorManager: PoseAnchorManager
) : ILocalizationOrchestrator {

    private val httpClient = OkHttpClient.Builder()
        // Le niveau 3 (SIFT + RANSAC sur le nuage COLMAP) peut prendre plusieurs secondes :
        // un timeout par defaut (10 s) serait trop court. A ajuster selon vos mesures reelles.
        .connectTimeout(10, TimeUnit.SECONDS)
        .readTimeout(30, TimeUnit.SECONDS)
        .writeTimeout(30, TimeUnit.SECONDS)
        .build()

    override suspend fun localize(
        bitmap: Bitmap,
        cameraHeadingRad: Double,
        arCorePose: Pose3D?,
        focalPxOverride: Double?,
        cameraIntrinsics: DoubleArray?,
        rotationDegreesForClassifiers: Float
    ): LocalizationOrchestrator.CascadeResult = withContext(Dispatchers.IO) {
        val tStart = System.nanoTime()

        // Meme rotation que la cascade locale (coherente avec l'entrainement APR/YOLO) avant envoi.
        val bitmapForClassifiers = com.thesis.arlocalizer.util.ImageUtils.rotateBitmap(bitmap, rotationDegreesForClassifiers)
        val jpegBytes = bitmapToJpeg(bitmapForClassifiers, quality = 90)

        val headingDeg = Math.toDegrees(cameraHeadingRad)

        val multipart = MultipartBody.Builder()
            .setType(MultipartBody.FORM)
            .addFormDataPart(
                "image", "frame.jpg",
                jpegBytes.toRequestBody("image/jpeg".toMediaType())
            )
            .addFormDataPart("heading_deg", headingDeg.toString())
            .build()

        val request = Request.Builder()
            .url("$baseUrl/localize")
            .post(multipart)
            .build()

        val json: JSONObject = httpClient.newCall(request).execute().use { response ->
            if (!response.isSuccessful) {
                throw RuntimeException("Echec /localize : HTTP ${response.code} ${response.message}")
            }
            JSONObject(response.body?.string() ?: throw RuntimeException("Reponse /localize vide"))
        }

        val level = json.getInt("level")
        val lat = json.getDouble("lat")
        val lon = json.getDouble("lon")
        val yawDeg = json.getDouble("yaw_deg")
        val sigma = json.optDouble("sigma_pos_m", 0.0)

        // ADAPTER : verifiez la signature reelle de votre classe GeoPose/Quaternion (non fournies
        // ici). Hypothese utilisee : Quaternion(qw,qx,qy,qz) rotation autour de l'axe vertical
        // (meme convention que yawToQuaternionFloatArray dans ArNavigationActivity), et
        // GeoPose(level, latitude, longitude, orientation, sigmaPosMeters).
        val orientation = yawDegToQuaternion(yawDeg)
        val finalPose = GeoPose(level = level, latitude = lat, longitude = lon, orientation = orientation, sigmaPosMeters = sigma)

        fun subPose(obj: JSONObject?, lvl: Int): GeoPose? {
            if (obj == null) return null
            return GeoPose(level = lvl, latitude = obj.getDouble("lat"), longitude = obj.getDouble("lon"), orientation = orientation, sigmaPosMeters = sigma)
        }
        val level1Pose = subPose(json.optJSONObject("level1"), 1) ?: finalPose
        val level2Pose = subPose(json.optJSONObject("level2"), 2)
        val level3Pose = subPose(json.optJSONObject("level3"), 3)

        if (arCorePose != null) {
            poseAnchorManager.onNewLocalization(finalPose, arCorePose)
        }

        val totalElapsedMs = (System.nanoTime() - tStart) / 1_000_000

        LocalizationOrchestrator.CascadeResult(
            finalPose = finalPose,
            level1Pose = level1Pose,
            level2Pose = level2Pose,
            level3Pose = level3Pose,
            level1ElapsedMs = 0L,
            level2ElapsedMs = 0L,
            level2Detections = emptyList(),
            level2UsedFallback = level2Pose == null,
            level3ElapsedMs = 0L,
            level3InlierCount = null,
            level3UsedFallback = level3Pose == null,
            totalElapsedMs = totalElapsedMs
        )
    }

    private fun bitmapToJpeg(bitmap: Bitmap, quality: Int): ByteArray {
        val stream = ByteArrayOutputStream()
        bitmap.compress(Bitmap.CompressFormat.JPEG, quality, stream)
        return stream.toByteArray()
    }

    private fun yawDegToQuaternion(yawDeg: Double): Quaternion {
        val halfRad = Math.toRadians(yawDeg) / 2.0
        return Quaternion(qw = Math.cos(halfRad), qx = 0.0, qy = Math.sin(halfRad), qz = 0.0)
    }
}
