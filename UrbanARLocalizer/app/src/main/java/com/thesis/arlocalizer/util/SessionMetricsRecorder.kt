package com.thesis.arlocalizer.util

import android.content.Context
import android.util.Log
import com.thesis.arlocalizer.model.Pose3D
import com.thesis.arlocalizer.orchestrator.LocalizationOrchestrator
import com.thesis.arlocalizer.poseanchor.PoseAnchorManager
import java.text.SimpleDateFormat
import java.util.Locale
import kotlin.math.abs
import kotlin.math.hypot
import kotlin.math.pow
import kotlin.math.sqrt

/**
 * Collecte, pendant une session AR, les metriques du tableau 5.3 (erreur initiale, drift apres
 * 100 m / 5 min, FPS, temps de localisation, recalibrations/min) avec statistiques completes
 * (n, moyenne, ecart-type, mediane, min, max), puis ecrit automatiquement un CSV dans Download/
 * a la fin de la session (aucun bouton).
 *
 * Conversion pixels <-> metres :  e_m ~ e_px * d / f   <=>   e_px = e_m * f / d
 *   f = focale en pixels, d = distance au plan observe.
 *
 * Reference de "verite" : la pose issue de la cascade (niveau >= MIN_REFERENCE_LEVEL).
 */
object SessionMetricsRecorder {

    private const val TAG = "SessionMetrics"

    const val DEFAULT_FOCAL_PX = 1400.0          // meme valeur par defaut que Level2Refiner
    const val DEFAULT_PLANE_DISTANCE_M = 10.0    // d : distance typique au plan observe (a ajuster)
    private const val MIN_REFERENCE_LEVEL = 3    // seules les poses Niveau 3 (~2 m) servent de reference
    private const val INITIAL_SAMPLES = 3
    private const val DRIFT_DIST_TARGET_M = 100.0
    private const val DRIFT_DIST_TOL_M = 20.0
    private const val DRIFT_TIME_TARGET_S = 300.0
    private const val DRIFT_TIME_TOL_S = 30.0

    private data class Sample(
        val tSinceInitS: Double, val distSinceInitM: Double, val level: Int, val sigmaM: Double,
        val errWithM: Double, val errWithoutM: Double,
        val errWithPx: Double, val errWithoutPx: Double,
        val focalPx: Double, val planeDistM: Double, val locMs: Long
    )

    private val lock = Any()
    private var active = false
    private var poseAnchorEnabled = true
    private var startMs = 0L

    // FPS
    private var firstFrameNs = 0L
    private var binStartNs = 0L
    private var binFrames = 0
    private val fpsPerSecond = mutableListOf<Double>()
    private var frameCount = 0L

    // Distance parcourue (plan horizontal ARCore : X/Z, Y vertical)
    private var lastArX = Double.NaN
    private var lastArZ = Double.NaN
    private var pathLengthM = 0.0
    private var pathAtInitM = 0.0

    // PoseAnchor
    private var snapshotBefore: Pose3D? = null
    private var baselineTransform: Pose3D? = null   // T_AR->WGS fige a l'initialisation (sans correction)
    private var originLat = 0.0
    private var originLon = 0.0
    private var initMs = 0L
    private var recalibrations = 0

    private val samples = mutableListOf<Sample>()
    private val locTimesMs = mutableListOf<Double>()

    // ------------------------------------------------------------------ API publique

    /** A appeler une fois au debut de la session (ArNavigationActivity.onCreate). */
    fun startSession(poseAnchorEnabled: Boolean = true): Unit = synchronized(lock) {
        resetState()
        this.poseAnchorEnabled = poseAnchorEnabled
        startMs = System.currentTimeMillis()
        active = true
    }

    /** A appeler a CHAQUE frame AR avec la pose camera ARCore courante (Frame.camera.pose), ou null si tracking perdu. */
    fun onFrame(liveArCorePose: Pose3D?): Unit = synchronized(lock) {
        if (!active) return
        val now = System.nanoTime()
        if (firstFrameNs == 0L) { firstFrameNs = now; binStartNs = now }
        frameCount++; binFrames++
        if (now - binStartNs >= 1_000_000_000L) {
            fpsPerSecond.add(binFrames * 1e9 / (now - binStartNs))
            binStartNs = now; binFrames = 0
        }
        if (liveArCorePose != null) {
            val x = liveArCorePose.translation[0]; val z = liveArCorePose.translation[2]
            if (!lastArX.isNaN()) {
                val step = hypot(x - lastArX, z - lastArZ)
                if (step > 0.005) pathLengthM += step   // ignore le bruit < 5 mm
            }
            lastArX = x; lastArZ = z
        }
    }

    /** A appeler JUSTE AVANT orchestrator.localize(...) : fige la transformation T_AR->WGS filtree. */
    fun beforeLocalize(pam: PoseAnchorManager): Unit = synchronized(lock) {
        if (!active) return
        snapshotBefore = pam.currentTransform()
    }

    /**
     * A appeler JUSTE APRES orchestrator.localize(...).
     * @param arCorePoseAtCapture pose camera ARCore au moment de la capture (la meme que celle passee a localize)
     * @param focalPx focale en pixels (cameraIntrinsics[0] = fx), sinon DEFAULT_FOCAL_PX
     * @param planeDistanceM distance d au plan observe (hit-test ARCore au centre de l'ecran)
     */
    fun afterLocalize(
        result: LocalizationOrchestrator.CascadeResult,
        arCorePoseAtCapture: Pose3D?,
        pam: PoseAnchorManager,
        focalPx: Double? = null,
        planeDistanceM: Double = DEFAULT_PLANE_DISTANCE_M
    ): Unit = synchronized(lock) {
        if (!active) return
        locTimesMs.add(result.totalElapsedMs.toDouble())
        val arPose = arCorePoseAtCapture ?: return   // pas de recalage AR ce cycle (cf. orchestrateur)
        recalibrations++

        val pose = result.finalPose
        val prior = snapshotBefore
        snapshotBefore = null

        // Premiere localisation : PoseAnchor vient d'etre initialise (origine ENU = cette position)
        if (prior == null) {
            originLat = pose.latitude; originLon = pose.longitude
            baselineTransform = pam.currentTransform()
            initMs = System.currentTimeMillis()
            pathAtInitM = pathLengthM
            return
        }
        val baseline = baselineTransform ?: return

        val measured = GeoUtils.geodeticToLocalMeters(pose.latitude, pose.longitude, originLat, originLon)
        val predWith = prior.compose(arPose).translation
        val predWithout = baseline.compose(arPose).translation

        val errWithM = hypot(predWith[0] - measured[0], predWith[1] - measured[1])
        val errWithoutM = hypot(predWithout[0] - measured[0], predWithout[1] - measured[1])

        val f = focalPx ?: DEFAULT_FOCAL_PX
        val d = planeDistanceM
        samples.add(
            Sample(
                tSinceInitS = (System.currentTimeMillis() - initMs) / 1000.0,
                distSinceInitM = pathLengthM - pathAtInitM,
                level = pose.level, sigmaM = pose.sigmaPosMeters,
                errWithM = errWithM, errWithoutM = errWithoutM,
                errWithPx = errWithM * f / d,          // e_px = e_m * f / d
                errWithoutPx = errWithoutM * f / d,
                focalPx = f, planeDistM = d, locMs = result.totalElapsedMs
            )
        )
    }

    /**
     * A appeler a la fin de la session (onDestroy). Ecrit le CSV dans Download/ et remet
     * l'etat a zero. Idempotent : sans effet si aucune session n'est active.
     * @return chemin/Uri du fichier cree, ou null.
     */
    fun endSession(context: Context): String? = synchronized(lock) {
        if (!active) return null
        active = false
        val content = buildCsv()
        val ts = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(startMs)
        val path = try {
            CsvExporter.exportTextToDownloads(context, "metriques_session_$ts.csv", content)
        } catch (e: Exception) {
            Log.e(TAG, "Echec export metriques", e); null
        }
        Log.i(TAG, "Metriques session exportees : $path")
        resetState()
        path
    }

    // ------------------------------------------------------------------ construction du CSV

    private fun buildCsv(): String {
        val sb = StringBuilder()
        val nowMs = System.currentTimeMillis()
        val durationS = (nowMs - startMs) / 1000.0
        val ref = samples.filter { it.level >= MIN_REFERENCE_LEVEL }
        val sinceInitMin = if (initMs > 0) (nowMs - initMs) / 60000.0 else 0.0

        // ---- Meta ----
        sb.append("# META\n")
        sb.append("cle;valeur\n")
        sb.append("debut_session_iso;").append(SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss", Locale.US).format(startMs)).append('\n')
        sb.append("duree_session_s;").append(f(durationS)).append('\n')
        sb.append("distance_parcourue_m;").append(f(pathLengthM - pathAtInitM)).append('\n')
        sb.append("poseanchor_actif;").append(poseAnchorEnabled).append('\n')
        sb.append("frames_total;").append(frameCount).append('\n')
        sb.append("echantillons_total;").append(samples.size).append('\n')
        sb.append("echantillons_reference_niveau_${MIN_REFERENCE_LEVEL}+;").append(ref.size).append('\n')
        sb.append("formule;e_px = e_m * f / d\n")
        if (samples.isNotEmpty()) {
            sb.append("focale_px_moyenne;").append(f(samples.map { it.focalPx }.average())).append('\n')
            sb.append("distance_plan_d_m;").append(f(samples.map { it.planeDistM }.average())).append('\n')
        }

        // ---- Resume statistique (tableau 5.3) ----
        sb.append("\n# RESUME (pour le tableau 5.3)\n")
        sb.append("metrique;condition;unite;n;moyenne;ecart_type;mediane;min;max\n")

        val initial = ref.take(INITIAL_SAMPLES)
        sb.append(statsRow("erreur_initiale", "avec_poseanchor", "px", initial.map { it.errWithPx }))
        sb.append(statsRow("erreur_initiale", "sans_poseanchor", "px", initial.map { it.errWithoutPx }))

        val w100 = ref.filter { abs(it.distSinceInitM - DRIFT_DIST_TARGET_M) <= DRIFT_DIST_TOL_M }
        sb.append(statsRow("drift_100m", "avec_poseanchor", "px", w100.map { it.errWithPx }))
        sb.append(statsRow("drift_100m", "sans_poseanchor", "px", w100.map { it.errWithoutPx }))

        val w5 = ref.filter { abs(it.tSinceInitS - DRIFT_TIME_TARGET_S) <= DRIFT_TIME_TOL_S }
        sb.append(statsRow("drift_5min", "avec_poseanchor", "px", w5.map { it.errWithPx }))
        sb.append(statsRow("drift_5min", "sans_poseanchor", "px", w5.map { it.errWithoutPx }))

        sb.append(statsRow("fps", "mesure", "img/s", fpsPerSecond))
        sb.append(statsRow("temps_localisation", "mesure", "ms", locTimesMs))

        val recalPerMin = if (sinceInitMin > 0) recalibrations / sinceInitMin else Double.NaN
        sb.append(statsRow("recalibrations_par_minute", "mesure", "1/min", listOf(recalPerMin).filter { !it.isNaN() }))

        // ---- Detail par recalibration ----
        sb.append("\n# DETAIL\n")
        sb.append("t_depuis_init_s;distance_depuis_init_m;niveau;sigma_pos_m;")
        sb.append("err_avec_m;err_sans_m;err_avec_px;err_sans_px;focale_px;distance_plan_m;temps_localisation_ms\n")
        for (s in samples) {
            sb.append(f(s.tSinceInitS)).append(';').append(f(s.distSinceInitM)).append(';')
                .append(s.level).append(';').append(f(s.sigmaM)).append(';')
                .append(f(s.errWithM)).append(';').append(f(s.errWithoutM)).append(';')
                .append(f(s.errWithPx)).append(';').append(f(s.errWithoutPx)).append(';')
                .append(f(s.focalPx)).append(';').append(f(s.planeDistM)).append(';')
                .append(s.locMs).append('\n')
        }
        return sb.toString()
    }

    private fun statsRow(metric: String, cond: String, unit: String, v: List<Double>): String {
        if (v.isEmpty()) return "$metric;$cond;$unit;0;NA;NA;NA;NA;NA\n"
        val mean = v.average()
        val std = if (v.size > 1) sqrt(v.sumOf { (it - mean).pow(2) } / (v.size - 1)) else 0.0
        val sorted = v.sorted()
        val median = if (sorted.size % 2 == 1) sorted[sorted.size / 2]
        else (sorted[sorted.size / 2 - 1] + sorted[sorted.size / 2]) / 2.0
        return "$metric;$cond;$unit;${v.size};${f(mean)};${f(std)};${f(median)};${f(sorted.first())};${f(sorted.last())}\n"
    }

    private fun f(x: Double): String = String.format(Locale.US, "%.3f", x)

    private fun resetState() {
        active = false; startMs = 0L
        firstFrameNs = 0L; binStartNs = 0L; binFrames = 0; frameCount = 0L
        fpsPerSecond.clear()
        lastArX = Double.NaN; lastArZ = Double.NaN; pathLengthM = 0.0; pathAtInitM = 0.0
        snapshotBefore = null; baselineTransform = null
        originLat = 0.0; originLon = 0.0; initMs = 0L; recalibrations = 0
        samples.clear(); locTimesMs.clear()
    }
}
