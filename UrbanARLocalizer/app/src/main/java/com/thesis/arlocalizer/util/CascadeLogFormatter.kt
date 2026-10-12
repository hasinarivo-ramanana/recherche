package com.thesis.arlocalizer.util

import com.thesis.arlocalizer.orchestrator.LocalizationOrchestrator
import java.text.SimpleDateFormat
import java.util.Locale

/**
 * Construit, pour un cycle de cascade donne (un clic sur "Localiser ma position"), un rapport
 * texte detaille des 3 niveaux (APR -> YOLO+SIG -> PG-PnP), sur le meme principe que les rapports
 * console de l'application Python de reference (cf. localisation_app-3/display_utils.py + main.py) :
 * un bloc par niveau, avec la pose obtenue, les indicateurs de diagnostic disponibles et le temps
 * d'execution, puis une synthese finale (pose retenue).
 *
 * Ce texte est ajoute a CascadeLogRepository et ecrit dans le fichier Downloads/... .txt
 * (cf. TxtLogExporter) apres chaque clic ; il n'affecte jamais l'affichage de la bande de statut
 * (arStatusText), qui garde exactement son format actuel ("Niveau X — lat, lon").
 */
object CascadeLogFormatter {

    private const val WIDTH = 78
    private val TIMESTAMP_FORMAT = SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.getDefault())

    fun format(sequenceNumber: Int, timestampMs: Long, result: LocalizationOrchestrator.CascadeResult): String {
        val sb = StringBuilder()

        title(sb, "LOCALISATION #$sequenceNumber — ${TIMESTAMP_FORMAT.format(timestampMs)}")

        // ---- Niveau 1 : APR ----
        section(sb, "NIVEAU 1 — Localisation grossiere par APR")
        kv(sb, "Latitude estimee", "%.7f deg".format(result.level1Pose.latitude))
        kv(sb, "Longitude estimee", "%.7f deg".format(result.level1Pose.longitude))
        kv(sb, "Ecart-type de position", "%.1f m".format(result.level1Pose.sigmaPosMeters))
        val (yaw1, pitch1, roll1) = result.level1Pose.orientation.toEulerZYXDeg()
        kv(sb, "Orientation (yaw/pitch/roll)", "%.1f / %.1f / %.1f deg".format(yaw1, pitch1, roll1))
        kv(sb, "Temps d'inference", "${result.level1ElapsedMs} ms")

        // ---- Niveau 2 : YOLO + SIG ----
        section(sb, "NIVEAU 2 — Detection YOLO + requete SIG")
        kv(sb, "Detections YOLO", "${result.level2Detections.size}")
        if (result.level2Detections.isNotEmpty()) {
            val best = result.level2Detections.maxByOrNull { it.confidence }
            kv(sb, "Meilleure detection", "${best?.classe} (confiance ${"%.2f".format(best?.confidence ?: 0f)})")
        }
        if (result.level2Pose != null) {
            kv(sb, "Position affinee (lat, lon)", "%.7f, %.7f".format(result.level2Pose.latitude, result.level2Pose.longitude))
            kv(sb, "Ecart-type de position", "%.1f m".format(result.level2Pose.sigmaPosMeters))
        } else {
            bullet(sb, "Repli sur le Niveau 1 : " + if (result.level2Detections.isEmpty())
                "aucun logo/enseigne detecte" else "aucun candidat SIG retenu")
        }
        kv(sb, "Temps de detection + requete SIG", "${result.level2ElapsedMs} ms")

        // ---- Niveau 3 : PG-PnP ----
        section(sb, "NIVEAU 3 — Affinement par nuage de points COLMAP (PG-PnP)")
        if (result.level3Pose != null && result.level3UsedFallback == false) {
            kv(sb, "Position finale (lat, lon)", "%.7f, %.7f".format(result.level3Pose.latitude, result.level3Pose.longitude))
            kv(sb, "Ecart-type de position", "%.1f m".format(result.level3Pose.sigmaPosMeters))
            kv(sb, "Inliers RANSAC retenus", "${result.level3InlierCount ?: 0}")
        } else {
            bullet(sb, "Repli sur le niveau precedent (Niveau 2/1) : correspondances 2D-3D insuffisantes")
        }
        kv(sb, "Temps d'affinement", "${result.level3ElapsedMs} ms")

        // ---- Synthese ----
        section(sb, "RESULTAT RETENU")
        kv(sb, "Niveau retenu", "${result.finalPose.level}")
        kv(sb, "Position finale (lat, lon)", "%.7f, %.7f".format(result.finalPose.latitude, result.finalPose.longitude))
        kv(sb, "Precision estimee", "±%.1f m".format(result.finalPose.sigmaPosMeters))
        kv(sb, "Temps total du cycle", "${result.totalElapsedMs} ms")
        hr(sb)
        sb.append('\n')

        return sb.toString()
    }

    private fun hr(sb: StringBuilder, char: Char = '-') {
        sb.append(char.toString().repeat(WIDTH)).append('\n')
    }

    private fun title(sb: StringBuilder, text: String) {
        hr(sb, '=')
        sb.append(text).append('\n')
        hr(sb, '=')
    }

    private fun section(sb: StringBuilder, text: String) {
        sb.append('\n').append("-- ").append(text).append(' ')
            .append("-".repeat(maxOf(0, WIDTH - text.length - 4))).append('\n')
    }

    private fun kv(sb: StringBuilder, key: String, value: String) {
        sb.append("  ").append(key.padEnd(38)).append(": ").append(value).append('\n')
    }

    private fun bullet(sb: StringBuilder, text: String) {
        sb.append("  - ").append(text).append('\n')
    }
}
