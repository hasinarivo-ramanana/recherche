package com.thesis.arlocalizer.util

import android.content.ContentValues
import android.content.Context
import android.os.Build
import android.os.Environment
import android.provider.MediaStore
import com.thesis.arlocalizer.model.GeoPose
import com.thesis.arlocalizer.repository.LocalizationRecord
import java.io.File
import java.io.OutputStream
import java.text.SimpleDateFormat
import java.util.Locale

/**
 * Exporte l'historique complet de la cascade de localisation (Niveaux 1/2/3) dans un fichier CSV,
 * avec position (lat, lon), quaternion (qw,qx,qy,qz) et angles d'Euler (yaw,pitch,roll en degres)
 * pour chaque niveau disponible a chaque instant d'echantillonnage.
 *
 * Enregistre dans le dossier public "Downloads" via MediaStore (API 29+) ou en ecriture directe
 * (API < 29, necessite WRITE_EXTERNAL_STORAGE).
 */
object CsvExporter {

    private const val CSV_HEADER =
        "timestamp_iso;timestamp_ms;niveau;latitude;longitude;sigma_pos_m;qw;qx;qy;qz;yaw_deg;pitch_deg;roll_deg\n"

    /** @return l'Uri (ou chemin) du fichier cree, ou null en cas d'echec. */
    fun exportToDownloads(context: Context, records: List<LocalizationRecord>, fileName: String = defaultFileName()): String? {
        val csvContent = buildCsvContent(records)

        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            exportViaMediaStore(context, fileName, csvContent)
        } else {
            exportLegacy(fileName, csvContent)
        }
    }

    private fun buildCsvContent(records: List<LocalizationRecord>): String {
        val sb = StringBuilder(CSV_HEADER)
        val isoFormat = SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss.SSS", Locale.US)

        for (record in records) {
            val isoTs = isoFormat.format(record.timestampMs)
            appendLevelRow(sb, isoTs, record.timestampMs, 1, record.level1)
            appendLevelRow(sb, isoTs, record.timestampMs, 2, record.level2)
            appendLevelRow(sb, isoTs, record.timestampMs, 3, record.level3)
        }
        return sb.toString()
    }

    private fun appendLevelRow(sb: StringBuilder, isoTs: String, tsMs: Long, level: Int, pose: GeoPose?) {
        if (pose == null) return // pas de mesure disponible a ce niveau pour cet instant (repli plus haut dans la cascade)
        val (yaw, pitch, roll) = pose.orientation.toEulerZYXDeg()
        sb.append(isoTs).append(';')
            .append(tsMs).append(';')
            .append(level).append(';')
            .append(pose.latitude).append(';')
            .append(pose.longitude).append(';')
            .append(pose.sigmaPosMeters).append(';')
            .append(pose.orientation.qw).append(';')
            .append(pose.orientation.qx).append(';')
            .append(pose.orientation.qy).append(';')
            .append(pose.orientation.qz).append(';')
            .append(yaw).append(';')
            .append(pitch).append(';')
            .append(roll).append('\n')
    }

    private fun exportViaMediaStore(context: Context, fileName: String, content: String): String? {
        val resolver = context.contentResolver
        val values = ContentValues().apply {
            put(MediaStore.MediaColumns.DISPLAY_NAME, fileName)
            put(MediaStore.MediaColumns.MIME_TYPE, "text/csv")
            put(MediaStore.MediaColumns.RELATIVE_PATH, Environment.DIRECTORY_DOWNLOADS)
        }
        val uri = resolver.insert(MediaStore.Downloads.EXTERNAL_CONTENT_URI, values) ?: return null
        resolver.openOutputStream(uri)?.use { out: OutputStream -> out.write(content.toByteArray(Charsets.UTF_8)) }
            ?: return null
        return uri.toString()
    }

    private fun exportLegacy(fileName: String, content: String): String? {
        return try {
            val downloadsDir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS)
            if (!downloadsDir.exists()) downloadsDir.mkdirs()
            val file = File(downloadsDir, fileName)
            file.writeText(content, Charsets.UTF_8)
            file.absolutePath
        } catch (e: Exception) {
            null
        }
    }

    private fun defaultFileName(): String {
        val ts = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(System.currentTimeMillis())
        return "localisation_cascade_$ts.csv"
    }
}
