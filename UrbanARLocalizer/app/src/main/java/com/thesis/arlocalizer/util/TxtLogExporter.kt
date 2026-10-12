package com.thesis.arlocalizer.util

import android.content.ContentValues
import android.content.Context
import android.net.Uri
import android.os.Build
import android.os.Environment
import android.provider.MediaStore
import java.io.File
import java.text.SimpleDateFormat
import java.util.Locale

/**
 * Gere le fichier texte detaille de la session de localisation :
 *  - [appendToDownloadsFile] est appele apres CHAQUE clic sur "Localiser ma position" : le fichier
 *    (un seul, cree au premier appel de la session) grossit au fil des clics, dans le dossier public
 *    "Downloads" du telephone.
 *  - [exportToDownloads] permet un export explicite (bouton "Exporter txt" de MapActivity), qui
 *    ecrit l'integralite du contenu accumule (CascadeLogRepository) dans un NOUVEAU fichier horodate --
 *    utile pour obtenir une copie independante a un instant donne, sur le meme principe que
 *    CsvExporter.exportToDownloads.
 *
 * Meme strategie de stockage que CsvExporter/MapSnapshotExporter : MediaStore (API 29+) ou ecriture
 * directe dans Environment.DIRECTORY_DOWNLOADS (API < 29).
 */
object TxtLogExporter {

    // Nom de fichier fixe pour la session en cours (recree a chaque nouveau processus, comme les
    // autres repositories en memoire de l'application) : tous les clics de la session s'accumulent
    // dans ce meme fichier.
    private val sessionFileName: String by lazy {
        val ts = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(System.currentTimeMillis())
        "localisation_cascade_details_$ts.txt"
    }

    // Uri MediaStore du fichier de session (API 29+), cree paresseusement au premier appel.
    private var sessionUri: Uri? = null
    // Chemin du fichier de session (API < 29), cree paresseusement au premier appel.
    private var sessionLegacyFile: File? = null

    /**
     * Ajoute [entryText] a la fin du fichier texte de la session (le cree s'il n'existe pas encore).
     * @return le chemin/Uri du fichier, ou null en cas d'echec.
     */
    @Synchronized
    fun appendToDownloadsFile(context: Context, entryText: String): String? {
        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            appendViaMediaStore(context, entryText)
        } else {
            appendLegacy(entryText)
        }
    }

    /** Export complet explicite (bouton "Exporter txt") : ecrit tout le contenu dans un nouveau fichier. */
    fun exportToDownloads(context: Context, fullText: String, fileName: String = defaultExportFileName()): String? {
        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            exportViaMediaStoreNewFile(context, fileName, fullText)
        } else {
            exportLegacyNewFile(fileName, fullText)
        }
    }

    // ---------------------------------------------------------------------
    // API 29+ (MediaStore, ecriture publique Downloads sans permission speciale)
    // ---------------------------------------------------------------------

    private fun appendViaMediaStore(context: Context, entryText: String): String? {
        val resolver = context.contentResolver
        var uri = sessionUri

        if (uri == null) {
            val values = ContentValues().apply {
                put(MediaStore.MediaColumns.DISPLAY_NAME, sessionFileName)
                put(MediaStore.MediaColumns.MIME_TYPE, "text/plain")
                put(MediaStore.MediaColumns.RELATIVE_PATH, Environment.DIRECTORY_DOWNLOADS)
            }
            uri = resolver.insert(MediaStore.Downloads.EXTERNAL_CONTENT_URI, values) ?: return null
            sessionUri = uri
        }

        return try {
            // Mode "wa" = write + append : necessaire pour faire grossir le fichier a chaque clic
            // plutot que d'ecraser son contenu (contrairement a l'export CSV, qui est un instantane unique).
            resolver.openOutputStream(uri, "wa")?.use { it.write(entryText.toByteArray(Charsets.UTF_8)) }
            uri.toString()
        } catch (e: Exception) {
            null
        }
    }

    private fun exportViaMediaStoreNewFile(context: Context, fileName: String, content: String): String? {
        val resolver = context.contentResolver
        val values = ContentValues().apply {
            put(MediaStore.MediaColumns.DISPLAY_NAME, fileName)
            put(MediaStore.MediaColumns.MIME_TYPE, "text/plain")
            put(MediaStore.MediaColumns.RELATIVE_PATH, Environment.DIRECTORY_DOWNLOADS)
        }
        val uri = resolver.insert(MediaStore.Downloads.EXTERNAL_CONTENT_URI, values) ?: return null
        resolver.openOutputStream(uri)?.use { it.write(content.toByteArray(Charsets.UTF_8)) } ?: return null
        return uri.toString()
    }

    // ---------------------------------------------------------------------
    // API < 29 (ecriture directe, necessite WRITE_EXTERNAL_STORAGE)
    // ---------------------------------------------------------------------

    private fun appendLegacy(entryText: String): String? {
        return try {
            var file = sessionLegacyFile
            if (file == null) {
                val downloadsDir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS)
                if (!downloadsDir.exists()) downloadsDir.mkdirs()
                file = File(downloadsDir, sessionFileName)
                sessionLegacyFile = file
            }
            file.appendText(entryText, Charsets.UTF_8)
            file.absolutePath
        } catch (e: Exception) {
            null
        }
    }

    private fun exportLegacyNewFile(fileName: String, content: String): String? {
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

    private fun defaultExportFileName(): String {
        val ts = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(System.currentTimeMillis())
        return "localisation_cascade_export_$ts.txt"
    }
}
