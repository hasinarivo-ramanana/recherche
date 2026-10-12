package com.thesis.arlocalizer.util

import android.content.ContentValues
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.os.Build
import android.os.Environment
import android.provider.MediaStore
import android.view.View
import java.io.File
import java.io.FileOutputStream
import java.text.SimpleDateFormat
import java.util.Locale

/**
 * Capture la vue de la carte OSM (avec tous les marqueurs de localisation accumules) sous forme
 * d'image PNG, et l'enregistre dans la galerie publique via MediaStore.
 */
object MapSnapshotExporter {

    fun saveViewAsImage(context: Context, view: View, fileName: String = defaultFileName()): String? {
        val bitmap = Bitmap.createBitmap(view.width, view.height, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(bitmap)
        view.draw(canvas)

        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            saveViaMediaStore(context, fileName, bitmap)
        } else {
            saveLegacy(fileName, bitmap)
        }
    }

    private fun saveViaMediaStore(context: Context, fileName: String, bitmap: Bitmap): String? {
        val resolver = context.contentResolver
        val values = ContentValues().apply {
            put(MediaStore.MediaColumns.DISPLAY_NAME, fileName)
            put(MediaStore.MediaColumns.MIME_TYPE, "image/png")
            put(MediaStore.MediaColumns.RELATIVE_PATH, Environment.DIRECTORY_PICTURES + "/UrbanARLocalizer")
        }
        val uri = resolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values) ?: return null
        resolver.openOutputStream(uri)?.use { out -> bitmap.compress(Bitmap.CompressFormat.PNG, 100, out) }
            ?: return null
        return uri.toString()
    }

    private fun saveLegacy(fileName: String, bitmap: Bitmap): String? {
        return try {
            val picturesDir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES)
            val targetDir = File(picturesDir, "UrbanARLocalizer")
            if (!targetDir.exists()) targetDir.mkdirs()
            val file = File(targetDir, fileName)
            FileOutputStream(file).use { out -> bitmap.compress(Bitmap.CompressFormat.PNG, 100, out) }
            file.absolutePath
        } catch (e: Exception) {
            null
        }
    }

    private fun defaultFileName(): String {
        val ts = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(System.currentTimeMillis())
        return "carte_localisation_$ts.png"
    }
}
