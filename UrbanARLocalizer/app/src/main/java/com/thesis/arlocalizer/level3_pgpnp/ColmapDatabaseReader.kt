package com.thesis.arlocalizer.level3_pgpnp

import android.content.Context
import android.database.sqlite.SQLiteDatabase
import android.os.Environment
import okhttp3.OkHttpClient
import okhttp3.Request
import java.io.File
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.TimeUnit

/**
 * Lit les descripteurs SIFT ORIGINAUX stockes par COLMAP dans database.db (format SQLite).
 * Confirme comme seule approche fiable pour un pipeline d'images reelles (descripteurs recalcules
 * autrement ne correspondent pas exactement aux points 3D du nuage triangule).
 *
 * Schema COLMAP (table `descriptors`) : image_id (INTEGER), rows (INTEGER), cols (INTEGER, =128),
 * data (BLOB, uint8 rows x 128). Table `keypoints` : image_id, rows, cols (>=2 : x,y[,scale,orientation]), data (BLOB float32).
 *
 * IMPORTANT (taille du fichier) : pour un `database.db` volumineux (plusieurs centaines de Mo a
 * quelques Go apres extraction SIFT dense), NE PAS l'embarquer dans assets/ -- cela double le
 * stockage utilise sur l'appareil (poids de l'APK + copie obligatoire vers le stockage prive, car
 * SQLiteDatabase ne peut pas ouvrir un fichier directement depuis les assets compresses de l'APK).
 * Utilisez plutot `fromLocalDownloadsFolder` ou `fromUrl` ci-dessous, qui mettent le fichier en
 * cache UNE SEULE FOIS dans le stockage prive, sans jamais le dupliquer dans l'APK.
 *
 * Le constructeur reste prive : passez par l'une des 3 fabriques ci-dessous selon votre strategie
 * de distribution du fichier. Toutes produisent un ColmapDatabaseReader identique en usage
 * (getDescriptors/getKeypoints/close inchanges), donc le reste de l'app (SiftCorrespondenceBuilder,
 * LocalizerServiceLocator) n'a rien a modifier au-dela du point d'appel de la fabrique choisie.
 */
class ColmapDatabaseReader private constructor(private val db: SQLiteDatabase) {

    companion object {
        private const val CACHE_FILE_NAME = "database.db"

        // ============================================================================
        // Strategie 1 (historique) : fichier embarque dans assets/ -- a reserver aux
        // fichiers de petite taille (quelques dizaines de Mo max). Copie vers le stockage
        // prive au premier lancement, comme avant.
        // ============================================================================
        fun fromAssets(context: Context, assetName: String = "database.db"): ColmapDatabaseReader {
            val dst = File(context.filesDir, CACHE_FILE_NAME)
            if (!dst.exists()) {
                context.assets.open(assetName).use { input -> dst.outputStream().use { input.copyTo(it) } }
            }
            return ColmapDatabaseReader(openReadOnly(dst))
        }

        // ============================================================================
        // Strategie 2 : fichier place manuellement par l'utilisateur dans
        // Download/arlocalizer/ (ex. via cable USB / gestionnaire de fichiers), copie une
        // seule fois vers le stockage prive de l'app au premier lancement.
        //
        // PREREQUIS : l'app doit avoir la permission speciale "Acces a tous les fichiers"
        // (MANAGE_EXTERNAL_STORAGE), a accorder manuellement par l'utilisateur : Parametres
        // > Applications > [votre app] > Autorisations > Fichiers et medias > "Autoriser
        // la gestion de tous les fichiers". Voir le README pour le declenchement de cette
        // demande depuis MainActivity (Settings.ACTION_MANAGE_APP_ALL_FILES_ACCESS_PERMISSION).
        // Cette permission est adaptee a un usage recherche/sideload sur votre propre
        // appareil ; une publication Play Store plus large necessiterait de justifier son
        // usage ou de migrer vers le Storage Access Framework (SAF).
        // ============================================================================
        fun fromLocalDownloadsFolder(
            context: Context,
            relativeSubfolder: String = "arlocalizer",
            sourceFileName: String = "database.db"
        ): ColmapDatabaseReader {
            val dst = File(context.filesDir, CACHE_FILE_NAME)
            if (!dst.exists()) {
                @Suppress("DEPRECATION")
                val downloadsDir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS)
                val source = File(downloadsDir, "$relativeSubfolder/$sourceFileName")
                require(source.exists()) {
                    "Fichier introuvable : ${source.absolutePath}\n" +
                    "Verifiez que :\n" +
                    "  1) le fichier a bien ete copie a cet emplacement exact sur l'appareil,\n" +
                    "  2) la permission 'Acces a tous les fichiers' est accordee a l'app " +
                    "(Parametres > Applications > ${context.packageName} > Autorisations > Fichiers et medias)."
                }
                source.copyTo(dst, overwrite = true)
            }
            return ColmapDatabaseReader(openReadOnly(dst))
        }

        // ============================================================================
        // Strategie 3 : telechargement HTTP depuis une URL statique (S3, Google Drive en
        // lien direct, GitHub Releases, votre propre serveur nginx...), mis en cache une
        // seule fois dans le stockage prive. AUCUNE requete reseau n'est faite ensuite --
        // les getDescriptors()/getKeypoints() de la cascade (toutes les 3-5 s) restent
        // strictement locales, comme avec les 2 autres strategies.
        //
        // FONCTION SUSPEND : le telechargement (potentiellement plusieurs minutes pour un
        // fichier de plusieurs centaines de Mo) NE DOIT JAMAIS s'executer sur le thread
        // principal. Appelez-la depuis une coroutine (ex. lifecycleScope.launch(Dispatchers.IO))
        // AVANT d'appeler LocalizerServiceLocator.getOrchestrator(context) -- une fois le
        // fichier present dans context.filesDir, LocalizerServiceLocator peut continuer a
        // utiliser ColmapDatabaseReader.fromAssets(context) tel quel (ou une variante
        // "fromCacheOnly" triviale, cf. remarque en bas de fichier), car le fichier y est deja.
        // ============================================================================
        suspend fun fromUrl(
            context: Context,
            url: String,
            onProgress: ((bytesDownloaded: Long, totalBytes: Long) -> Unit)? = null
        ): ColmapDatabaseReader = kotlinx.coroutines.withContext(kotlinx.coroutines.Dispatchers.IO) {
            val dst = File(context.filesDir, CACHE_FILE_NAME)
            if (!dst.exists()) {
                downloadToFile(url, dst, onProgress)
            }
            ColmapDatabaseReader(openReadOnly(dst))
        }

        /** Telecharge vers un fichier temporaire puis renomme de facon atomique : un fichier
         * .db partiel (app tuee/reseau coupe en cours de telechargement) n'est donc jamais
         * pris pour un fichier complet au prochain lancement. */
        private fun downloadToFile(url: String, destination: File, onProgress: ((Long, Long) -> Unit)?) {
            val client = OkHttpClient.Builder()
                .connectTimeout(30, TimeUnit.SECONDS)
                .readTimeout(0, TimeUnit.MILLISECONDS) // pas de timeout de lecture : fichier volumineux
                .build()

            val tempFile = File(destination.parentFile, destination.name + ".part")
            val request = Request.Builder().url(url).build()

            client.newCall(request).execute().use { response ->
                if (!response.isSuccessful) {
                    throw IOException("Echec du telechargement de $url (HTTP ${response.code})")
                }
                val body = response.body ?: throw IOException("Reponse HTTP vide pour $url")
                val totalBytes = body.contentLength()
                var bytesRead = 0L

                body.byteStream().use { input ->
                    tempFile.outputStream().use { output ->
                        val buffer = ByteArray(64 * 1024)
                        var read: Int
                        while (input.read(buffer).also { read = it } != -1) {
                            output.write(buffer, 0, read)
                            bytesRead += read
                            onProgress?.invoke(bytesRead, totalBytes)
                        }
                    }
                }
            }

            if (!tempFile.renameTo(destination)) {
                // Repli si renameTo echoue (parfois le cas entre systemes de fichiers differents)
                tempFile.copyTo(destination, overwrite = true)
                tempFile.delete()
            }
        }

        private fun openReadOnly(file: File): SQLiteDatabase =
            SQLiteDatabase.openDatabase(file.absolutePath, null, SQLiteDatabase.OPEN_READONLY)
    }

    /** Descripteurs SIFT (uint8, 128-D) pour une image donnee, indexes par point2D_idx. */
    fun getDescriptors(imageId: Long): Array<IntArray> {
        val cursor = db.rawQuery("SELECT rows, cols, data FROM descriptors WHERE image_id = ?", arrayOf(imageId.toString()))
        cursor.use {
            if (!it.moveToFirst()) return emptyArray()
            val rows = it.getInt(0)
            val cols = it.getInt(1) // = 128
            val blob = it.getBlob(2)
            val result = Array(rows) { IntArray(cols) }
            var offset = 0
            for (r in 0 until rows) {
                for (c in 0 until cols) {
                    result[r][c] = blob[offset].toInt() and 0xFF
                    offset++
                }
            }
            return result
        }
    }

    /** Positions 2D (x,y) des points-cles SIFT pour une image donnee, memes indices que getDescriptors. */
    fun getKeypoints(imageId: Long): Array<DoubleArray> {
        val cursor = db.rawQuery("SELECT rows, cols, data FROM keypoints WHERE image_id = ?", arrayOf(imageId.toString()))
        cursor.use {
            if (!it.moveToFirst()) return emptyArray()
            val rows = it.getInt(0)
            val cols = it.getInt(1)
            val blob = it.getBlob(2)
            val bb = ByteBuffer.wrap(blob).order(ByteOrder.LITTLE_ENDIAN)
            val result = Array(rows) { DoubleArray(2) }
            for (r in 0 until rows) {
                val x = bb.float.toDouble()
                val y = bb.float.toDouble()
                result[r][0] = x; result[r][1] = y
                for (c in 2 until cols) bb.float // consomme scale/orientation si presents
            }
            return result
        }
    }

    fun close() = db.close()
}
