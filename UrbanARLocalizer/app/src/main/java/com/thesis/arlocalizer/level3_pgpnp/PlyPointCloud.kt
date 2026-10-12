package com.thesis.arlocalizer.level3_pgpnp

import android.content.Context
import com.thesis.arlocalizer.model.Point3D
import java.io.BufferedInputStream
import java.io.InputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Chargeur du nuage de points COLMAP exporte au format PLY (points3D.ply, deja disponible cote utilisateur).
 * Supporte les en-tetes ASCII et binary_little_endian, avec les proprietes standards COLMAP :
 * x, y, z (float/double), [nx, ny, nz], [red, green, blue] (uchar).
 *
 * Les coordonnees brutes du fichier sont exprimees dans le repere du modele SfM (arbitraire, PAS
 * en metres). Elles sont systematiquement converties en ENU metrique local ici, via la
 * transformation de similarite GeorefCalibration.aligner() (portee depuis
 * cache/colmap_to_enu_transform.json du script Python de reference) -- voir SfmToEnuAligner.
 */
object PlyPointCloud {

    /**
     * @param aligner transformation de similarite SfM -> ENU a appliquer a chaque point au chargement
     *   (cf. GeorefCalibration.aligner()). Passer SfmToEnuAligner.identity() UNIQUEMENT si points3D.ply
     *   a deja ete georegistre en amont (ex. via `colmap model_aligner`) -- ce qui n'est PAS le cas du
     *   modele actuel (cf. cache/colmap_to_enu_transform.json genere par le script Python de reference).
     */
    fun loadFromAssets(
        context: Context,
        assetName: String = "points3D.ply",
        aligner: SfmToEnuAligner = GeorefCalibration.aligner()
    ): List<Point3D> {
        context.assets.open(assetName).use { input ->
            return parse(BufferedInputStream(input), aligner)
        }
    }

    private fun parse(input: InputStream, aligner: SfmToEnuAligner): List<Point3D> {
        val headerLines = mutableListOf<String>()
        var line: String
        var vertexCount = 0
        var format = "ascii"
        val properties = mutableListOf<Pair<String, String>>() // (type, name)

        // Lecture ligne par ligne de l'en-tete (jusqu'a "end_header")
        val byteBuffer = mutableListOf<Byte>()
        while (true) {
            val b = input.read()
            if (b == -1) break
            if (b == '\n'.code) {
                line = String(byteBuffer.toByteArray()).trim()
                byteBuffer.clear()
                headerLines.add(line)
                when {
                    line.startsWith("format") -> format = line.split(" ")[1]
                    line.startsWith("element vertex") -> vertexCount = line.split(" ")[2].trim().toInt()
                    line.startsWith("property") -> {
                        val parts = line.split(" ")
                        properties.add(parts[1] to parts[2])
                    }
                    line == "end_header" -> break
                }
            } else {
                byteBuffer.add(b.toByte())
            }
        }

        val points = ArrayList<Point3D>(vertexCount)
        val xIdx = properties.indexOfFirst { it.second == "x" }
        val yIdx = properties.indexOfFirst { it.second == "y" }
        val zIdx = properties.indexOfFirst { it.second == "z" }

        if (format == "ascii") {
            val reader = input.bufferedReader()
            var count = 0
            var l: String?
            while (count < vertexCount) {
                l = reader.readLine() ?: break
                val tokens = l.trim().split(Regex("\\s+"))
                if (tokens.size < properties.size) continue
                val x = tokens[xIdx].toDouble()
                val y = tokens[yIdx].toDouble()
                val z = tokens[zIdx].toDouble()
                val enu = aligner.applyXYZ(x, y, z)
                points.add(Point3D(id = count.toLong(), x = enu[0], y = enu[1], z = enu[2]))
                count++
            }
        } else {
            // binary_little_endian (le plus courant en sortie de COLMAP)
            val bytesPerVertex = properties.sumOf { byteSizeOf(it.first) }
            val buf = ByteArray(bytesPerVertex)
            for (i in 0 until vertexCount) {
                var readTotal = 0
                while (readTotal < bytesPerVertex) {
                    val r = input.read(buf, readTotal, bytesPerVertex - readTotal)
                    if (r == -1) break
                    readTotal += r
                }
                if (readTotal < bytesPerVertex) break
                val bb = ByteBuffer.wrap(buf).order(ByteOrder.LITTLE_ENDIAN)
                var x = 0.0; var y = 0.0; var z = 0.0
                var offset = 0
                for ((idx, prop) in properties.withIndex()) {
                    val (type, _) = prop
                    val v = readTyped(bb, type)
                    when (idx) {
                        xIdx -> x = v
                        yIdx -> y = v
                        zIdx -> z = v
                    }
                    offset += byteSizeOf(type)
                }
                val enu = aligner.applyXYZ(x, y, z)
                points.add(Point3D(id = i.toLong(), x = enu[0], y = enu[1], z = enu[2]))
            }
        }
        return points
    }

    private fun byteSizeOf(type: String): Int = when (type) {
        "float", "float32", "int", "int32", "uint", "uint32" -> 4
        "double", "float64" -> 8
        "uchar", "uint8", "char", "int8" -> 1
        "short", "int16", "ushort", "uint16" -> 2
        else -> 4
    }

    private fun readTyped(bb: ByteBuffer, type: String): Double = when (type) {
        "float", "float32" -> bb.float.toDouble()
        "double", "float64" -> bb.double
        "int", "int32" -> bb.int.toDouble()
        "uint", "uint32" -> bb.int.toDouble()
        "uchar", "uint8" -> (bb.get().toInt() and 0xFF).toDouble()
        "char", "int8" -> bb.get().toDouble()
        "short", "int16" -> bb.short.toDouble()
        "ushort", "uint16" -> (bb.short.toInt() and 0xFFFF).toDouble()
        else -> bb.float.toDouble()
    }
}

/**
 * Transformation de similarite (echelle + rotation + translation) alignant le repere arbitraire du
 * modele SfM COLMAP sur le repere ENU metrique local (origine = un point de reference du campus).
 * Calibree hors-ligne via des points de controle GPS (Umeyama) -- voir GeorefCalibration pour les
 * valeurs actuellement utilisees par l'application.
 */
data class SfmToEnuAligner(val scale: Double, val rotation: DoubleArray, val translation: DoubleArray) {
    fun apply(p: Point3D): DoubleArray = applyXYZ(p.x, p.y, p.z)

    /** Meme transformation, pour un point brut (x,y,z) sans avoir a construire un Point3D factice
     * (utilise par ColmapImageIndexLoader pour aligner les centres de camera). */
    fun applyXYZ(x: Double, y: Double, z: Double): DoubleArray {
        val rv = doubleArrayOf(
            rotation[0] * x + rotation[1] * y + rotation[2] * z,
            rotation[3] * x + rotation[4] * y + rotation[5] * z,
            rotation[6] * x + rotation[7] * y + rotation[8] * z
        )
        return doubleArrayOf(
            scale * rv[0] + translation[0],
            scale * rv[1] + translation[1],
            scale * rv[2] + translation[2]
        )
    }

    companion object {
        fun identity() = SfmToEnuAligner(1.0, doubleArrayOf(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0), doubleArrayOf(0.0, 0.0, 0.0))
    }
}
