package com.thesis.arlocalizer.level3_pgpnp

import android.content.Context
import com.thesis.arlocalizer.util.GeoUtils

/**
 * Construit un index spatial des images de reference COLMAP (images.txt), permettant de restreindre
 * l'appariement SIFT aux images proches d'une position APR donnee (au lieu d'un matching exhaustif
 * sur l'ensemble des images du campus).
 *
 * Format attendu (genere par COLMAP), DEUX lignes par image :
 *   IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
 *   X Y POINT3D_ID  X Y POINT3D_ID ...   (liste des observations 2D, ignoree ici)
 *
 * Chaque image COLMAP stocke sa pose (rotation R, translation t) telle que X_camera = R*X_monde + t
 * (convention monde->camera standard de COLMAP). Le centre de la camera dans le repere du modele SfM
 * est donc C = -R^T * t. Ce centre est ensuite transforme dans le repere ENU via l'alignement SfM->ENU
 * deja utilise pour le nuage de points (SfmToEnuAligner -- identite si points3D est deja georegistre
 * via `colmap model_aligner`), puis reconverti en (lat, lon) approximatif pour l'indexation spatiale.
 *
 * Placez le fichier resultant dans app/src/main/assets/images.txt.
 * (Pour l'obtenir depuis un modele binaire : colmap model_converter --output_type TXT, comme pour points3D.txt)
 *
 * IMPORTANT (taille du fichier) : la ligne "points2D" de chaque image (potentiellement des milliers
 * de triplets X,Y,POINT3D_ID) N'EST JAMAIS UTILISEE ici -- seule la ligne de pose est lue. Sur un
 * jeu de donnees avec un `max_num_features` eleve, cette ligne inutile peut representer 90%+ du
 * poids du fichier (voir scripts/strip_images_txt.py pour la retirer a la source, recommande avant
 * de placer le fichier dans assets/). Le chargement ci-dessous lit en outre le fichier LIGNE PAR
 * LIGNE (useLines, paresseux) plutot que de charger tout le fichier en List<String> d'un coup
 * (l'ancien `reader.readLines()` pouvait doubler la consommation memoire d'un gros fichier texte,
 * risquant un OutOfMemoryError cumule avec les modeles TFLite et le nuage de points deja en memoire).
 */
object ColmapImageIndexLoader {

    data class ReferenceImage(val imageId: Long, val name: String, val lat: Double, val lon: Double)

    fun loadFromAssets(
        context: Context,
        assetName: String = "images.txt",
        aligner: SfmToEnuAligner,
        originLat: Double,
        originLon: Double
    ): List<ReferenceImage> {
        val result = mutableListOf<ReferenceImage>()
        context.assets.open(assetName).bufferedReader().useLines { lineSequence ->
            val state = ParseState()
            for (rawLine in lineSequence) {
                processLine(rawLine, state, aligner, originLat, originLon)?.let { result.add(it) }
            }
        }
        return result
    }

    /** Expose separement pour les tests unitaires (pas de dependance Android Context). */
    fun parse(
        lines: List<String>,
        aligner: SfmToEnuAligner,
        originLat: Double,
        originLon: Double
    ): List<ReferenceImage> {
        val result = mutableListOf<ReferenceImage>()
        val state = ParseState()
        for (rawLine in lines) {
            processLine(rawLine, state, aligner, originLat, originLon)?.let { result.add(it) }
        }
        return result
    }

    /**
     * Etat mutable de l'automate d'alternance pose/points2D, partage entre la version "streaming"
     * (loadFromAssets) et la version "liste en memoire" (parse, pour les tests). Dans images.txt,
     * les lignes non-commentaires alternent STRICTEMENT : une ligne de pose, puis une ligne de
     * points 2D -- laquelle peut etre VIDE si l'image n'a aucune observation (ou si le fichier a
     * ete allege par strip_images_txt.py). On ne saute donc les lignes vides que tant qu'on est
     * encore dans l'en-tete (avant la 1ere pose) ; une fois les donnees commencees, l'alternance
     * doit etre respectee meme sur une ligne vide.
     */
    private class ParseState {
        var expectPoseLine = true
        var dataStarted = false
    }

    /** Traite une ligne ; retourne une ReferenceImage si cette ligne etait une ligne de pose valide. */
    private fun processLine(
        rawLine: String,
        state: ParseState,
        aligner: SfmToEnuAligner,
        originLat: Double,
        originLon: Double
    ): ReferenceImage? {
        if (!state.dataStarted) {
            val trimmed = rawLine.trim()
            if (trimmed.isEmpty() || trimmed.startsWith("#")) return null
            state.dataStarted = true
        }

        if (!state.expectPoseLine) {
            state.expectPoseLine = true // ligne points2D (ignoree, meme vide) : on revient a une pose ensuite
            return null
        }

        val tokens = rawLine.trim().split(Regex("\\s+")).filter { it.isNotEmpty() }
        if (tokens.size < 10) { state.expectPoseLine = false; return null }

        val imageId = tokens[0].toLong()
        val qw = tokens[1].toDouble(); val qx = tokens[2].toDouble()
        val qy = tokens[3].toDouble(); val qz = tokens[4].toDouble()
        val tx = tokens[5].toDouble(); val ty = tokens[6].toDouble(); val tz = tokens[7].toDouble()
        val name = tokens[9]

        val r = quaternionToRotationMatrix(qw, qx, qy, qz)
        // Centre camera dans le repere SfM : C = -R^T * t
        val rt = transpose3(r)
        val cx = -(rt[0] * tx + rt[1] * ty + rt[2] * tz)
        val cy = -(rt[3] * tx + rt[4] * ty + rt[5] * tz)
        val cz = -(rt[6] * tx + rt[7] * ty + rt[8] * tz)

        // Alignement SfM -> ENU (identite si deja georegistre par model_aligner)
        val enuPoint = aligner.applyXYZ(cx, cy, cz)

        // Reconversion ENU (approximation plane locale) -> (lat, lon)
        val dLat = enuPoint[1] / GeoUtils.EARTH_RADIUS_M
        val dLon = enuPoint[0] / (GeoUtils.EARTH_RADIUS_M * kotlin.math.cos(Math.toRadians(originLat)))
        val lat = originLat + Math.toDegrees(dLat)
        val lon = originLon + Math.toDegrees(dLon)

        state.expectPoseLine = false // la ligne suivante (meme vide) est la liste de points 2D a ignorer
        return ReferenceImage(imageId, name, lat, lon)
    }

    private fun quaternionToRotationMatrix(qw: Double, qx: Double, qy: Double, qz: Double): DoubleArray {
        val n = kotlin.math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz).let { if (it < 1e-12) 1.0 else it }
        val w = qw / n; val x = qx / n; val y = qy / n; val z = qz / n
        return doubleArrayOf(
            1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
            2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
            2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)
        )
    }

    private fun transpose3(m: DoubleArray) = doubleArrayOf(m[0], m[3], m[6], m[1], m[4], m[7], m[2], m[5], m[8])
}

/**
 * Requete spatiale simple (rayon + top-N le plus proche) sur l'ensemble des images de reference,
 * utilisee comme `referenceImageIdsProvider` par LocalizerServiceLocator.
 */
class SpatialImageIndex(
    private val images: List<ColmapImageIndexLoader.ReferenceImage>,
    private val radiusM: Double = 150.0,
    private val maxImages: Int = 20
) {
    fun query(lat: Double, lon: Double): List<Long> {
        return images
            .map { it to GeoUtils.haversine(lat, lon, it.lat, it.lon) }
            .filter { it.second <= radiusM }
            .sortedBy { it.second }
            .take(maxImages)
            .map { it.first.imageId }
    }
}
