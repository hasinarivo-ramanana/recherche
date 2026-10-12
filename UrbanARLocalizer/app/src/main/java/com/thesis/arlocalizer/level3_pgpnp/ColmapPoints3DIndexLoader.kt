package com.thesis.arlocalizer.level3_pgpnp

import android.content.Context
import com.thesis.arlocalizer.model.Point3D

/**
 * Charge points3D.txt (format texte COLMAP) et construit l'index (image_id, point2D_idx) -> Point3D
 * necessaire a SiftCorrespondenceBuilder pour retrouver le point 3D correspondant a un appariement SIFT
 * (chaque point 3D reconstruit par COLMAP reference un "track" de detections 2D dans plusieurs images).
 *
 * Format attendu (genere par COLMAP), une ligne par point 3D :
 *   POINT3D_ID X Y Z R G B ERROR IMAGE_ID POINT2D_IDX IMAGE_ID POINT2D_IDX ...
 *
 * Pour generer points3D.txt a partir d'une reconstruction binaire COLMAP :
 *   colmap model_converter --input_path sparse/0 --output_path sparse/0_txt --output_type TXT
 * (ou depuis l'interface graphique COLMAP : File > Export model as text)
 *
 * Placez le fichier resultant dans app/src/main/assets/points3D.txt.
 *
 * Les coordonnees XYZ lues ici sont en repere SfM arbitraire (PAS en metres) ; elles sont converties
 * en ENU metrique local a la volee via GeorefCalibration.aligner() (cf. commentaire de loadFromAssets
 * ci-dessous) -- c'est cet index, et non PlyPointCloud, qui alimente reellement le PG-PnP.
 *
 * NOTE (contrairement a images.txt) : ici, CHAQUE ligne est reellement utilisee -- il n'y a pas de
 * donnees "mortes" a retirer a la source (pas de script de nettoyage equivalent a
 * strip_images_txt.py). En revanche, comme pour images.txt, on lit le fichier LIGNE PAR LIGNE
 * (useLines, paresseux) plutot que de charger tout le fichier en List<String> d'un coup :
 * l'ancien `reader.readLines()` pouvait significativement augmenter le pic memoire au demarrage
 * (chaque ligne devient un objet String en tas Java/Kotlin, UTF-16, jusqu'a ~2x la taille brute
 * du fichier), ce qui se cumule avec les autres fichiers volumineux charges au meme moment
 * (nuage de points, database.db, modeles TFLite) dans LocalizerServiceLocator.build().
 */
object ColmapPoints3DIndexLoader {

    /**
     * @param aligner transformation de similarite SfM -> ENU a appliquer a chaque point 3D
     *   (cf. GeorefCalibration.aligner()). CRITIQUE : ce sont CES points (pas ceux charges par
     *   PlyPointCloud) qui alimentent reellement SiftCorrespondenceBuilder -> PgPnpEngine (filtrage
     *   geographique + RANSAC + PnP) via Level3Refiner. Un mauvais alignement ici (ou une absence
     *   d'alignement) reintroduit directement l'erreur de plusieurs kilometres observee sur le
     *   Niveau 3, meme si PlyPointCloud est correctement aligne par ailleurs.
     */
    fun loadFromAssets(
        context: Context,
        assetName: String = "points3D.txt",
        aligner: SfmToEnuAligner = GeorefCalibration.aligner()
    ): Map<Pair<Long, Int>, Point3D> {
        val index = HashMap<Pair<Long, Int>, Point3D>()
        context.assets.open(assetName).bufferedReader().useLines { lineSequence ->
            for (rawLine in lineSequence) {
                parseLine(rawLine, index, aligner)
            }
        }
        return index
    }

    /** Expose separement pour les tests unitaires (pas de dependance Android Context). */
    fun parse(lines: List<String>, aligner: SfmToEnuAligner = GeorefCalibration.aligner()): Map<Pair<Long, Int>, Point3D> {
        val index = HashMap<Pair<Long, Int>, Point3D>()
        for (line in lines) {
            parseLine(line, index, aligner)
        }
        return index
    }

    private fun parseLine(line: String, index: HashMap<Pair<Long, Int>, Point3D>, aligner: SfmToEnuAligner) {
        val trimmed = line.trim()
        if (trimmed.isEmpty() || trimmed.startsWith("#")) return

        val tokens = trimmed.split(Regex("\\s+"))
        if (tokens.size < 8) return // ligne malformee (point sans aucune observation), on ignore

        val pointId = tokens[0].toLong()
        val x = tokens[1].toDouble()
        val y = tokens[2].toDouble()
        val z = tokens[3].toDouble()
        // tokens[4..6] = R,G,B (couleur) ; tokens[7] = erreur de reprojection moyenne -- non utilises ici

        val enu = aligner.applyXYZ(x, y, z)
        val point3D = Point3D(id = pointId, x = enu[0], y = enu[1], z = enu[2])

        // Le "track" (observations 2D) commence a l'index 8, par paires (IMAGE_ID, POINT2D_IDX)
        var i = 8
        while (i + 1 < tokens.size) {
            val imageId = tokens[i].toLongOrNull()
            val point2DIdx = tokens[i + 1].toIntOrNull()
            if (imageId != null && point2DIdx != null) {
                index[imageId to point2DIdx] = point3D
            }
            i += 2
        }
    }
}
