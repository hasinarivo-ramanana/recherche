package com.thesis.arlocalizer.level3_pgpnp

/**
 * Transformation de similarite COLMAP (repere SfM arbitraire) -> ENU (metres reels), calibree
 * hors-ligne par le script Python de reference (localisation_app-3/main.py --gps-csv
 * colmap/reference_gps_colmap.txt), qui resout un alignement de Umeyama sur les 631 images
 * d'entrainement communes entre images.txt et reference_gps_colmap.txt.
 *
 * Valeurs copiees telles quelles depuis cache/colmap_to_enu_transform.json genere par ce script :
 * dst_ENU = s * (R @ src_SfM) + t, ENU exprime en metres (Est, Nord, Haut) autour de (lat0, lon0).
 *
 * IMPORTANT : ces valeurs doivent rester strictement identiques entre le pipeline Python (utilise
 * pour entrainer/valider) et l'application Android, sous peine d'erreurs de localisation Niveau 3
 * de plusieurs kilometres (cf. le nuage points3D.ply n'est PAS georegistre nativement : ses
 * coordonnees XYZ sont en unites SfM arbitraires, pas en metres). Si le modele COLMAP est
 * reconstruit/remicro-recalibre a l'avenir, regenerer ce fichier a partir du nouveau
 * cache/colmap_to_enu_transform.json.
 */
object GeorefCalibration {

    // Rotation R (3x3, row-major : R[0..2]=ligne0, R[3..5]=ligne1, R[6..8]=ligne2)
    val ROTATION: DoubleArray = doubleArrayOf(
        0.33868656358720184, -0.8529856422771627, 0.3971232878017969,
        -0.8281673355030403, -0.06992886067058979, 0.556101446546499,
        -0.44657617049198595, -0.5172286230617266, -0.7300988121031172
    )

    // Echelle (SfM -> metres)
    const val SCALE: Double = 0.9297312825209757

    // Translation (metres, apres rotation+echelle)
    val TRANSLATION: DoubleArray = doubleArrayOf(
        -17.61611841070531, -21.782564674299554, -0.6118754330337612
    )

    // Origine geographique du repere ENU local (moyenne des positions GPS de reference utilisees
    // pour le calibrage) -- DOIT etre utilisee comme enuOriginLat/enuOriginLon partout ou cet
    // aligneur est applique (LocalizerServiceLocator, Level3Refiner, ColmapImageIndexLoader), pour
    // que la reconversion ENU -> lat/lon finale soit coherente avec cette meme origine.
    const val LAT0: Double = -18.914591159429477
    const val LON0: Double = 47.55227018922345

    const val N_IMAGES_USED: Int = 631

    /** Aligneur SfM -> ENU pret a l'emploi, a utiliser a la place de SfmToEnuAligner.identity(). */
    fun aligner(): SfmToEnuAligner = SfmToEnuAligner(SCALE, ROTATION, TRANSLATION)
}
