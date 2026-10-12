package com.thesis.arlocalizer.model

/**
 * Amer semantique du SIG (logo, enseigne) - correspond a un candidat c_k = (lambda_k, phi_k, delta_k, sigma_k)
 * du chapitre "Localisation par detection des logos et des enseignes".
 *
 * Attendu au format GeoJSON, proprietes minimales :
 * {
 *   "type": "Feature",
 *   "geometry": { "type": "Point", "coordinates": [lon, lat] },
 *   "properties": {
 *      "id": "bni_001",
 *      "classe": "bni",              // une des 6 classes : misa, buvette, bni, bfv, boa, pharmacie
 *      "hauteur_reelle_m": 0.6,      // H_kappa^reel, hauteur reelle du logo (m)
 *      "azimut_facade_deg": 45.0,    // delta_k, orientation de la facade (0 = Nord)
 *      "nom_batiment": "Agence BNI Ankatso",
 *      "salle_ou_service": null
 *   }
 * }
 */
data class SigLandmark(
    val id: String,
    val classe: String,
    val latitude: Double,
    val longitude: Double,
    val hauteurReelleM: Double,
    val azimutFacadeDeg: Double,
    val nomBatiment: String
)

/** Detection brute issue de YOLOv10 (Etape 1 du pipeline d'affinement, chapitre 2). */
data class YoloDetection(
    val classe: String,
    /** bbox en pixels dans l'image analysee, (u,v) = coin haut-gauche */
    val u: Float,
    val v: Float,
    val widthPx: Float,
    val heightPx: Float,
    val confidence: Float
) {
    val centerU: Float get() = u + widthPx / 2f
    val centerV: Float get() = v + heightPx / 2f
}

/** Score multi-criteres d'un candidat SIG pour une detection donnee (Section "Scoring Multi-Criteres"). */
data class ScoredCandidate(
    val landmark: SigLandmark,
    val sTaille: Double,
    val sAngle: Double,
    val sRapport: Double,
    val sCote: Double,
    val total: Double
)

/** Point 3D du nuage COLMAP, avec identifiant reliant au descripteur SIFT dans database.db. */
data class Point3D(
    val id: Long,
    val x: Double,
    val y: Double,
    val z: Double,
    /** id(s) de points-image 2D (image_id, point2D_idx) associes, utilise pour retrouver les descripteurs SIFT */
    val trackImageIds: LongArray = LongArray(0),
    val trackPoint2DIdx: IntArray = IntArray(0)
)

/** Correspondance 2D (image requete) <-> 3D (nuage COLMAP), entree du PnP. */
data class Correspondence2D3D(
    val point2D: DoubleArray, // (u, v) px
    val point3D: DoubleArray  // (X, Y, Z) m, repere local ENU
)
