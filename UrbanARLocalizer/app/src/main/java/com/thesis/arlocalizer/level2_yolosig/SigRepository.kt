package com.thesis.arlocalizer.level2_yolosig

import android.content.Context
import com.google.gson.JsonParser
import com.thesis.arlocalizer.model.SigLandmark
import com.thesis.arlocalizer.util.GeoUtils

/**
 * Charge et interroge le SIG (fichier GeoJSON, assets/sig_data.geojson).
 * Correspond a l'"ETAPE 2 : REQUETE SIG" du pipeline d'affinement (Chapitre 2).
 */
class SigRepository private constructor(private val landmarks: List<SigLandmark>) {

    companion object {
        /** Rayon par defaut de la requete SIG autour de la position APR (~100 m, cf. Fig. pipeline_affinement). */
        const val DEFAULT_RADIUS_M = 100.0

        fun loadFromAssets(context: Context, assetName: String = "sig_data.geojson"): SigRepository {
            val text = context.assets.open(assetName).bufferedReader().use { it.readText() }
            val root = JsonParser.parseString(text).asJsonObject
            val features = root.getAsJsonArray("features")
            val list = mutableListOf<SigLandmark>()
            for (el in features) {
                val f = el.asJsonObject
                val geom = f.getAsJsonObject("geometry")
                val coords = geom.getAsJsonArray("coordinates") // [lon, lat]
                val props = f.getAsJsonObject("properties")
                list.add(
                    SigLandmark(
                        id = props.get("id")?.asString ?: "unknown",
                        classe = props.get("classe")?.asString ?: "",
                        longitude = coords[0].asDouble,
                        latitude = coords[1].asDouble,
                        hauteurReelleM = props.get("hauteur_reelle_m")?.asDouble ?: 1.0,
                        azimutFacadeDeg = props.get("azimut_facade_deg")?.asDouble ?: 0.0,
                        nomBatiment = props.get("nom_batiment")?.asString ?: ""
                    )
                )
            }
            return SigRepository(list)
        }
    }

    /**
     * Retourne les candidats c_k de la classe demandee, dans un rayon donne autour de la position APR.
     * Correspond a C = SIG(p_APR, kappa*, r_APR) de l'Algorithme "Selection du candidat optimal".
     */
    fun queryCandidates(centerLat: Double, centerLon: Double, classe: String, radiusM: Double = DEFAULT_RADIUS_M): List<SigLandmark> {
        return landmarks.filter { it.classe == classe }
            .filter { GeoUtils.haversine(centerLat, centerLon, it.latitude, it.longitude) <= radiusM }
    }

    /** Requete generale sans filtre de classe, utilisee par le filtrage geographique du Niveau 3. */
    fun queryAllWithinRadius(centerLat: Double, centerLon: Double, radiusM: Double): List<SigLandmark> {
        return landmarks.filter { GeoUtils.haversine(centerLat, centerLon, it.latitude, it.longitude) <= radiusM }
    }

    fun all(): List<SigLandmark> = landmarks
}
