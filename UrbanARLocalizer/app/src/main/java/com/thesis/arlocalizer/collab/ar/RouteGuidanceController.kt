package com.thesis.arlocalizer.collab.ar

import com.thesis.arlocalizer.collab.model.VirtualObjectDefinition
import com.thesis.arlocalizer.model.GeoPose
import com.thesis.arlocalizer.util.GeoUtils
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.sin

/**
 * Calcule la direction et la distance vers un objet virtuel cible, a partir de la pose courante
 * fournie par la cascade de localisation. Reutilise directement GeoUtils (Haversine, azimuth) deja
 * utilise par le Niveau 2 -- meme logique geometrique, pas de nouveau calcul de zero.
 *
 * La fleche de guidage (models/arrow.glb) est orientee selon `bearingRad` et positionnee a une
 * distance fixe devant l'utilisateur (meme principe que direction_nav dans ArNavigationActivity),
 * jusqu'a ce que ARRIVAL_RADIUS_M soit atteint, ou` le mode passe automatiquement a la manipulation.
 */
class RouteGuidanceController(private val arrivalRadiusM: Double = 10.0) {

    data class GuidanceInfo(
        val target: VirtualObjectDefinition,
        val distanceM: Double,
        val bearingRad: Double, // azimut absolu (0 = Nord), vers la cible
        val hasArrived: Boolean
    )

    fun computeGuidance(currentPose: GeoPose, target: VirtualObjectDefinition): GuidanceInfo {
        val distance = GeoUtils.haversine(currentPose.latitude, currentPose.longitude, target.latitude, target.longitude)
        val bearing = GeoUtils.azimuth(currentPose.latitude, currentPose.longitude, target.latitude, target.longitude)
        return GuidanceInfo(
            target = target,
            distanceM = distance,
            bearingRad = bearing,
            hasArrived = distance <= arrivalRadiusM
        )
    }

    /**
     * Direction de la fleche dans le repere ecran/camera (repere ENU local), meme principe que
     * l'Eq. direction_nav deja utilisee pour les annotations de navigation : vecteur unitaire
     * horizontal pointant vers la cible, exprime dans le repere ENU (Est, Nord).
     */
    fun arrowDirectionEnu(bearingRad: Double): DoubleArray {
        // azimut 0 = Nord (axe Y de l'ENU), sens horaire -> composantes (Est, Nord)
        val east = sin(bearingRad)
        val north = cos(bearingRad)
        return doubleArrayOf(east, north)
    }
}
