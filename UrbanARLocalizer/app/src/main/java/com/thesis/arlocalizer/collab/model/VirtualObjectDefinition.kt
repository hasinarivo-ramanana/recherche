package com.thesis.arlocalizer.collab.model

/**
 * Definition statique/catalogue d'un objet virtuel geolocalise (l'equivalent, pour les objets RA,
 * de SigLandmark pour les amers de localisation). Un VirtualObjectDefinition ne change JAMAIS de
 * position geographique une fois cree (contrairement a ObjectManipulationState, qui capture les
 * modifications collaboratives en temps reel : rotation, decalage local, couleur).
 *
 * objectType doit correspondre a une entree de ModelCatalog (chaise, table, ordinateur, ...).
 */
data class VirtualObjectDefinition(
    val objectId: String,
    val objectType: String,
    val latitude: Double,
    val longitude: Double,
    val label: String = "",
    val defaultRotationYDeg: Float = 0f,
    val defaultColorArgb: Int = -1 // android.graphics.Color.WHITE, evite la dependance Android ici
)
