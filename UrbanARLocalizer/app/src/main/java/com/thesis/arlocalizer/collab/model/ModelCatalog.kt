package com.thesis.arlocalizer.collab.model

/**
 * Association objectType -> fichier .glb (fournis par l'utilisateur, a placer dans
 * app/src/main/assets/models/). Point d'extension unique si de nouveaux types d'objets sont ajoutes.
 */
object ModelCatalog {

    const val TYPE_CHAISE = "chaise"
    const val TYPE_TABLE = "table"
    const val TYPE_ORDINATEUR = "ordinateur"

    /** Modele de la fleche de guidage (distinct des objets manipulables du catalogue). */
    const val ARROW_ASSET = "models/arrow.glb"

    private val assetByType = mapOf(
        TYPE_CHAISE to "models/chaise.glb",
        TYPE_TABLE to "models/table.glb",
        TYPE_ORDINATEUR to "models/ordinateur.glb"
    )

    /** Couleur par defaut du marqueur sur la mini-carte, par type (distinction visuelle rapide). */
    private val markerColorByType = mapOf(
        TYPE_CHAISE to 0xFF8D6E63.toInt(),      // brun
        TYPE_TABLE to 0xFFFF9800.toInt(),        // orange
        TYPE_ORDINATEUR to 0xFF1E88E5.toInt()    // bleu
    )

    fun assetPathFor(objectType: String): String =
        assetByType[objectType] ?: throw IllegalArgumentException(
            "Type d'objet inconnu : '$objectType'. Types enregistres : ${assetByType.keys}"
        )

    fun markerColorFor(objectType: String): Int = markerColorByType[objectType] ?: 0xFF9E9E9E.toInt()

    fun allTypes(): Set<String> = assetByType.keys
}
