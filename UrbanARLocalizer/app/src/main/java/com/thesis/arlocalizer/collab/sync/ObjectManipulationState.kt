package com.thesis.arlocalizer.collab.sync

import org.json.JSONObject

/**
 * Etat manipulable et replique d'un objet virtuel (equivalent de ObjectCRDTState dans la
 * demonstration, mais adapte a des modeles .glb geolocalises au lieu d'un cube sur Cloud Anchor) :
 *
 * - offsetX/Y/Z : decalage en metres par rapport a la position CANONIQUE de l'objet (celle de son
 *   VirtualObjectDefinition, convertie en position AR-locale par PoseAnchorManager.placeVirtualObject).
 *   C'est ce decalage, et non une Cloud Anchor, qui permet a plusieurs utilisateurs -- non
 *   co-presents, potentiellement a des heures differentes -- de voir et manipuler le meme objet au
 *   meme endroit physique reel : chacun localise cet endroit independamment via sa propre cascade
 *   de localisation (APR + YOLO/SIG + PG-PnP) plutot que par resolution d'ancre partagee.
 * - rotationYDeg : rotation autour de l'axe vertical (lacet), en degres. Suffisant pour "tourner"
 *   un meuble poser au sol ; les rotations en tangage/roulis ne sont pas exposees a l'utilisateur.
 * - colorArgb : teinte appliquee au modele (multiplicateur de couleur sur le materiau de base).
 *
 * Le reste (priority, timestamp, precedes/merge, ordre total) est identique a la demonstration :
 * priorite SARA d'abord, HLC en depart ; l'operateur merge est commutatif/associatif/idempotent
 * (semi-treillis), garantissant la convergence forte independamment de l'ordre d'arrivee reseau.
 */
data class ObjectManipulationState(
    val objectId: String,
    val offsetX: Float,
    val offsetY: Float,
    val offsetZ: Float,
    val rotationYDeg: Float,
    val colorArgb: Int,
    val priority: Int,
    val timestamp: HLCTimestamp,
    val originReplica: String
) {
    private fun precedes(other: ObjectManipulationState): Boolean {
        if (priority != other.priority) return priority < other.priority
        return timestamp < other.timestamp
    }

    /** Operateur de fusion ⊕ : max au sens de l'ordre total ci-dessus. */
    infix fun merge(other: ObjectManipulationState): ObjectManipulationState =
        if (this.precedes(other)) other else this

    fun toJson(): JSONObject = JSONObject()
        .put("type", "object_state")
        .put("objectId", objectId)
        .put("offsetX", offsetX)
        .put("offsetY", offsetY)
        .put("offsetZ", offsetZ)
        .put("rotationYDeg", rotationYDeg)
        .put("colorArgb", colorArgb)
        .put("priority", priority)
        .put("timestamp", timestamp.toJson())
        .put("originReplica", originReplica)

    companion object {
        fun fromJson(json: JSONObject): ObjectManipulationState = ObjectManipulationState(
            objectId = json.getString("objectId"),
            offsetX = json.getDouble("offsetX").toFloat(),
            offsetY = json.getDouble("offsetY").toFloat(),
            offsetZ = json.getDouble("offsetZ").toFloat(),
            rotationYDeg = json.getDouble("rotationYDeg").toFloat(),
            colorArgb = json.getInt("colorArgb"),
            priority = json.getInt("priority"),
            timestamp = HLCTimestamp.fromJson(json.getJSONObject("timestamp")),
            originReplica = json.getString("originReplica")
        )

        /** Etat initial : objet a sa position canonique, sans decalage, couleur/rotation par defaut. */
        fun default(
            objectId: String,
            replicaId: String,
            defaultRotationYDeg: Float = 0f,
            defaultColorArgb: Int = -1
        ): ObjectManipulationState = ObjectManipulationState(
            objectId = objectId,
            offsetX = 0f, offsetY = 0f, offsetZ = 0f,
            rotationYDeg = defaultRotationYDeg,
            colorArgb = defaultColorArgb,
            priority = 1,
            timestamp = HLCTimestamp.EPOCH,
            originReplica = replicaId
        )
    }
}
