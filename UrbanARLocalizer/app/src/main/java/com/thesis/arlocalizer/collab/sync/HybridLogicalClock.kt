package com.thesis.arlocalizer.collab.sync

import org.json.JSONObject

/**
 * Horloge logique hybride (HLC) - triplet (physical, logical, replicaId), repris a l'identique de
 * la demonstration CRDT (UrbanLocalizerAR-CRDTDemoV2/sync/HybridLogicalClock.kt) : la logique de
 * fusion CRDT est independante du mecanisme d'ancrage spatial, donc reutilisable sans changement.
 *
 * L'ordre lexicographique sur (physical, logical) fournit l'ordre causal <_HLC utilise par l'ordre
 * total (priorite puis HLC) implemente dans ObjectManipulationState.
 */
data class HLCTimestamp(
    val physical: Long,
    val logical: Int,
    val replicaId: String
) : Comparable<HLCTimestamp> {

    override fun compareTo(other: HLCTimestamp): Int {
        if (physical != other.physical) return physical.compareTo(other.physical)
        if (logical != other.logical) return logical.compareTo(other.logical)
        return replicaId.compareTo(other.replicaId)
    }

    fun toJson(): JSONObject = JSONObject()
        .put("physical", physical)
        .put("logical", logical)
        .put("replicaId", replicaId)

    companion object {
        fun fromJson(json: JSONObject): HLCTimestamp = HLCTimestamp(
            physical = json.getLong("physical"),
            logical = json.getInt("logical"),
            replicaId = json.getString("replicaId")
        )

        val EPOCH = HLCTimestamp(0L, 0, "")
    }
}

class HybridLogicalClock(private val replicaId: String) {
    private var lastPhysical = 0L
    private var counter = 0

    @Synchronized
    fun tick(): HLCTimestamp {
        val now = System.currentTimeMillis()
        if (now > lastPhysical) {
            lastPhysical = now
            counter = 0
        } else {
            counter += 1
        }
        return HLCTimestamp(lastPhysical, counter, replicaId)
    }

    @Synchronized
    fun update(received: HLCTimestamp): HLCTimestamp {
        val now = System.currentTimeMillis()
        lastPhysical = maxOf(lastPhysical, received.physical, now)
        counter = if (lastPhysical == received.physical) received.logical + 1 else 0
        return HLCTimestamp(lastPhysical, counter, replicaId)
    }
}
