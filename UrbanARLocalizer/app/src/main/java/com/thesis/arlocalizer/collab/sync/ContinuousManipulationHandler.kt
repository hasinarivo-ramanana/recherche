package com.thesis.arlocalizer.collab.sync

/**
 * Gere la diffusion des mises a jour pendant une manipulation continue (glisser pour translater,
 * geste de rotation, defilement de la palette de couleurs) : diffusion a frequence reduite pendant
 * le geste plutot qu'a chaque frame, avec diffusion + fusion finale garantie a la fin du geste.
 * Logique inchangee par rapport a la demonstration (ContinuousManipulationHandler.kt).
 */
class ContinuousManipulationHandler(
    private val sync: CollabSyncRepository,
    private val broadcastHz: Float = 10f
) {
    private var lastBroadcast = 0L
    private val intervalMs = (1000f / broadcastHz).toLong()

    fun onDragUpdate(newState: ObjectManipulationState) {
        val now = System.currentTimeMillis()
        if (now - lastBroadcast >= intervalMs) {
            sync.broadcastLocalUpdate(newState)
            lastBroadcast = now
        }
    }

    fun onDragEnd(finalState: ObjectManipulationState) {
        sync.broadcastLocalUpdate(finalState)
        lastBroadcast = System.currentTimeMillis()
    }
}
