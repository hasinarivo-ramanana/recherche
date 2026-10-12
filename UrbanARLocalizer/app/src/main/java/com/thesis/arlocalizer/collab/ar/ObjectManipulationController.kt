package com.thesis.arlocalizer.collab.ar

import com.thesis.arlocalizer.collab.sync.CollabSyncRepository
import com.thesis.arlocalizer.collab.sync.ContinuousManipulationHandler
import com.thesis.arlocalizer.collab.sync.ObjectManipulationState
import kotlin.math.cos
import kotlin.math.sin

/**
 * Pilote les modifications de l'objet virtuel "actif" (celui que l'utilisateur a atteint, cf.
 * RouteGuidanceController.hasArrived) et les diffuse via CRDT. Les gestes discrets (un seul
 * tap = une seule mise a jour) sont diffuses immediatement ; les gestes continus (appui maintenu
 * pour tourner/deplacer en continu) passent par ContinuousManipulationHandler pour limiter le
 * debit reseau (10 Hz par defaut), avec diffusion + fusion garantie a la fin du geste.
 *
 * priority : niveau de priorite SARA de CET utilisateur pour cette session (reglable, ex. pour
 * donner la main a un enseignant/animateur sur les etudiants lors d'une demonstration collaborative).
 */
class ObjectManipulationController(
    private val sync: CollabSyncRepository,
    private val replicaId: String,
    private var priority: Int = 1
) {
    private val continuousHandler = ContinuousManipulationHandler(sync)
    private var current: ObjectManipulationState? = null

    fun setPriority(newPriority: Int) { priority = newPriority }

    fun bindObject(initialState: ObjectManipulationState) {
        current = initialState
    }

    fun currentState(): ObjectManipulationState? = current

    /** Rotation discrete (tap sur un bouton +/- degres). */
    fun rotateBy(deltaDeg: Float) {
        val c = current ?: return
        val updated = c.copy(rotationYDeg = normalizeAngle(c.rotationYDeg + deltaDeg), priority = priority)
        current = updated
        sync.broadcastLocalUpdate(updated)
    }

    /** Rotation continue (appui maintenu) : a appeler a chaque frame pendant le maintien. */
    fun rotateContinuous(deltaDegPerTick: Float) {
        val c = current ?: return
        val updated = c.copy(rotationYDeg = normalizeAngle(c.rotationYDeg + deltaDegPerTick), priority = priority)
        current = updated
        continuousHandler.onDragUpdate(updated)
    }

    fun endContinuousGesture() {
        current?.let { continuousHandler.onDragEnd(it.copy(priority = priority)) }
    }

    /**
     * Translation dans le repere de l'objet (avant/arriere/gauche/droite relatifs a sa rotation
     * courante), en metres. Discrete (tap) ou continue (maintien), meme distinction que rotateBy.
     */
    fun translateBy(forwardM: Float, rightM: Float, continuous: Boolean = false) {
        val c = current ?: return
        val yawRad = Math.toRadians(c.rotationYDeg.toDouble())
        val dx = (forwardM * sin(yawRad) + rightM * cos(yawRad)).toFloat()
        val dz = (forwardM * cos(yawRad) - rightM * sin(yawRad)).toFloat()
        val updated = c.copy(offsetX = c.offsetX + dx, offsetZ = c.offsetZ + dz, priority = priority)
        current = updated
        if (continuous) continuousHandler.onDragUpdate(updated) else sync.broadcastLocalUpdate(updated)
    }

    /** Changement de couleur (palette de teintes predefinies dans l'UI). */
    fun setColor(argb: Int) {
        val c = current ?: return
        val updated = c.copy(colorArgb = argb, priority = priority)
        current = updated
        sync.broadcastLocalUpdate(updated)
    }

    private fun normalizeAngle(deg: Float): Float {
        var a = deg % 360f
        if (a < 0f) a += 360f
        return a
    }
}
