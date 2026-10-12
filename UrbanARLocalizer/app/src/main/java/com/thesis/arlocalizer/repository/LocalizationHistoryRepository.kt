package com.thesis.arlocalizer.repository

import com.thesis.arlocalizer.model.GeoPose
import java.util.concurrent.CopyOnWriteArrayList

/**
 * Un enregistrement de localisation : les poses (eventuellement nulles) obtenues a un instant donne
 * pour chacun des 3 niveaux de la cascade. Utilise pour peupler la carte OSM et l'export CSV.
 */
data class LocalizationRecord(
    val timestampMs: Long,
    val level1: GeoPose?,
    val level2: GeoPose?,
    val level3: GeoPose?
)

/**
 * Accumule l'historique complet d'une session de navigation (echantillonnage toutes les 5 secondes,
 * cf. ArNavigationActivity). Singleton en memoire, vide au redemarrage du processus.
 */
object LocalizationHistoryRepository {

    private val records = CopyOnWriteArrayList<LocalizationRecord>()
    private val listeners = CopyOnWriteArrayList<() -> Unit>()

    fun add(record: LocalizationRecord) {
        records.add(record)
        listeners.forEach { it.invoke() }
    }

    fun all(): List<LocalizationRecord> = records.toList()

    fun clear() {
        records.clear()
        listeners.forEach { it.invoke() }
    }

    /** S'abonner aux mises a jour (ex: rafraichir la carte). Retourne une fonction de desabonnement. */
    fun observe(listener: () -> Unit): () -> Unit {
        listeners.add(listener)
        return { listeners.remove(listener) }
    }

    fun isEmpty(): Boolean = records.isEmpty()
}
