package com.thesis.arlocalizer.repository

import java.util.concurrent.CopyOnWriteArrayList

/**
 * Accumule, en memoire, le texte detaille (rapport par etapes Niveau 1 / Niveau 2 / Niveau 3)
 * produit a chaque clic sur "Localiser ma position" (cf. ArNavigationActivity + CascadeLogFormatter).
 *
 * Ce contenu est :
 *  - ecrit de facon incrementale dans un fichier texte du dossier Downloads apres CHAQUE clic
 *    (cf. util/TxtLogExporter.appendToDownloadsFile), pour que le fichier grossisse au fil de la
 *    navigation, comme demande ;
 *  - re-exportable a tout moment (nouveau fichier, meme contenu accumule) via le bouton
 *    "Exporter txt" de l'ecran carte (MapActivity), sur le meme principe que l'export CSV.
 *
 * Singleton en memoire, vide au redemarrage du processus (comme LocalizationHistoryRepository).
 */
object CascadeLogRepository {

    private val entries = CopyOnWriteArrayList<String>()
    private val listeners = CopyOnWriteArrayList<() -> Unit>()

    fun add(entryText: String) {
        entries.add(entryText)
        listeners.forEach { it.invoke() }
    }

    fun all(): List<String> = entries.toList()

    /** Concatene toutes les entrees, dans l'ordre chronologique, separees par une ligne vide. */
    fun fullText(): String = entries.joinToString(separator = "\n")

    fun isEmpty(): Boolean = entries.isEmpty()

    fun clear() {
        entries.clear()
        listeners.forEach { it.invoke() }
    }

    fun observe(listener: () -> Unit): () -> Unit {
        listeners.add(listener)
        return { listeners.remove(listener) }
    }
}
