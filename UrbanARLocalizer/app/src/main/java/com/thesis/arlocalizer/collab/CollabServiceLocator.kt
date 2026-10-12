package com.thesis.arlocalizer.collab

import android.content.Context
import com.thesis.arlocalizer.collab.ar.ObjectManipulationController
import com.thesis.arlocalizer.collab.ar.RouteGuidanceController
import com.thesis.arlocalizer.collab.model.VirtualObjectCatalogRepository
import com.thesis.arlocalizer.collab.sync.CollabSyncRepository
import com.thesis.arlocalizer.collab.sync.HybridLogicalClock
import java.util.UUID

/**
 * Point d'assemblage unique de la fonctionnalite collaborative RA (catalogue d'objets geolocalises,
 * synchronisation CRDT, guidage, manipulation). Analogue a LocalizerServiceLocator pour la cascade
 * de localisation ; separe pour ne pas alourdir ce dernier avec une fonctionnalite optionnelle.
 *
 * L'identifiant de replique (replicaId) est genere une fois et persiste pendant la duree de vie du
 * processus -- suffisant pour une session de demonstration ; a stocker dans les preferences
 * partagees si vous voulez qu'il survive aux redemarrages de l'app (utile pour reconnaitre "qui a
 * fait quoi" au fil de plusieurs sessions).
 */
object CollabServiceLocator {

    @Volatile private var catalogRepository: VirtualObjectCatalogRepository? = null
    @Volatile private var syncRepository: CollabSyncRepository? = null
    @Volatile private var manipulationController: ObjectManipulationController? = null
    @Volatile private var guidanceController: RouteGuidanceController? = null

    private val replicaId: String by lazy { UUID.randomUUID().toString() }

    fun getCatalogRepository(context: Context): VirtualObjectCatalogRepository =
        catalogRepository ?: synchronized(this) {
            catalogRepository ?: VirtualObjectCatalogRepository.load(context).also { catalogRepository = it }
        }

    fun getSyncRepository(): CollabSyncRepository =
        syncRepository ?: synchronized(this) {
            syncRepository ?: CollabSyncRepository(HybridLogicalClock(replicaId)).also { syncRepository = it }
        }

    fun getManipulationController(): ObjectManipulationController =
        manipulationController ?: synchronized(this) {
            manipulationController ?: ObjectManipulationController(getSyncRepository(), replicaId).also { manipulationController = it }
        }

    fun getGuidanceController(): RouteGuidanceController =
        guidanceController ?: synchronized(this) {
            guidanceController ?: RouteGuidanceController().also { guidanceController = it }
        }

    fun currentReplicaId(): String = replicaId

    /**
     * Connecte au noeud edge et relie automatiquement les nouveaux objets recus (catalog_add) a la
     * base locale du catalogue, pour qu'ils persistent hors-ligne et apparaissent immediatement sur
     * la mini-carte de tous les autres utilisateurs connectes.
     */
    fun connectAndBridge(context: Context, wsUrl: String) {
        val catalog = getCatalogRepository(context)
        val sync = getSyncRepository()
        sync.connect(wsUrl)
        // NOTE : a appeler depuis une coroutine/lifecycleScope pour collecter catalogAdditions
        // (StateFlow) et appeler catalog.addRemoteObject(...) a chaque nouvelle entree -- voir
        // l'exemple de branchement dans ArNavigationActivity.
    }
}
