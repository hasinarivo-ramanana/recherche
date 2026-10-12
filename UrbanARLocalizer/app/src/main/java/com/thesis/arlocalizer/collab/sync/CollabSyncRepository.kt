package com.thesis.arlocalizer.collab.sync

import android.os.Handler
import android.os.Looper
import com.thesis.arlocalizer.collab.model.VirtualObjectDefinition
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.update
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.Response
import okhttp3.WebSocket
import okhttp3.WebSocketListener
import org.json.JSONObject
import java.util.concurrent.TimeUnit

enum class ConnectionState { DISCONNECTED, CONNECTING, CONNECTED }

/**
 * Client de synchronisation WebSocket, adapte de SyncRepository de la demonstration CRDT.
 * Deux differences avec la V2 :
 *  1) Le message "anchor" (ARCore Cloud Anchor) est supprime : le referentiel spatial commun est
 *     desormais fourni par la cascade de localisation (PoseAnchorManager), independamment pour
 *     chaque replique -- aucune coordination reseau n'est necessaire pour "placer" un objet.
 *  2) Un nouveau type de message "catalog_add" permet a un utilisateur en "mode placement" d'ajouter
 *     un nouvel objet geolocalise au catalogue partage, diffuse en temps reel a tous les autres
 *     utilisateurs connectes (cf. VirtualObjectCatalogRepository).
 */
class CollabSyncRepository(private val clock: HybridLogicalClock) {

    private val client = OkHttpClient.Builder()
        .pingInterval(15, TimeUnit.SECONDS)
        .retryOnConnectionFailure(true)
        .build()

    private var socket: WebSocket? = null
    private var retryDelayMs = 1_000L
    private var manuallyDisconnected = false
    private var currentUrl: String = ""
    private val mainHandler = Handler(Looper.getMainLooper())

    private val _states = MutableStateFlow<Map<String, ObjectManipulationState>>(emptyMap())
    val states: StateFlow<Map<String, ObjectManipulationState>> = _states

    private val _catalogAdditions = MutableStateFlow<Map<String, VirtualObjectDefinition>>(emptyMap())
    val catalogAdditions: StateFlow<Map<String, VirtualObjectDefinition>> = _catalogAdditions

    private val _connectionState = MutableStateFlow(ConnectionState.DISCONNECTED)
    val connectionState: StateFlow<ConnectionState> = _connectionState

    val eventLog = MutableSharedFlow<String>(replay = 0, extraBufferCapacity = 32)

    fun connect(url: String) {
        currentUrl = url
        manuallyDisconnected = false
        retryDelayMs = 1_000L
        openSocket()
    }

    fun disconnect() {
        manuallyDisconnected = true
        mainHandler.removeCallbacksAndMessages(null)
        socket?.close(1000, "client disconnect")
        socket = null
        _connectionState.value = ConnectionState.DISCONNECTED
    }

    private fun openSocket() {
        if (currentUrl.isBlank()) return
        _connectionState.value = ConnectionState.CONNECTING
        val request = Request.Builder().url(currentUrl).build()
        socket = client.newWebSocket(request, object : WebSocketListener() {

            override fun onOpen(ws: WebSocket, response: Response) {
                retryDelayMs = 1_000L
                _connectionState.value = ConnectionState.CONNECTED
                eventLog.tryEmit("Connecte au noeud edge ($currentUrl)")
            }

            override fun onMessage(ws: WebSocket, text: String) {
                try {
                    val json = JSONObject(text)
                    when (json.optString("type", "object_state")) {
                        "catalog_add" -> handleIncomingCatalogAdd(json)
                        else -> {
                            val incoming = ObjectManipulationState.fromJson(json)
                            clock.update(incoming.timestamp)
                            mergeIncoming(incoming, remote = true)
                        }
                    }
                } catch (e: Exception) {
                    eventLog.tryEmit("Message ignore (format invalide): ${e.message}")
                }
            }

            override fun onClosed(ws: WebSocket, code: Int, reason: String) {
                _connectionState.value = ConnectionState.DISCONNECTED
                eventLog.tryEmit("Connexion fermee ($reason)")
                scheduleReconnect()
            }

            override fun onFailure(ws: WebSocket, t: Throwable, r: Response?) {
                _connectionState.value = ConnectionState.DISCONNECTED
                eventLog.tryEmit("Echec de connexion: ${t.message}")
                scheduleReconnect()
            }
        })
    }

    private fun scheduleReconnect() {
        if (manuallyDisconnected || currentUrl.isBlank()) return
        mainHandler.postDelayed({
            openSocket()
            retryDelayMs = (retryDelayMs * 2).coerceAtMost(30_000L)
        }, retryDelayMs)
    }

    private fun mergeIncoming(incoming: ObjectManipulationState, remote: Boolean) {
        _states.update { current ->
            val existing = current[incoming.objectId]
            val merged = existing?.merge(incoming) ?: incoming
            if (remote && existing != null) {
                val winner = if (merged === incoming) "distant (${short(incoming.originReplica)}, pi=${incoming.priority})"
                             else "local (${short(existing.originReplica)}, pi=${existing.priority})"
                eventLog.tryEmit("Fusion objet '${incoming.objectId}' -> gagnant: $winner")
            }
            current + (incoming.objectId to merged)
        }
    }

    private fun handleIncomingCatalogAdd(json: JSONObject) {
        val def = VirtualObjectDefinition(
            objectId = json.getString("objectId"),
            objectType = json.getString("objectType"),
            latitude = json.getDouble("latitude"),
            longitude = json.getDouble("longitude"),
            label = json.optString("label", ""),
            defaultRotationYDeg = json.optDouble("defaultRotationYDeg", 0.0).toFloat(),
            defaultColorArgb = json.optInt("defaultColorArgb", -1)
        )
        _catalogAdditions.update { it + (def.objectId to def) }
        eventLog.tryEmit("Nouvel objet recu du catalogue partage : '${def.objectId}' (${def.objectType})")
    }

    /** Diffuse une mise a jour locale de manipulation, avec application optimiste immediate. */
    fun broadcastLocalUpdate(state: ObjectManipulationState) {
        val stamped = state.copy(timestamp = clock.tick())
        mergeIncoming(stamped, remote = false)
        socket?.send(stamped.toJson().toString())
    }

    /** Diffuse un nouvel objet place par l'utilisateur (mode placement) a tous les utilisateurs connectes. */
    fun broadcastCatalogAdd(def: VirtualObjectDefinition) {
        _catalogAdditions.update { it + (def.objectId to def) }
        val json = JSONObject()
            .put("type", "catalog_add")
            .put("objectId", def.objectId)
            .put("objectType", def.objectType)
            .put("latitude", def.latitude)
            .put("longitude", def.longitude)
            .put("label", def.label)
            .put("defaultRotationYDeg", def.defaultRotationYDeg)
            .put("defaultColorArgb", def.defaultColorArgb)
        socket?.send(json.toString())
    }

    fun seedLocalState(state: ObjectManipulationState) {
        _states.update { it + (state.objectId to state) }
    }

    private fun short(id: String): String = if (id.length > 6) id.substring(0, 6) else id
}
