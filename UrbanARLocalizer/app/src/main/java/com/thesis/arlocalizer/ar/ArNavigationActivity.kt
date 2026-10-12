package com.thesis.arlocalizer.ar

import android.app.AlertDialog
import android.content.Context
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.Color
import android.os.Bundle
import android.view.Gravity
import android.widget.Button
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import com.google.ar.core.Anchor
import com.google.ar.core.Frame
import com.google.ar.core.TrackingState
import com.google.ar.core.exceptions.NotYetAvailableException
import com.thesis.arlocalizer.R
import com.thesis.arlocalizer.collab.CollabServiceLocator
import com.thesis.arlocalizer.collab.ar.ObjectManipulationController
import com.thesis.arlocalizer.collab.ar.RouteGuidanceController
import com.thesis.arlocalizer.collab.ar.VirtualObjectRenderer
import com.thesis.arlocalizer.collab.map.MiniMapController
import com.thesis.arlocalizer.collab.model.ModelCatalog
import com.thesis.arlocalizer.collab.model.VirtualObjectCatalogRepository
import com.thesis.arlocalizer.collab.model.VirtualObjectDefinition
import com.thesis.arlocalizer.collab.sync.ObjectManipulationState
import com.thesis.arlocalizer.map.MapActivity
import com.thesis.arlocalizer.model.GeoPose
import com.thesis.arlocalizer.model.Pose3D
import com.thesis.arlocalizer.model.Quaternion
import com.thesis.arlocalizer.orchestrator.ILocalizationOrchestrator
import com.thesis.arlocalizer.orchestrator.LocalizationOrchestrator
import com.thesis.arlocalizer.orchestrator.RemoteLocalizationOrchestrator
import com.thesis.arlocalizer.poseanchor.PoseAnchorManager
import com.thesis.arlocalizer.repository.CascadeLogRepository
import com.thesis.arlocalizer.repository.LocalizationHistoryRepository
import com.thesis.arlocalizer.repository.LocalizationRecord
import com.thesis.arlocalizer.service.LocalizerServiceLocator
import com.thesis.arlocalizer.util.CascadeLogFormatter
import com.thesis.arlocalizer.util.CompassProvider
import com.thesis.arlocalizer.util.ImageUtils
import com.thesis.arlocalizer.util.TxtLogExporter
import io.github.sceneview.ar.ARSceneView
import io.github.sceneview.ar.node.AnchorNode
import io.github.sceneview.loaders.ModelLoader
import io.github.sceneview.node.ModelNode
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.osmdroid.config.Configuration
import org.osmdroid.views.MapView

/**
 * Active la scene RA (ARCore via SceneView) et pilote :
 *  1) la cascade de localisation sans GPS (Niveaux 1/2/3 + PoseAnchor), toutes les 5 s ;
 *  2) la fonctionnalite collaborative "Pokemon Go" : mini-carte des objets geolocalises, guidage
 *     par fleche .glb vers l'objet selectionne, puis manipulation collaborative (rotation,
 *     translation, couleur) synchronisee par CRDT une fois l'objet atteint.
 *
 * Etats de l'ecran (mutuellement exclusifs) :
 *   NAVIGATION  -> mini-carte visible, aucun objet selectionne
 *   GUIDANCE    -> un objet a ete tape sur la mini-carte ; fleche + panneau de distance affiches
 *   MANIPULATION -> l'utilisateur est arrive (< arrivalRadiusM) ; panneau de manipulation affiche
 */
class ArNavigationActivity : AppCompatActivity() {

    // Cascade de localisation : desormais DEPORTEE sur le serveur Python (RemoteLocalizationOrchestrator),
    // le niveau 3 en Kotlin presentant une erreur de plusieurs centaines de metres. L'interface
    // ILocalizationOrchestrator garde la possibilite de repasser a la cascade locale
    // (LocalizationOrchestrator, qui l'implemente aussi) si besoin, sans toucher au reste du fichier.
    private lateinit var orchestrator: ILocalizationOrchestrator
    private lateinit var statusView: TextView
    private lateinit var sceneView: ARSceneView
    private lateinit var compassProvider: CompassProvider

    private var navigationAnchorNode: AnchorNode? = null
    private var virtualObjectRenderer: VirtualObjectRenderer? = null

    // ---- Modele 3D affiche a chaque resultat de localisation (assets/models/arrow.glb) ----
    private val positionMarkerModelLoader: ModelLoader by lazy { ModelLoader(sceneView.engine, sceneView.context) }
    private var cascadeSequenceCounter = 0
    // La boucle continue (5 s) ne tourne que pendant la phase CRDT (une fois arrive pres de la
    // cible) ; avant cela, la localisation est declenchee manuellement par "Localiser ma position".
    private var continuousLoopRunning = false

    // ---- Fonctionnalite collaborative ----
    private lateinit var catalogRepository: VirtualObjectCatalogRepository
    private lateinit var miniMapController: MiniMapController
    private lateinit var guidanceController: RouteGuidanceController
    private lateinit var manipulationController: ObjectManipulationController

    private var selectedTarget: VirtualObjectDefinition? = null
    private var manipulationTarget: VirtualObjectDefinition? = null
    private var latestPose: GeoPose? = null

    companion object {
        /** Periode d'echantillonnage de la cascade et de l'historique (carte OSM + CSV), demandee a 5 s. */
        private const val LOCALIZATION_PERIOD_MS = 5000L
        private const val PREFS_NAME = "collab_prefs"
        private const val PREF_WS_URL = "ws_url"

        private val COLOR_SWATCHES = intArrayOf(
            Color.WHITE, Color.RED, Color.rgb(0, 170, 0), Color.BLUE, Color.YELLOW, Color.MAGENTA
        )
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_ar_navigation)
        statusView = findViewById(R.id.arStatusText)
        sceneView = findViewById(R.id.sceneView)

        compassProvider = CompassProvider(applicationContext)

        findViewById<Button>(R.id.viewMapButton).setOnClickListener {
            startActivity(Intent(this, MapActivity::class.java))
        }

        findViewById<Button>(R.id.localiserPositionButton).setOnClickListener {
            onLocaliserMaPositionClicked()
        }

        // La cascade de localisation (niv 1/2/3) est desormais DEPORTEE sur le serveur Python
        // (server.py, POST /localize) : le niveau 3 en Kotlin presentait une erreur de plusieurs
        // centaines de metres. On ne charge donc plus de modeles TFLite/COLMAP en local ici -- on
        // a seulement besoin de l'adresse du serveur, qui est la MEME que celle du WebSocket /ws
        // de collaboration (server.py sert les deux sur le meme port). On la demande donc une
        // seule fois ici, avant de construire l'orchestrateur ET avant setupCollabFeature() (qui
        // n'a plus besoin de la redemander, cf. connectToRelayServer() simplifiee plus bas).
        //
        // IMPORTANT : virtualObjectRenderer et getPoseAnchorManager() dependent de l'orchestrateur
        // (le PoseAnchorManager est construit avec lui) -- ils doivent donc etre construits APRES
        // la fin de l'initialisation, jamais avant.
        val loadingOverlay = findViewById<android.widget.LinearLayout>(R.id.loadingOverlay)
        val loadingStatusText = findViewById<TextView>(R.id.loadingStatusText)

        ensureEdgeServerUrl { wsUrl ->
            val baseUrl = wsUrlToHttpBaseUrl(wsUrl)
            lifecycleScope.launch {
                try {
                    orchestrator = withContext(Dispatchers.IO) {
                        val poseAnchorManager = PoseAnchorManager()
                        LocalizerServiceLocator.setPoseAnchorManagerForRemoteMode(poseAnchorManager)
                        RemoteLocalizationOrchestrator(baseUrl, poseAnchorManager)
                    }
                    virtualObjectRenderer = VirtualObjectRenderer(sceneView, LocalizerServiceLocator.getPoseAnchorManager())
                    loadingOverlay.visibility = android.view.View.GONE
                    setupCollabFeature()
                    // La cascade de localisation ne demarre plus automatiquement en boucle : elle est
                    // desormais declenchee manuellement par le bouton "Localiser ma position", jusqu'a
                    // l'arrivee pres de la cible CRDT (cf. enterManipulationMode / startContinuousCrdtLoop).
                } catch (e: Exception) {
                    loadingStatusText.text = getString(R.string.loading_failed_format, e.message ?: e.toString())
                }
            }
        }
    }

    /**
     * Demande l'adresse du noeud edge (ex. ws://192.168.1.42:8765) une seule fois, memorisee
     * ensuite dans les prefs -- reutilisee a la fois pour /localize (HTTP) et /ws (WebSocket),
     * puisque server.py sert les deux depuis le meme process/port. Remplace l'ancienne demande
     * qui n'avait lieu que dans connectToRelayServer() (trop tard : l'orchestrateur en a besoin
     * avant meme le premier clic sur "Localiser ma position").
     */
    private fun ensureEdgeServerUrl(onReady: (wsUrl: String) -> Unit) {
        val saved = getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE).getString(PREF_WS_URL, null)
        if (saved != null) {
            onReady(saved)
            return
        }
        val input = EditText(this).apply { hint = "ws://192.168.1.42:8765" }
        AlertDialog.Builder(this)
            .setTitle("Connexion au noeud edge")
            .setMessage("Adresse du serveur (localisation + collaboration)")
            .setView(input)
            .setPositiveButton("Connecter") { _, _ ->
                val url = input.text.toString().trim()
                if (url.isNotEmpty()) {
                    getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE).edit().putString(PREF_WS_URL, url).apply()
                    onReady(url)
                }
            }
            // Non annulable : sans adresse de serveur, ni la localisation ni la collaboration
            // ne peuvent fonctionner (contrairement a l'ancien flux ou seule la collab en dependait).
            .setCancelable(false)
            .show()
    }

    /** ws://host:port[/ws] -> http://host:port (server.py expose /localize en HTTP sur le meme port). */
    private fun wsUrlToHttpBaseUrl(wsUrl: String): String =
        wsUrl.trim()
            .replaceFirst(Regex("^ws://"), "http://")
            .replaceFirst(Regex("^wss://"), "https://")
            .removeSuffix("/ws")
            .removeSuffix("/")

    override fun onResume() {
        super.onResume()
        compassProvider.start()
    }

    override fun onPause() {
        super.onPause()
        compassProvider.stop()
    }

    // ==================================================================================
    // Fonctionnalite collaborative : catalogue, mini-carte, sync CRDT
    // ==================================================================================

    private fun setupCollabFeature() {
        Configuration.getInstance().load(applicationContext, androidx.preference.PreferenceManager.getDefaultSharedPreferences(applicationContext))
        Configuration.getInstance().userAgentValue = packageName

        catalogRepository = CollabServiceLocator.getCatalogRepository(applicationContext)
        guidanceController = CollabServiceLocator.getGuidanceController()
        manipulationController = CollabServiceLocator.getManipulationController()

        val miniMapView = findViewById<MapView>(R.id.miniMapView)
        miniMapController = MiniMapController(miniMapView)
        miniMapController.setObjects(catalogRepository.all(), this)
        miniMapController.onObjectSelected = { def -> onObjectSelectedOnMap(def) }

        setupManipulationPanel()
        connectToRelayServer()
    }

    /**
     * L'URL du noeud edge a deja ete obtenue dans onCreate (ensureEdgeServerUrl), avant meme la
     * construction de l'orchestrateur de localisation -- setupCollabFeature() (qui appelle cette
     * fonction) n'est lui-meme execute qu'apres cette construction, donc la valeur est garantie
     * presente ici.
     */
    private fun connectToRelayServer() {
        val savedUrl = getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE).getString(PREF_WS_URL, null)
            ?: return
        // server.py (nouveau) expose le WebSocket sur /ws, contrairement a l'ancien relay_server.py
        // qui l'exposait a la racine -- on complete l'URL saisie si besoin.
        val wsFullUrl = if (savedUrl.trimEnd('/').endsWith("/ws")) savedUrl else savedUrl.trimEnd('/') + "/ws"
        startSync(wsFullUrl)
    }

    private fun startSync(wsUrl: String) {
        val sync = CollabServiceLocator.getSyncRepository()
        sync.connect(wsUrl)

        // Nouveaux objets recus d'autres utilisateurs -> persistes localement + ajoutes a la mini-carte
        lifecycleScope.launch {
            sync.catalogAdditions.collect { additions ->
                additions.values.forEach { catalogRepository.addRemoteObject(it) }
                runOnUiThread { miniMapController.setObjects(catalogRepository.all(), this@ArNavigationActivity) }
            }
        }

        // Etats de manipulation recus -> re-rendu de l'objet concerne
        lifecycleScope.launch {
            sync.states.collect { states ->
                val session = sceneView.session ?: return@collect
                states.values.forEach { state ->
                    catalogRepository.findById(state.objectId)?.let { def ->
                        virtualObjectRenderer?.updateObject(this@ArNavigationActivity, session, def, state)
                    }
                }
            }
        }
    }

    /** Tap sur un objet de la mini-carte : passe en mode guidage vers cet objet. */
    private fun onObjectSelectedOnMap(def: VirtualObjectDefinition) {
        selectedTarget = def
        manipulationTarget = null
        findViewById<LinearLayout>(R.id.guidancePanel).visibility = android.view.View.VISIBLE
        findViewById<LinearLayout>(R.id.manipulationPanel).visibility = android.view.View.GONE
        findViewById<TextView>(R.id.guidanceTargetText).text =
            getString(R.string.guidance_target_format, def.label.ifBlank { def.objectType })

        // Amorce l'etat CRDT de cet objet s'il n'existe pas encore (premiere personne a l'atteindre)
        val sync = CollabServiceLocator.getSyncRepository()
        if (sync.states.value[def.objectId] == null) {
            sync.seedLocalState(
                ObjectManipulationState.default(def.objectId, CollabServiceLocator.currentReplicaId(), def.defaultRotationYDeg, def.defaultColorArgb)
            )
        }
    }

    /** Appele a chaque cycle de la cascade : met a jour la fleche de guidage ou detecte l'arrivee. */
    private fun updateGuidance(pose: GeoPose) {
        val target = selectedTarget ?: run {
            virtualObjectRenderer?.clearArrow()
            return
        }
        val session = sceneView.session ?: return

        val info = guidanceController.computeGuidance(pose, target)
        runOnUiThread {
            findViewById<TextView>(R.id.guidanceDistanceText).text =
                if (info.hasArrived) getString(R.string.guidance_arrived)
                else getString(R.string.guidance_distance_format, info.distanceM)
        }

        if (info.hasArrived) {
            selectedTarget = null
            manipulationTarget = target
            virtualObjectRenderer?.clearArrow()
            runOnUiThread { enterManipulationMode(target) }
            return
        }

        // Positionne la fleche a 2 m devant l'utilisateur, orientee vers la cible
        val userEnu = LocalizerServiceLocator.getPoseAnchorManager().placeVirtualObject(pose.latitude, pose.longitude) ?: return
        val dir = guidanceController.arrowDirectionEnu(info.bearingRad)
        val arrowPos = doubleArrayOf(userEnu[0] + dir[0] * 2.0, userEnu[1] + dir[1] * 2.0, userEnu[2])
        virtualObjectRenderer?.updateGuidanceArrow(session, arrowPos, info.bearingRad)
    }

    // ==================================================================================
    // Manipulation collaborative (rotation, translation, couleur)
    // ==================================================================================

    private fun setupManipulationPanel() {
        findViewById<Button>(R.id.rotateLeftButton).setOnClickListener { manipulationController.rotateBy(-15f) }
        findViewById<Button>(R.id.rotateRightButton).setOnClickListener { manipulationController.rotateBy(15f) }
        findViewById<Button>(R.id.moveForwardButton).setOnClickListener { manipulationController.translateBy(forwardM = 0.2f, rightM = 0f) }
        findViewById<Button>(R.id.moveBackButton).setOnClickListener { manipulationController.translateBy(forwardM = -0.2f, rightM = 0f) }

        val swatchRow = findViewById<LinearLayout>(R.id.colorSwatchRow)
        COLOR_SWATCHES.forEach { argb ->
            val swatch = Button(this).apply {
                setBackgroundColor(argb)
                layoutParams = LinearLayout.LayoutParams(60, 60).apply { setMargins(6, 0, 6, 0) }
                setOnClickListener { manipulationController.setColor(argb) }
            }
            swatchRow.addView(swatch)
        }

        findViewById<Button>(R.id.closeManipulationButton).setOnClickListener {
            manipulationTarget = null
            continuousLoopRunning = false
            findViewById<LinearLayout>(R.id.manipulationPanel).visibility = android.view.View.GONE
            // Symetrique de enterManipulationMode() : on les avait caches, on les remet pour
            // pouvoir relancer une localisation / consulter la carte.
            findViewById<Button>(R.id.localiserPositionButton).visibility = android.view.View.VISIBLE
            findViewById<Button>(R.id.viewMapButton).visibility = android.view.View.VISIBLE
        }
    }

    private fun enterManipulationMode(target: VirtualObjectDefinition) {
        findViewById<LinearLayout>(R.id.guidancePanel).visibility = android.view.View.GONE
        // localiserPositionButton et viewMapButton ne font PAS partie de guidancePanel (vues
        // independantes) : sans cette ligne, ils restaient affiches par-dessus manipulationPanel.
        findViewById<Button>(R.id.localiserPositionButton).visibility = android.view.View.GONE
        findViewById<Button>(R.id.viewMapButton).visibility = android.view.View.GONE
        val panel = findViewById<LinearLayout>(R.id.manipulationPanel)
        panel.visibility = android.view.View.VISIBLE
        findViewById<TextView>(R.id.manipulationTitleText).text =
            getString(R.string.manipulation_title_format, target.label.ifBlank { target.objectType })

        val sync = CollabServiceLocator.getSyncRepository()
        val currentState = sync.states.value[target.objectId]
            ?: ObjectManipulationState.default(target.objectId, CollabServiceLocator.currentReplicaId(), target.defaultRotationYDeg, target.defaultColorArgb)
        manipulationController.bindObject(currentState)

        // Une fois arrive pres de la cible, la localisation continue "comme actuellement"
        // (echantillonnage toutes les 5 s) pour garder le repere CRDT recale pendant la manipulation.
        startContinuousCrdtLoop()
    }

    // ==================================================================================
    // Localisation manuelle : bouton "Localiser ma position" (declenche la cascade UNE fois,
    // journalise le detail des etapes dans le fichier txt) + boucle continue CRDT (inchangee dans
    // son principe, mais desormais active uniquement une fois arrive pres de la cible).
    // ==================================================================================

    /** Un seul cycle de cascade, declenche par un clic sur "Localiser ma position". */
    private fun onLocaliserMaPositionClicked() {
        if (!::orchestrator.isInitialized) return
        lifecycleScope.launch {
            val frame = currentArCoreFrame()
            if (frame == null || frame.camera.trackingState != TrackingState.TRACKING) {
                statusView.text = getString(R.string.ar_tracking_lost)
                return@launch
            }
            val bitmap = captureCameraImage(frame) ?: return@launch
            val heading = if (compassProvider.isReady()) compassProvider.currentHeadingRad() else headingFromArCorePose(frame)
            val arCorePose = poseFromArCore(frame)
            val focalPx = focalLengthPxFromArCore(frame)
            val intrinsics = cameraIntrinsicsFromArCore(frame)

            // logDetails = true : ce cycle alimente le fichier txt (details des etapes), en plus
            // de la bande de statut habituelle.
            val result = runCascadeAndRecord(bitmap, heading, arCorePose, focalPx, intrinsics, logDetails = true)
            applyLocalizationResult(result)
        }
    }

    /** Boucle continue (5 s), demarree une seule fois a l'arrivee pres de la cible CRDT. */
    private fun startContinuousCrdtLoop() {
        if (continuousLoopRunning) return
        continuousLoopRunning = true
        lifecycleScope.launch {
            while (continuousLoopRunning && manipulationTarget != null) {
                val frame = currentArCoreFrame()
                if (frame != null && frame.camera.trackingState == TrackingState.TRACKING) {
                    val bitmap = captureCameraImage(frame)
                    val heading = if (compassProvider.isReady()) compassProvider.currentHeadingRad() else headingFromArCorePose(frame)
                    val arCorePose = poseFromArCore(frame)

                    if (bitmap != null) {
                        val focalPx = focalLengthPxFromArCore(frame)
                        val intrinsics = cameraIntrinsicsFromArCore(frame)
                        // logDetails = false : la phase CRDT continue n'alourdit pas le fichier txt,
                        // qui reste dedie aux clics explicites sur "Localiser ma position".
                        val result = runCascadeAndRecord(bitmap, heading, arCorePose, focalPx, intrinsics, logDetails = false)
                        applyLocalizationResult(result)
                    }
                } else {
                    runOnUiThread { statusView.text = getString(R.string.ar_tracking_lost) }
                }
                delay(LOCALIZATION_PERIOD_MS)
            }
            continuousLoopRunning = false
        }
    }

    /** Met a jour la bande de statut, la mini-carte, la scene RA et le guidage a partir d'un resultat de cascade. */
    private fun applyLocalizationResult(result: LocalizationOrchestrator.CascadeResult) {
        latestPose = result.finalPose
        runOnUiThread {
            statusView.text = getString(
                R.string.localization_status_format,
                result.finalPose.level, result.finalPose.latitude, result.finalPose.longitude, result.finalPose.sigmaPosMeters
            )
            miniMapController.updateUserPosition(result.finalPose.latitude, result.finalPose.longitude, this@ArNavigationActivity, recenter = true)
        }
        updateArScene(result)
        updateGuidance(result.finalPose)
    }

    private fun currentArCoreFrame(): Frame? {
        return sceneView.frame
    }

    private fun captureCameraImage(frame: Frame): Bitmap? {
        return try {
            val image = frame.acquireCameraImage()
            val bitmap = ImageUtils.yuv420888ToBitmap(image)
            image.close()
            bitmap
        } catch (e: NotYetAvailableException) {
            null
        } catch (e: Exception) {
            null
        }
    }

    private fun poseFromArCore(frame: Frame): Pose3D {
        val pose = frame.camera.pose
        val t = pose.translation
        val q = pose.rotationQuaternion
        val quaternion = Quaternion(qw = q[3].toDouble(), qx = q[0].toDouble(), qy = q[1].toDouble(), qz = q[2].toDouble())
        return Pose3D(quaternion.toRotationMatrix(), doubleArrayOf(t[0].toDouble(), t[1].toDouble(), t[2].toDouble()))
    }

    private fun headingFromArCorePose(frame: Frame): Double {
        val zAxis = frame.camera.pose.zAxis
        return kotlin.math.atan2((-zAxis[0]).toDouble(), (-zAxis[2]).toDouble())
    }

    private fun focalLengthPxFromArCore(frame: Frame): Double? {
        return try {
            frame.camera.imageIntrinsics.focalLength[0].toDouble()
        } catch (e: Exception) {
            null
        }
    }

    private suspend fun runCascadeAndRecord(
        frameBitmap: Bitmap,
        heading: Double,
        arCorePose: Pose3D,
        focalPx: Double?,
        cameraIntrinsics: DoubleArray?,
        logDetails: Boolean = false
    ): LocalizationOrchestrator.CascadeResult {
        val result = orchestrator.localize(frameBitmap, heading, arCorePose, focalPx, cameraIntrinsics)
        val timestampMs = System.currentTimeMillis()
        LocalizationHistoryRepository.add(
            LocalizationRecord(
                timestampMs = timestampMs,
                level1 = result.level1Pose,
                level2 = result.level2Pose,
                level3 = result.level3Pose
            )
        )

        // Journal texte detaille (Niveau 1/2/3), uniquement pour les cycles declenches par
        // "Localiser ma position" -- ecrit/complete immediatement le fichier dans Downloads pour
        // que son contenu augmente a chaque clic, comme demande.
        if (logDetails) {
            cascadeSequenceCounter++
            val entryText = CascadeLogFormatter.format(cascadeSequenceCounter, timestampMs, result)
            CascadeLogRepository.add(entryText)
            kotlinx.coroutines.withContext(kotlinx.coroutines.Dispatchers.IO) {
                TxtLogExporter.appendToDownloadsFile(applicationContext, entryText)
            }
        }
        return result
    }

    /**
     * Intrinseques completes (fx, fy, cx, cy) de l'image CPU ARCore reellement analysee par la
     * cascade (Frame.acquireCameraImage(), meme flux que Frame.camera.imageIntrinsics -- pas de
     * mise a l'echelle necessaire). CRITIQUE pour le Niveau 3 : sans ces vraies valeurs, PgPnpEngine
     * utilise des valeurs par defaut (1920x1080 supposees) qui faussent systematiquement toute
     * erreur de reprojection et empechent RANSAC de jamais valider un inlier (voir
     * PgPnpEngine.updateIntrinsics et Level3Refiner.refine).
     */
    private fun cameraIntrinsicsFromArCore(frame: Frame): DoubleArray? {
        return try {
            val intrinsics = frame.camera.imageIntrinsics
            val focal = intrinsics.focalLength      // (fx, fy)
            val principal = intrinsics.principalPoint // (cx, cy)
            val dims = intrinsics.imageDimensions    // (width, height) du flux CPU ARCore reellement utilise

            android.util.Log.d(
                "ArCoreIntrinsics",
                "Flux ARCore CPU : ${dims[0]}x${dims[1]} px, fx=${focal[0]}, fy=${focal[1]}, " +
                        "cx=${principal[0]}, cy=${principal[1]} (photos COLMAP d'entrainement : 2448x3264 px)"
            )

            doubleArrayOf(focal[0].toDouble(), focal[1].toDouble(), principal[0].toDouble(), principal[1].toDouble())
        } catch (e: Exception) {
            null
        }
    }

    /**
     * Affiche le resultat de pose courant dans la scene RA avec le modele 3D assets/models/arrow.glb
     * (au lieu d'une simple sphere), oriente selon le yaw de la pose retenue -- appele apres chaque
     * resultat de localisation (clic manuel ou cycle continu de la phase CRDT).
     */
    private fun updateArScene(result: LocalizationOrchestrator.CascadeResult) {
        val session = sceneView.session ?: return
        val poseAnchorManager = LocalizerServiceLocator.getPoseAnchorManager()
        val arLocalPosition = poseAnchorManager.placeVirtualObject(result.finalPose.latitude, result.finalPose.longitude) ?: return

        try {
            val (yawDeg, _, _) = result.finalPose.orientation.toEulerZYXDeg()
            val arCorePose = com.google.ar.core.Pose(
                floatArrayOf(arLocalPosition[0].toFloat(), arLocalPosition[1].toFloat(), arLocalPosition[2].toFloat()),
                yawToQuaternionFloatArray(yawDeg.toFloat())
            )
            val anchor: Anchor = session.createAnchor(arCorePose)

            navigationAnchorNode?.let { sceneView.removeChildNode(it) }
            val anchorNode = AnchorNode(sceneView.engine, anchor)

            // NOTE ADAPTATION SCENEVIEW : voir la note detaillee dans VirtualObjectRenderer.kt --
            // la logique de position/orientation ci-dessus est stable, seule cette creation de noeud
            // peut necessiter un ajustement selon votre version de sceneview-android.
            val modelInstance = positionMarkerModelLoader.createModelInstance(ModelCatalog.ARROW_ASSET)
            val modelNode = ModelNode(modelInstance = modelInstance, scaleToUnits = 0.3f)
            anchorNode.addChildNode(modelNode)
            sceneView.addChildNode(anchorNode)
            navigationAnchorNode = anchorNode
        } catch (e: Exception) {
            // Creation d'ancre ARCore echouee (tracking limite, etc.) : on conserve l'ancre precedente.
        }
    }

    private fun yawToQuaternionFloatArray(yawDeg: Float): FloatArray {
        val halfRad = Math.toRadians(yawDeg.toDouble()) / 2.0
        return floatArrayOf(0f, Math.sin(halfRad).toFloat(), 0f, Math.cos(halfRad).toFloat())
    }
}