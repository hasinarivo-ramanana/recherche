# Câblage dans ArNavigationActivity.kt (diffs, pas un fichier complet)

## 1. Type du champ orchestrateur
```kotlin
// AVANT
private lateinit var orchestrator: LocalizationOrchestrator
// APRES
private lateinit var orchestrator: ILocalizationOrchestrator
```

## 2. Construction (dans onCreate, à la place de LocalizerServiceLocator.getOrchestrator)
```kotlin
lifecycleScope.launch {
    try {
        val baseUrl = getEdgeServerBaseUrl() // ex: lu depuis les mêmes prefs que ws_url,
                                              // http://<host>:8765 au lieu de ws://<host>:8765
        orchestrator = withContext(Dispatchers.IO) {
            // PoseAnchorManager reste géré localement (il n'a rien à voir avec la cascade,
            // c'est le recalage AR local -> WGS). On le construit avant, comme aujourd'hui,
            // et on l'injecte dans RemoteLocalizationOrchestrator.
            val poseAnchorManager = PoseAnchorManager()
            LocalizerServiceLocator.setPoseAnchorManagerForRemoteMode(poseAnchorManager) // à ajouter, cf. note plus bas
            RemoteLocalizationOrchestrator(baseUrl, poseAnchorManager)
        }
        virtualObjectRenderer = VirtualObjectRenderer(sceneView, LocalizerServiceLocator.getPoseAnchorManager())
        loadingOverlay.visibility = android.view.View.GONE
        setupCollabFeature()
    } catch (e: Exception) {
        loadingStatusText.text = getString(R.string.loading_failed_format, e.message ?: e.toString())
    }
}
```

Le reste du fichier (onLocaliserMaPositionClicked, runCascadeAndRecord, applyLocalizationResult,
startContinuousCrdtLoop, updateArScene, updateGuidance) **ne change pas** : ils appellent
`orchestrator.localize(...)` et lisent un `LocalizationOrchestrator.CascadeResult`, exactement
comme avant — c'est tout l'intérêt de l'interface `ILocalizationOrchestrator`.

## 3. LocalizerServiceLocator.kt
Vous n'avez plus besoin de charger les modèles TFLite/COLMAP en local sur le téléphone
(APR, YOLO, nuage de points, database.db...) — c'est justement ce qui causait l'ANR corrigé
par le `Dispatchers.IO` dans votre code, et l'obligation de la permission "Accès à tous les
fichiers". En basculant vers le mode distant, vous pouvez :
- soit supprimer entièrement `build()` et la construction des niveaux 1/2/3 côté Kotlin,
- soit les garder comme repli hors-ligne optionnel (si un jour vous voulez un mode "pas de
  réseau"), en conservant `LocalizationOrchestrator` (implémente aussi `ILocalizationOrchestrator`)
  à côté de `RemoteLocalizationOrchestrator`.

Ajoutez juste une fonction pour partager le PoseAnchorManager en mode distant :
```kotlin
object LocalizerServiceLocator {
    @Volatile private var poseAnchorManager: PoseAnchorManager? = null

    fun setPoseAnchorManagerForRemoteMode(pam: PoseAnchorManager) { poseAnchorManager = pam }
    fun getPoseAnchorManager(): PoseAnchorManager =
        poseAnchorManager ?: throw IllegalStateException("PoseAnchorManager non initialisé")
}
```

## 4. Réutiliser la même IP pour /localize et /ws
Votre dialogue `connectToRelayServer()` demande déjà `ws://<ip>:8765`. Comme le nouveau
`server.py` sert les deux sur le même port, dérivez simplement l'URL HTTP à partir de celle-ci :
```kotlin
private fun getEdgeServerBaseUrl(): String {
    val wsUrl = getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE).getString(PREF_WS_URL, null)
        ?: throw IllegalStateException("URL du noeud edge non configurée")
    return wsUrl.replaceFirst("ws://", "http://").removeSuffix("/ws")
}
```
(Adaptez si vous saisissez déjà juste `ip:port` sans `/ws`.)

## 5. build.gradle (app)
OkHttp est probablement déjà présent en transitif (Retrofit, etc.) ; sinon ajoutez :
```gradle
implementation "com.squareup.okhttp3:okhttp:4.12.0"
```

## 6. Manifeste
Si le serveur tourne en HTTP simple (non chiffré) sur le réseau local, comme c'est déjà
implicitement le cas pour `ws://` :
```xml
<application android:usesCleartextTraffic="true" ...>
```
(probablement déjà présent, sinon requis aussi pour le WebSocket actuel).

## 7. Transition automatique vers le mode collaboratif à 10 m
C'est déjà géré par le code existant : `updateGuidance()` appelle
`guidanceController.computeGuidance(pose, target)` puis, si `info.hasArrived` (rayon
d'arrivée configuré dans `RouteGuidanceController`, à vérifier qu'il est bien réglé à 10 m),
bascule vers `enterManipulationMode()` qui démarre `startContinuousCrdtLoop()`. Aucun
changement nécessaire ici — cette logique est indépendante de local vs distant, elle
consomme juste `result.finalPose`, qui a exactement la même forme dans les deux cas.
