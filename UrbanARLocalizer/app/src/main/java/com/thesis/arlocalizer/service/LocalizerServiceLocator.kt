package com.thesis.arlocalizer.service

import android.content.Context
import com.thesis.arlocalizer.level1_apr.AprEngine
import com.thesis.arlocalizer.level2_yolosig.Level2Refiner
import com.thesis.arlocalizer.level2_yolosig.SigRepository
import com.thesis.arlocalizer.level2_yolosig.YoloDetector
import com.thesis.arlocalizer.level3_pgpnp.*
import com.thesis.arlocalizer.model.Point3D
import com.thesis.arlocalizer.orchestrator.LocalizationOrchestrator
import com.thesis.arlocalizer.poseanchor.PoseAnchorManager
import org.opencv.android.OpenCVLoader
import android.util.Log

/**
 * Point d'assemblage unique de tous les modules de la cascade, a partir des fichiers deja fournis
 * par l'utilisateur (places dans app/src/main/assets/, voir README.md) :
 *   - apr_model.tflite + apr_norm_params.json   (Niveau 1, deja entraine)
 *   - yolo_model.tflite                          (Niveau 2, deja entraine)
 *   - sig_data.geojson                           (Niveau 2, base SIG)
 *   - points3D.ply                               (Niveau 3, nuage COLMAP -- geometrie)
 *   - points3D.txt                                (Niveau 3, index (image_id,point2D_idx)->point3D)
 *   - images.txt                                  (Niveau 3, poses camera COLMAP -> index spatial)
 *   - database.db                                (Niveau 3, descripteurs SIFT originaux COLMAP)
 *
 * Singleton applicatif simple ; a remplacer par une injection de dependances (Hilt/Koin) dans un
 * projet de production.
 */
object LocalizerServiceLocator {

    @Volatile private var orchestrator: LocalizationOrchestrator? = null
    @Volatile private var poseAnchorManager: PoseAnchorManager? = null

    fun getOrchestrator(context: Context): LocalizationOrchestrator {
        if (!OpenCVLoader.initLocal()) {
            Log.e("LocalizerServiceLocator", "OpenCV initLocal failed!")
        }
        return orchestrator ?: synchronized(this) {
            orchestrator ?: build(context).also { orchestrator = it }
        }
    }

    /** Instance partagee du PoseAnchorManager utilise par l'orchestrateur (meme etat, meme filtre de Kalman). */
    fun getPoseAnchorManager(): PoseAnchorManager =
        poseAnchorManager ?: throw IllegalStateException("Appeler getOrchestrator(context) avant getPoseAnchorManager()")

    /**
     * Mode cascade DEPORTEE (RemoteLocalizationOrchestrator, cf. ArNavigationActivity) : le
     * PoseAnchorManager n'est plus construit par build() (qui n'est alors jamais appele, puisque
     * getOrchestrator(context) ne l'est plus), donc getPoseAnchorManager() lancerait toujours
     * IllegalStateException sans cet appel explicite pour l'enregistrer depuis l'exterieur.
     */
    fun setPoseAnchorManagerForRemoteMode(pam: PoseAnchorManager) {
        poseAnchorManager = pam
    }
    private fun build(context: Context): LocalizationOrchestrator {
        // ---- Niveau 1 : APR ----
        val aprEngine = AprEngine(context)

        // ---- Niveau 2 : YOLO + SIG ----
        val yoloDetector = YoloDetector(context)
        val sigRepository = SigRepository.loadFromAssets(context)
        val level2Refiner = Level2Refiner(yoloDetector, sigRepository)

        // ---- Niveau 3 : PG-PnP ----

        // Origine du repere ENU local : centroide REEL de vos 1123 positions d'entrainement
        // (memes valeurs que lat_mean/lon_mean de apr_norm_params.json). A ajuster uniquement si
        // vous preferez un autre point de reference explicite (ex. entree principale du campus).
        val originLat = -18.914389024933215
        val originLon = 47.55198071386168

        val pointCloud: List<Point3D> = PlyPointCloud.loadFromAssets(context)
        // Choisissez la strategie adaptee a votre distribution de database.db (voir
        // ColmapDatabaseReader.kt pour le detail des 3 options) :
        //   - fromAssets(context)                         : fichier embarque dans assets/ (petits fichiers uniquement)
        //   - fromLocalDownloadsFolder(context)            : fichier place manuellement dans Download/arlocalizer/
        //   - fromUrl(context, url) [suspend, coroutine]   : telecharge une fois depuis un hebergement statique
        // Par defaut ici : Download/arlocalizer/ (pas de reseau requis, pas de duplication APK).
        val dbReader = ColmapDatabaseReader.fromLocalDownloadsFolder(context)

        // Alignement SfM -> ENU : identite si points3D est deja georegistre (ex. via `colmap
        // model_aligner` avec vos positions GPS). Sinon, remplacer par la transformation de
        // similarite calibree (echelle + rotation + translation).
        val sfmToEnuAligner = SfmToEnuAligner.identity()

        // Index point-image -> point-3D, construit depuis points3D.txt (voir ColmapPoints3DIndexLoader)
        val point3DIndex: Map<Pair<Long, Int>, Point3D> =
            ColmapPoints3DIndexLoader.loadFromAssets(context)

        // Index spatial des images de reference, construit depuis images.txt (voir ColmapImageIndexLoader)
        val referenceImages = ColmapImageIndexLoader.loadFromAssets(
            context = context,
            aligner = sfmToEnuAligner,
            originLat = originLat,
            originLon = originLon
        )
        val spatialImageIndex = SpatialImageIndex(referenceImages)
        val referenceImageIdsProvider: (Double, Double) -> List<Long> = { lat, lon ->
            spatialImageIndex.query(lat, lon)
        }

        // Extraction SIFT reelle via OpenCV (necessite le module opencv, voir README section 4)
        val siftExtractor: SiftExtractor = OpenCvSiftExtractor()

        val correspondenceBuilder = SiftCorrespondenceBuilder(siftExtractor, dbReader)
        val pgPnpEngine = PgPnpEngine()
        val level3Refiner = Level3Refiner(
            pointCloud = pointCloud,
            pgPnpEngine = pgPnpEngine,
            correspondenceBuilder = correspondenceBuilder,
            point3DIndex = point3DIndex,
            referenceImageIdsProvider = referenceImageIdsProvider,
            enuOriginLat = originLat,
            enuOriginLon = originLon
        )

        // ---- PoseAnchor ----
        val poseAnchorManagerInstance = PoseAnchorManager()
        poseAnchorManager = poseAnchorManagerInstance

        return LocalizationOrchestrator(aprEngine, level2Refiner, level3Refiner, poseAnchorManagerInstance)
    }
}
