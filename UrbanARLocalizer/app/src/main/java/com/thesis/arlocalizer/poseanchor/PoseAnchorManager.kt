package com.thesis.arlocalizer.poseanchor

import com.thesis.arlocalizer.model.GeoPose
import com.thesis.arlocalizer.model.Pose3D
import com.thesis.arlocalizer.util.GeoUtils
import kotlin.math.sqrt

/**
 * Contribution "PoseAnchor" (Chapitre 4) : gere la transformation rigide T_AR->WGS entre le repere
 * local ARCore (initialise arbitrairement au demarrage de la session) et le repere geographique WGS-84
 * (approxime localement par un plan ENU centre sur la premiere position connue), la corrige au fil de
 * l'eau via HybridKalmanSE3, et fournit le placement d'objets virtuels geo-referencies ainsi que la
 * strategie d'ancre unique par zone (Eq. ancre_relative).
 *
 * Point cle (corrige par rapport a une version simplifiee anterieure) : la composition de T_AR->WGS
 * utilise la VRAIE pose caméra ARCore (T_cam->AR, rapportee par Frame.camera.pose) a chaque mesure,
 * et non une identification abusive du repere AR-local avec le repere ENU. Cf. Eqs. R_init / R_ENU /
 * ecef du manuscrit, generalisees ici a une recomposition continue :
 *
 *   T_AR->WGS = T_cam->WGS  ∘  (T_cam->AR)^-1
 *
 * ou T_cam->WGS est la pose de la camera dans le repere ENU telle que fournie par la cascade de
 * localisation (Niveau 1/2/3), et T_cam->AR est la pose de la meme camera, au meme instant, dans le
 * repere local ARCore (obtenue via Frame.camera.pose).
 */
class PoseAnchorManager(private val zoneRadiusM: Double = 20.0) {

    private var kalman: HybridKalmanSE3? = null
    private var originLat: Double = 0.0
    private var originLon: Double = 0.0

    /** Zones deja ancrees : cle = identifiant de zone (grille), valeur = position ENU de l'ancre. */
    private val activeZoneAnchors = mutableMapOf<String, DoubleArray>()

    val isInitialized: Boolean get() = kalman != null

    /**
     * Initialise T_AR->WGS a partir de la premiere pose fiable de la cascade et de la pose caméra
     * ARCore correspondante (capturee au meme instant). L'origine du repere ENU local est fixee sur
     * cette premiere position geographique.
     */
    fun initialize(firstGeoPose: GeoPose, firstArCorePose: Pose3D) {
        originLat = firstGeoPose.latitude
        originLon = firstGeoPose.longitude

        // Pose de la camera dans le repere ENU (origine = elle-meme, translation nulle par construction)
        val tCamToWgs0 = Pose3D(firstGeoPose.orientation.toRotationMatrix(), doubleArrayOf(0.0, 0.0, 0.0))

        val tArToWgs0 = tCamToWgs0.compose(firstArCorePose.inverse())
        kalman = HybridKalmanSE3(tArToWgs0, initialUncertaintyM = firstGeoPose.sigmaPosMeters)
    }

    /**
     * Recalage periodique (boucle "toutes les 5 s") : combine une nouvelle pose de la cascade avec la
     * pose caméra ARCore capturee au meme instant pour obtenir une mesure de T_AR->WGS, puis met a jour
     * le filtre de Kalman (Eqs. xi_update_simple - P_update).
     */
    fun onNewLocalization(pose: GeoPose, currentArCorePose: Pose3D) {
        val k = kalman ?: run { initialize(pose, currentArCorePose); return }

        val measuredEnu = GeoUtils.geodeticToLocalMeters(pose.latitude, pose.longitude, originLat, originLon)
        val tCamToWgsMeasured = Pose3D(pose.orientation.toRotationMatrix(), doubleArrayOf(measuredEnu[0], measuredEnu[1], 0.0))
        val tArToWgsMeasured = tCamToWgsMeasured.compose(currentArCorePose.inverse())

        k.update(tArToWgsMeasured, pose.sigmaPosMeters)
    }

    /** Transformation courante T_AR->WGS, filtree par Kalman (repere ENU local, origine = premiere position). */
    fun currentTransform(): Pose3D? = kalman?.currentPose()

    /**
     * Position/orientation ENU courante de la camera, en appliquant la transformation filtree a la
     * derniere pose caméra ARCore connue (mise a jour en continu par le VIO d'ARCore, contrairement au
     * recalage qui n'intervient que toutes les 5 s) : T_cam->WGS(now) = T_AR->WGS ∘ T_cam->AR(now).
     */
    fun currentCameraPoseEnu(liveArCorePose: Pose3D): Pose3D? = currentTransform()?.compose(liveArCorePose)

    /**
     * Placement d'un objet virtuel a une position geographique (Eq. placement_virtuel) :
     * X_AR = (T_AR->WGS)^-1 * X_WGS_local
     */
    fun placeVirtualObject(targetLat: Double, targetLon: Double): DoubleArray? {
        val t = currentTransform() ?: return null
        val targetEnu = GeoUtils.geodeticToLocalMeters(targetLat, targetLon, originLat, originLon)
        val targetEnu3 = doubleArrayOf(targetEnu[0], targetEnu[1], 0.0)
        return t.inverse().apply(targetEnu3)
    }

    /**
     * Strategie d'ancre unique par zone (Eq. ancre_relative) : retourne l'identifiant de zone et la
     * position relative a l'ancre de cette zone, en creant une nouvelle ancre si necessaire.
     */
    fun resolveZoneAnchor(objectEnu: DoubleArray): Pair<String, DoubleArray> {
        for ((zoneId, anchorPos) in activeZoneAnchors) {
            val dx = objectEnu[0] - anchorPos[0]; val dy = objectEnu[1] - anchorPos[1]
            if (sqrt(dx * dx + dy * dy) <= zoneRadiusM) {
                return zoneId to doubleArrayOf(objectEnu[0] - anchorPos[0], objectEnu[1] - anchorPos[1], objectEnu.getOrElse(2) { 0.0 })
            }
        }
        val newZoneId = "zone_${activeZoneAnchors.size}"
        activeZoneAnchors[newZoneId] = objectEnu
        return newZoneId to doubleArrayOf(0.0, 0.0, 0.0)
    }

    fun reset() {
        kalman = null
        activeZoneAnchors.clear()
    }
}
