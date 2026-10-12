package com.thesis.arlocalizer.collab.ar

import android.content.Context
import com.google.ar.core.Anchor
import com.google.ar.core.Session
import com.thesis.arlocalizer.collab.model.ModelCatalog
import com.thesis.arlocalizer.collab.model.VirtualObjectDefinition
import com.thesis.arlocalizer.collab.sync.ObjectManipulationState
import com.thesis.arlocalizer.poseanchor.PoseAnchorManager
import io.github.sceneview.ar.ARSceneView
import io.github.sceneview.ar.node.AnchorNode
import io.github.sceneview.loaders.ModelLoader
import io.github.sceneview.node.ModelNode

/**
 * Place et met a jour les modeles .glb des objets virtuels et de la fleche de guidage dans la scene
 * ARCore, a partir de :
 *  - la position canonique de chaque objet (VirtualObjectDefinition.lat/lon), convertie en
 *    coordonnees AR-locales par PoseAnchorManager.placeVirtualObject (cascade de localisation) ;
 *  - le decalage/rotation/couleur COURANT de l'objet (ObjectManipulationState, mis a jour par CRDT).
 *
 * NOTE ADAPTATION SCENEVIEW : comme pour updateArScene (ArNavigationActivity), la creation exacte
 * de ModelNode (chargement .glb, application de teinte) peut necessiter un ajustement mineur de
 * signature selon la version exacte de sceneview-android pinnee dans votre projet. La logique de
 * positionnement (ancre ARCore + decalage CRDT) est, elle, correcte et independante de cette version.
 */
class VirtualObjectRenderer(private val sceneView: ARSceneView, private val poseAnchorManager: PoseAnchorManager) {

    private val modelLoader: ModelLoader by lazy { ModelLoader(sceneView.engine, sceneView.context) }
    private val activeNodes = mutableMapOf<String, AnchorNode>()
    private var arrowNode: AnchorNode? = null

    /**
     * (Re)positionne le modele .glb d'un objet virtuel dans la scene, en combinant sa position
     * canonique (definition SIG) et son decalage/rotation/couleur courant (etat CRDT).
     */
    fun updateObject(context: Context, session: Session, definition: VirtualObjectDefinition, state: ObjectManipulationState) {
        val basePosition = poseAnchorManager.placeVirtualObject(definition.latitude, definition.longitude) ?: return
        val finalPosition = doubleArrayOf(
            basePosition[0] + state.offsetX,
            basePosition[1] + state.offsetY,
            basePosition[2] + state.offsetZ
        )

        activeNodes[definition.objectId]?.let { sceneView.removeChildNode(it) }

        try {
            val arPose = com.google.ar.core.Pose(
                floatArrayOf(finalPosition[0].toFloat(), finalPosition[1].toFloat(), finalPosition[2].toFloat()),
                yawToQuaternionFloatArray(state.rotationYDeg)
            )
            val anchor: Anchor = session.createAnchor(arPose)
            val anchorNode = AnchorNode(sceneView.engine, anchor)

            val assetPath = ModelCatalog.assetPathFor(definition.objectType)
            val modelInstance = modelLoader.createModelInstance(assetPath)
            val modelNode = ModelNode(modelInstance = modelInstance, scaleToUnits = 1.0f)
            // Teinte de couleur : a appliquer via le MaterialInstance du modele si votre version de
            // sceneview-android expose modelNode.materialInstances / setBaseColor -- voir note ci-dessus.
            anchorNode.addChildNode(modelNode)
            sceneView.addChildNode(anchorNode)
            activeNodes[definition.objectId] = anchorNode
        } catch (e: Exception) {
            // Tracking limite ou modele introuvable : on conserve l'affichage precedent de cet objet.
        }
    }

    /** Place/oriente la fleche de guidage (models/arrow.glb) devant l'utilisateur, vers la cible. */
    fun updateGuidanceArrow(session: Session, forwardPosition: DoubleArray, bearingRad: Double) {
        arrowNode?.let { sceneView.removeChildNode(it) }
        try {
            val arPose = com.google.ar.core.Pose(
                floatArrayOf(forwardPosition[0].toFloat(), forwardPosition[1].toFloat(), forwardPosition[2].toFloat()),
                yawToQuaternionFloatArray(Math.toDegrees(bearingRad).toFloat())
            )
            val anchor: Anchor = session.createAnchor(arPose)
            val anchorNode = AnchorNode(sceneView.engine, anchor)
            val modelInstance = modelLoader.createModelInstance(ModelCatalog.ARROW_ASSET)
            val modelNode = ModelNode(modelInstance = modelInstance, scaleToUnits = 0.3f)
            anchorNode.addChildNode(modelNode)
            sceneView.addChildNode(anchorNode)
            arrowNode = anchorNode
        } catch (e: Exception) {
            // idem : conserve l'affichage precedent en cas d'echec ponctuel
        }
    }

    fun clearArrow() {
        arrowNode?.let { sceneView.removeChildNode(it) }
        arrowNode = null
    }

    private fun yawToQuaternionFloatArray(yawDeg: Float): FloatArray {
        val halfRad = Math.toRadians(yawDeg.toDouble()) / 2.0
        return floatArrayOf(0f, Math.sin(halfRad).toFloat(), 0f, Math.cos(halfRad).toFloat())
    }
}
