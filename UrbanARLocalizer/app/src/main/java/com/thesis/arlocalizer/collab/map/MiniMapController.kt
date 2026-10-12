package com.thesis.arlocalizer.collab.map

import android.content.Context
import com.thesis.arlocalizer.collab.model.ModelCatalog
import com.thesis.arlocalizer.collab.model.VirtualObjectDefinition
import com.thesis.arlocalizer.map.CircleMarkerFactory
import org.osmdroid.tileprovider.tilesource.TileSourceFactory
import org.osmdroid.util.GeoPoint
import org.osmdroid.views.MapView
import org.osmdroid.views.overlay.Marker

/**
 * Pilote la mini-carte OSM affichee en haut a droite de l'ecran RA (cf. activity_ar_navigation.xml,
 * vue `miniMapView`). Affiche les objets du catalogue (couleur par type, cf. ModelCatalog) et la
 * position courante de l'utilisateur (point cyan, reutilise CircleMarkerFactory du Niveau carte
 * principale). Un tap sur un objet declenche le callback `onObjectSelected`.
 */
class MiniMapController(private val mapView: MapView) {

    var onObjectSelected: ((VirtualObjectDefinition) -> Unit)? = null

    private var userMarker: Marker? = null
    private val objectMarkers = mutableMapOf<String, Marker>()

    init {
        mapView.setTileSource(TileSourceFactory.MAPNIK)
        // Mini-carte : pas de zoom/pan libre pour rester compacte et lisible (juste un aperçu spatial).
        mapView.setMultiTouchControls(false)
        mapView.setBuiltInZoomControls(false)
        mapView.controller.setZoom(19.0)
        // Centre par defaut sur la zone d'entrainement (Campus) en attendant la premiere localisation
        mapView.controller.setCenter(GeoPoint(-18.91459, 47.55227))
    }

    fun setObjects(objects: List<VirtualObjectDefinition>, context: Context) {
        objectMarkers.values.forEach { mapView.overlays.remove(it) }
        objectMarkers.clear()

        for (obj in objects) {
            val marker = Marker(mapView)
            marker.position = GeoPoint(obj.latitude, obj.longitude)
            marker.icon = CircleMarkerFactory.forColor(context, ModelCatalog.markerColorFor(obj.objectType), diameterDp = 14f)
            marker.setAnchor(Marker.ANCHOR_CENTER, Marker.ANCHOR_CENTER)
            marker.title = obj.label.ifBlank { obj.objectId }
            marker.setOnMarkerClickListener { _, _ ->
                onObjectSelected?.invoke(obj)
                true
            }
            objectMarkers[obj.objectId] = marker
            mapView.overlays.add(marker)
        }
        mapView.invalidate()
    }

    fun updateUserPosition(lat: Double, lon: Double, context: Context, recenter: Boolean = true) {
        val point = GeoPoint(lat, lon)
        if (userMarker == null) {
            userMarker = Marker(mapView).apply {
                icon = CircleMarkerFactory.forColor(context, android.graphics.Color.CYAN, diameterDp = 16f)
                setAnchor(Marker.ANCHOR_CENTER, Marker.ANCHOR_CENTER)
                title = "Vous"
            }
            mapView.overlays.add(userMarker)
        }
        userMarker?.position = point
        if (recenter) mapView.controller.setCenter(point)
        mapView.invalidate()
    }

    fun refresh() = mapView.invalidate()
}
