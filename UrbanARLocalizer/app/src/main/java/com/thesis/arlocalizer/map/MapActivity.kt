package com.thesis.arlocalizer.map

import android.os.Bundle
import android.widget.Button
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.preference.PreferenceManager
import com.thesis.arlocalizer.R
import com.thesis.arlocalizer.repository.CascadeLogRepository
import com.thesis.arlocalizer.repository.LocalizationHistoryRepository
import com.thesis.arlocalizer.repository.LocalizationRecord
import com.thesis.arlocalizer.util.CsvExporter
import com.thesis.arlocalizer.util.MapSnapshotExporter
import com.thesis.arlocalizer.util.TxtLogExporter
import org.osmdroid.config.Configuration
import org.osmdroid.tileprovider.tilesource.TileSourceFactory
import org.osmdroid.util.GeoPoint
import org.osmdroid.views.MapView
import org.osmdroid.views.overlay.Marker

/**
 * Ecran "carte" : affiche une carte OpenStreetMap et accumule, au fil de la navigation, les positions
 * estimees par chacun des 3 niveaux de la cascade (Niveau 1 = rouge, Niveau 2 = bleu, Niveau 3 = vert),
 * cf. LocalizationHistoryRepository alimente par ArNavigationActivity toutes les 5 secondes.
 *
 * Deux actions disponibles :
 *  - "Enregistrer la carte" : capture la carte (avec tous les marqueurs) en image PNG (galerie)
 *  - "Exporter CSV"          : exporte l'historique complet (position + quaternion + Euler) en CSV
 */
class MapActivity : AppCompatActivity() {

    private lateinit var mapView: MapView
    private lateinit var statusText: TextView
    private var unsubscribe: (() -> Unit)? = null

    /** Coordonnees de repli si aucune localisation n'est encore disponible (centre du campus, cf. service locator). */
    private val fallbackCenter = GeoPoint(-18.91459, 47.55227)

    override fun onCreate(savedInstanceState: Bundle?) {
        // osmdroid necessite une configuration (user-agent + repertoire de cache) avant tout MapView
        Configuration.getInstance().load(applicationContext, PreferenceManager.getDefaultSharedPreferences(applicationContext))
        Configuration.getInstance().userAgentValue = packageName

        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_map)

        mapView = findViewById(R.id.mapView)
        statusText = findViewById(R.id.mapStatusText)

        mapView.setTileSource(TileSourceFactory.MAPNIK)
        mapView.setMultiTouchControls(true)
        mapView.controller.setZoom(18.0)
        mapView.controller.setCenter(fallbackCenter)

        findViewById<Button>(R.id.saveMapButton).setOnClickListener { onSaveMapClicked() }
        findViewById<Button>(R.id.exportCsvButton).setOnClickListener { onExportCsvClicked() }
        findViewById<Button>(R.id.exportTxtButton).setOnClickListener { onExportTxtClicked() }

        refreshMarkers()
        unsubscribe = LocalizationHistoryRepository.observe { runOnUiThread { refreshMarkers() } }
    }

    override fun onResume() {
        super.onResume()
        mapView.onResume()
    }

    override fun onPause() {
        super.onPause()
        mapView.onPause()
    }

    override fun onDestroy() {
        unsubscribe?.invoke()
        super.onDestroy()
    }

    /** Redessine tous les marqueurs a partir de l'historique complet accumule (Niveaux 1/2/3). */
    private fun refreshMarkers() {
        val records = LocalizationHistoryRepository.all()
        mapView.overlays.clear()

        var lastPoint: GeoPoint? = null
        var count1 = 0; var count2 = 0; var count3 = 0

        for (record in records) {
            record.level1?.let { addMarker(it.latitude, it.longitude, 1, record); count1++; lastPoint = GeoPoint(it.latitude, it.longitude) }
            record.level2?.let { addMarker(it.latitude, it.longitude, 2, record); count2++; lastPoint = GeoPoint(it.latitude, it.longitude) }
            record.level3?.let { addMarker(it.latitude, it.longitude, 3, record); count3++; lastPoint = GeoPoint(it.latitude, it.longitude) }
        }

        mapView.invalidate()
        statusText.text = getString(R.string.map_status_format, count1, count2, count3)

        // Centre la carte sur la derniere position connue au premier rafraichissement utile
        if (lastPoint != null && records.size <= 1) {
            mapView.controller.animateTo(lastPoint)
        }
    }

    private fun addMarker(lat: Double, lon: Double, level: Int, record: LocalizationRecord) {
        val marker = Marker(mapView)
        marker.position = GeoPoint(lat, lon)
        marker.icon = CircleMarkerFactory.forLevel(this, level)
        marker.setAnchor(Marker.ANCHOR_CENTER, Marker.ANCHOR_CENTER)
        marker.title = getString(R.string.map_marker_title_format, level)
        marker.snippet = getString(R.string.map_marker_snippet_format, lat, lon, java.util.Date(record.timestampMs).toString())
        mapView.overlays.add(marker)
    }

    private fun onSaveMapClicked() {
        val path = MapSnapshotExporter.saveViewAsImage(this, mapView)
        val message = if (path != null) getString(R.string.map_saved_format, path) else getString(R.string.map_save_failed)
        Toast.makeText(this, message, Toast.LENGTH_LONG).show()
    }

    private fun onExportCsvClicked() {
        val records = LocalizationHistoryRepository.all()
        if (records.isEmpty()) {
            Toast.makeText(this, getString(R.string.csv_export_empty), Toast.LENGTH_SHORT).show()
            return
        }
        val path = CsvExporter.exportToDownloads(this, records)
        val message = if (path != null) getString(R.string.csv_exported_format, path) else getString(R.string.csv_export_failed)
        Toast.makeText(this, message, Toast.LENGTH_LONG).show()
    }

    /**
     * Exporte le journal detaille (Niveau 1/2/3) accumule a chaque clic sur "Localiser ma position"
     * (cf. CascadeLogRepository, alimente par ArNavigationActivity). Le fichier est deja ecrit de
     * facon incrementale dans Downloads apres chaque clic (cf. TxtLogExporter.appendToDownloadsFile) ;
     * ce bouton produit en plus une copie complete, independante, horodatee au moment de l'export.
     */
    private fun onExportTxtClicked() {
        if (CascadeLogRepository.isEmpty()) {
            Toast.makeText(this, getString(R.string.txt_export_empty), Toast.LENGTH_SHORT).show()
            return
        }
        val path = TxtLogExporter.exportToDownloads(this, CascadeLogRepository.fullText())
        val message = if (path != null) getString(R.string.txt_exported_format, path) else getString(R.string.txt_export_failed)
        Toast.makeText(this, message, Toast.LENGTH_LONG).show()
    }
}
