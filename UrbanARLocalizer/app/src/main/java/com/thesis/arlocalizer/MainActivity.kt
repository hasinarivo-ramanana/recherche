package com.thesis.arlocalizer

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.Environment
import android.provider.Settings
import android.util.Log
import android.widget.Button
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import com.thesis.arlocalizer.ar.ArNavigationActivity
import org.opencv.android.OpenCVLoader

/**
 * Ecran d'accueil : verifie les permissions (camera, localisation) et la disponibilite d'ARCore,
 * puis lance ArNavigationActivity qui heberge la scene RA et la cascade de localisation.
 *
 * Le chargement effectif des modeles (APR .tflite, YOLO .tflite, SIG .geojson, nuage .ply,
 * database.db) est realise dans ArNavigationActivity via LocalizerServiceLocator, pour eviter
 * de dupliquer des objets lourds entre activites.
 *
 * Si LocalizerServiceLocator est configure pour lire database.db depuis Download/arlocalizer/
 * (ColmapDatabaseReader.fromLocalDownloadsFolder, cf. README section 10), la permission speciale
 * "Acces a tous les fichiers" doit etre accordee manuellement par l'utilisateur -- ce qui n'est
 * PAS necessaire si vous utilisez plutot fromAssets ou fromUrl (voir la meme section du README).
 */
class MainActivity : AppCompatActivity() {

    private val requiredPermissions = arrayOf(Manifest.permission.CAMERA, Manifest.permission.ACCESS_FINE_LOCATION)

    private val permissionLauncher = registerForActivityResult(
        androidx.activity.result.contract.ActivityResultContracts.RequestMultiplePermissions()
    ) { results ->
        if (results.values.all { it }) {
            checkAllFilesAccessThenStart()
        } else {
            findViewById<TextView>(R.id.statusText).text = getString(R.string.permissions_required)
        }
    }

    private val allFilesAccessLauncher = registerForActivityResult(
        androidx.activity.result.contract.ActivityResultContracts.StartActivityForResult()
    ) {
        // Que l'utilisateur ait accorde ou refuse, on retente le flux normal : si refuse,
        // ColmapDatabaseReader.fromLocalDownloadsFolder loggera une erreur explicite au chargement.
        startNavigation()
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        if (!OpenCVLoader.initLocal()) {
            Log.e("OpenCV", "Unable to load OpenCV via initLocal!")
        } else {
            Log.d("OpenCV", "OpenCV loaded successfully via initLocal")
        }

        findViewById<Button>(R.id.startButton).setOnClickListener {
            if (hasAllPermissions()) checkAllFilesAccessThenStart() else permissionLauncher.launch(requiredPermissions)
        }
    }

    private fun hasAllPermissions(): Boolean =
        requiredPermissions.all { ContextCompat.checkSelfPermission(this, it) == PackageManager.PERMISSION_GRANTED }

    /**
     * Ne s'applique qu'a la strategie Download/arlocalizer/ (ColmapDatabaseReader.fromLocalDownloadsFolder).
     * Si vous utilisez fromAssets ou fromUrl a la place, cette verification est inutile -- vous
     * pouvez supprimer cet appel et retirer MANAGE_EXTERNAL_STORAGE du manifeste.
     */
    private fun checkAllFilesAccessThenStart() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R && !Environment.isExternalStorageManager()) {
            val intent = Intent(Settings.ACTION_MANAGE_APP_ALL_FILES_ACCESS_PERMISSION).apply {
                data = Uri.parse("package:$packageName")
            }
            allFilesAccessLauncher.launch(intent)
        } else {
            startNavigation()
        }
    }

    private fun startNavigation() {
        startActivity(Intent(this, ArNavigationActivity::class.java))
    }
}
