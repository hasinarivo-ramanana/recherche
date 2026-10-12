package com.thesis.arlocalizer.util

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager

/**
 * Fournit le cap (yaw), le tangage et le roulis de l'appareil a partir du capteur fusionne
 * TYPE_ROTATION_VECTOR (accelerometre + gyroscope + magnetometre), conformement a la formule de
 * reconstruction d'orientation Android du manuscrit (fusion Mahony IMU+magnetometre).
 *
 * Utilise par le scoring multi-criteres du Niveau 2 (MultiCriteriaScorer.scoreAngle/scoreCote), qui a
 * besoin du cap absolu de la camera (independant du repere local arbitraire d'ARCore).
 */
class CompassProvider(private val context: Context) : SensorEventListener {

    private val sensorManager = context.getSystemService(Context.SENSOR_SERVICE) as SensorManager
    private val rotationVectorSensor = sensorManager.getDefaultSensor(Sensor.TYPE_ROTATION_VECTOR)

    @Volatile private var azimuthRad: Double = 0.0
    @Volatile private var pitchRad: Double = 0.0
    @Volatile private var rollRad: Double = 0.0
    @Volatile private var hasReading: Boolean = false

    fun start() {
        rotationVectorSensor?.let {
            sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_UI)
        }
    }

    fun stop() {
        sensorManager.unregisterListener(this)
    }

    /** Cap (yaw) actuel en radians, 0 = Nord, sens horaire — conforme a GeoUtils.azimuth. */
    fun currentHeadingRad(): Double = azimuthRad

    fun currentPitchRad(): Double = pitchRad

    fun currentRollRad(): Double = rollRad

    fun isReady(): Boolean = hasReading

    override fun onSensorChanged(event: SensorEvent?) {
        if (event == null || event.sensor.type != Sensor.TYPE_ROTATION_VECTOR) return

        val rotationMatrix = FloatArray(9)
        SensorManager.getRotationMatrixFromVector(rotationMatrix, event.values)

        val orientation = FloatArray(3)
        SensorManager.getOrientation(rotationMatrix, orientation)

        // getOrientation renvoie [azimuth, pitch, roll] en radians (azimuth: 0 = Nord, sens horaire)
        azimuthRad = orientation[0].toDouble()
        pitchRad = orientation[1].toDouble()
        rollRad = orientation[2].toDouble()
        hasReading = true
    }

    override fun onAccuracyChanged(sensor: android.hardware.Sensor?, accuracy: Int) { /* non utilise */ }
}
