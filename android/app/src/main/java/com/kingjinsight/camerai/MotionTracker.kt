package com.kingjinsight.camerai

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import kotlin.math.sqrt

/** Device orientation (rotation vector) and hand-shake level (gyroscope). */
class MotionTracker(context: Context, private val onRotation: () -> Unit) : SensorEventListener {
    private val sensors = context.getSystemService(SensorManager::class.java)
    private val matrix = FloatArray(9)

    @Volatile
    var rotation = floatArrayOf(1f, 0f, 0f, 0f, 1f, 0f, 0f, 0f, 1f)
        private set

    @Volatile
    private var angularSpeed = 0f

    val steady get() = angularSpeed < STEADY_RAD_S

    fun start() {
        val rv = sensors.getDefaultSensor(Sensor.TYPE_GAME_ROTATION_VECTOR)
            ?: sensors.getDefaultSensor(Sensor.TYPE_ROTATION_VECTOR)
        rv?.let { sensors.registerListener(this, it, SensorManager.SENSOR_DELAY_GAME) }
        sensors.getDefaultSensor(Sensor.TYPE_GYROSCOPE)
            ?.let { sensors.registerListener(this, it, SensorManager.SENSOR_DELAY_GAME) }
    }

    fun stop() = sensors.unregisterListener(this)

    override fun onSensorChanged(event: SensorEvent) {
        when (event.sensor.type) {
            Sensor.TYPE_GYROSCOPE -> {
                val (x, y, z) = event.values
                angularSpeed = 0.8f * angularSpeed + 0.2f * sqrt(x * x + y * y + z * z)
            }
            else -> {
                SensorManager.getRotationMatrixFromVector(matrix, event.values)
                rotation = matrix.copyOf()
                onRotation()
            }
        }
    }

    override fun onAccuracyChanged(sensor: Sensor, accuracy: Int) = Unit

    private companion object {
        const val STEADY_RAD_S = 0.15f
    }
}
