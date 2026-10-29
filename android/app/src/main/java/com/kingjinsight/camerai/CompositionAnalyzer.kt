package com.kingjinsight.camerai

import android.graphics.Bitmap
import android.graphics.Matrix
import android.os.SystemClock
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy

/** Camera zoom the frame was taken at, and the user's viewfinder zoom. */
class ZoomSnapshot(val camera: Float, val user: Float)

/**
 * Scores candidate framings of each analysis frame and reports the best one relative to the
 * current viewfinder. The camera frame is wider than the viewfinder, so candidates may extend
 * beyond it (zoom out) as well as inside it (zoom in / pan).
 */
class CompositionAnalyzer(
    private val model: CompositionModel,
    private val zoom: () -> ZoomSnapshot?,
    private val motion: MotionTracker,
    private val onResult: (Suggestion, inferenceMs: Long) -> Unit,
) : ImageAnalysis.Analyzer {

    override fun analyze(image: ImageProxy) {
        image.use {
            val z = zoom() ?: return
            val rotation = motion.rotation
            val steady = motion.steady
            val frame = upright(image)
            val viewSize = (z.camera / z.user).coerceAtMost(1f)
            val boxes = Geometry.candidates(viewSize)
            val start = SystemClock.elapsedRealtime()
            val scores = model.score(frame, boxes)
            val elapsed = SystemClock.elapsedRealtime() - start

            var best = 0
            var bestValue = scores[0]
            for (i in 1 until boxes.size) {
                val v = scores[i] - SIZE_PRIOR_WEIGHT * model.sizePrior(boxes[i].w / viewSize) -
                    MOVE_PENALTY * Geometry.moveCost(boxes[i], viewSize)
                if (v > bestValue) {
                    best = i
                    bestValue = v
                }
            }
            onResult(Suggestion(boxes[best], bestValue - scores[0], z.camera, rotation, steady), elapsed)
        }
    }

    private fun upright(image: ImageProxy): Bitmap {
        val src = image.toBitmap()
        val rotated = Bitmap.createBitmap(src, 0, 0, src.width, src.height,
            Matrix().apply { postRotate(image.imageInfo.rotationDegrees.toFloat()) }, false)
        return Bitmap.createScaledBitmap(rotated, CompositionModel.INPUT_W, CompositionModel.INPUT_H, true)
    }

    private companion object {
        /** Share of the model's bias toward larger framings that is removed (see training/calibrate.py). */
        const val SIZE_PRIOR_WEIGHT = 0.5f

        /** Score (MOS std) a framing must beat per view-width of movement, so nearby framings win ties. */
        const val MOVE_PENALTY = 0.5f
    }
}
