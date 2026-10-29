package com.kingjinsight.camerai

import android.graphics.Bitmap
import android.graphics.Matrix
import android.os.SystemClock
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import kotlin.math.abs
import kotlin.math.hypot

/** Camera zoom the frame was taken at, and the user's viewfinder zoom. */
class ZoomSnapshot(val camera: Float, val user: Float)

/**
 * Scores candidate framings of each analysis frame and reports the best one relative to the
 * current viewfinder. When the camera frame is wider than the viewfinder, candidates may extend
 * beyond it (zoom out) as well as inside it (zoom in / pan). When it is not, only equal-size
 * framings are compared, which turns the model's position preference into a pan hint.
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
            val start = SystemClock.elapsedRealtime()
            val (best, gain) = if (viewSize * MIN_CONTEXT > 1f) bestPan(frame, viewSize) else bestFraming(frame, viewSize)
            val elapsed = SystemClock.elapsedRealtime() - start
            val actionable = best?.takeIf { needsMove(it, viewSize) }
            onResult(Suggestion(actionable, gain, z.camera, rotation, steady), elapsed)
        }
    }

    private fun bestFraming(frame: Bitmap, viewSize: Float): Pair<Box?, Float> {
        val boxes = Geometry.candidates(viewSize)
        val scores = model.score(frame, boxes)
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
        val gain = bestValue - scores[0]
        return (if (gain >= FRAMING_GAIN) boxes[best] else null) to gain
    }

    private fun bestPan(frame: Bitmap, viewSize: Float): Pair<Box?, Float> {
        val boxes = Geometry.panCandidates(PAN_PROBE * viewSize)
        val scores = model.score(frame, boxes)
        val center = boxes.size / 2
        var best = center
        var bestValue = scores[center]
        for (i in boxes.indices) {
            val v = scores[i] - PAN_MOVE_PENALTY * hypot(boxes[i].cx - 0.5f, boxes[i].cy - 0.5f) / viewSize
            if (v > bestValue) {
                best = i
                bestValue = v
            }
        }
        val gain = bestValue - scores[center]
        // Shift the whole viewfinder so its center lands on the better-placed window's center.
        val target = boxes[best].let { Box(it.cx, it.cy, viewSize, viewSize) }
        return (if (gain >= PAN_GAIN) target else null) to gain
    }

    /** A framing already within alignment tolerance of the viewfinder gives nothing to do. */
    private fun needsMove(b: Box, viewSize: Float) =
        abs(b.cx - 0.5f) / viewSize >= GuidanceEngine.PAN_TOL || abs(b.cy - 0.5f) / viewSize >= GuidanceEngine.PAN_TOL ||
            abs(b.w / viewSize - 1f) >= GuidanceEngine.ZOOM_TOL

    private fun upright(image: ImageProxy): Bitmap {
        val src = image.toBitmap()
        val rotated = Bitmap.createBitmap(src, 0, 0, src.width, src.height,
            Matrix().apply { postRotate(image.imageInfo.rotationDegrees.toFloat()) }, false)
        return Bitmap.createScaledBitmap(rotated, CompositionModel.INPUT_W, CompositionModel.INPUT_H, true)
    }

    private companion object {
        /** Camera frame must be this much wider than the viewfinder to suggest zooming out. */
        const val MIN_CONTEXT = 1.25f

        /** Share of the model's bias toward larger framings that is removed (see training/calibrate.py). */
        const val SIZE_PRIOR_WEIGHT = 0.5f

        /** Score (MOS std) a framing must beat per view-width of movement, so nearby framings win ties. */
        const val MOVE_PENALTY = 0.5f

        /** Minimum predicted gain (MOS std) worth guiding toward; ~45% of GAICD frames get guidance. */
        const val FRAMING_GAIN = 0.8f

        /** Pan mode compares windows of this fraction of the viewfinder at 5x5 positions. */
        const val PAN_PROBE = 0.7f
        const val PAN_MOVE_PENALTY = 0.5f

        /** On GAICD, pan hints at this threshold point back toward the original framing 82% of the time. */
        const val PAN_GAIN = 0.6f
    }
}
