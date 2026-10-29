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
 * Detected people add penalties for framings that cut or drop them (see [SubjectRules]).
 */
class CompositionAnalyzer(
    private val model: CompositionModel,
    private val subjects: SubjectDetector,
    private val zoom: () -> ZoomSnapshot?,
    private val motion: MotionTracker,
    private val onResult: (Suggestion, inferenceMs: Long) -> Unit,
) : ImageAnalysis.Analyzer {

    override fun analyze(image: ImageProxy) {
        image.use {
            val z = zoom() ?: return
            val rotation = motion.rotation
            val steady = motion.steady
            val upright = upright(image)
            val viewSize = (z.camera / z.user).coerceAtMost(1f)
            val start = SystemClock.elapsedRealtime()
            val heads = subjects.heads(Bitmap.createScaledBitmap(upright, DETECT_W, DETECT_H, true))
            val frame = Bitmap.createScaledBitmap(upright, CompositionModel.INPUT_W, CompositionModel.INPUT_H, true)
            val (best, gain) =
                if (viewSize * MIN_CONTEXT > 1f) bestPan(frame, viewSize, heads) else bestFraming(frame, viewSize, heads)
            val elapsed = SystemClock.elapsedRealtime() - start
            val actionable = best?.takeIf { needsMove(it, viewSize) }
            onResult(Suggestion(actionable, gain, z.camera, rotation, steady, heads), elapsed)
        }
    }

    private fun bestFraming(frame: Bitmap, viewSize: Float, heads: List<Box>): Pair<Box?, Float> {
        val boxes = Geometry.candidates(viewSize)
        val view = boxes[0]
        val scores = model.score(frame, boxes)
        val current = scores[0] - SubjectRules.penalty(view, heads, view)
        var best = 0
        var bestValue = current
        for (i in 1 until boxes.size) {
            val v = scores[i] - SIZE_PRIOR_WEIGHT * model.sizePrior(boxes[i].w / viewSize) -
                MOVE_PENALTY * Geometry.moveCost(boxes[i], viewSize) - SubjectRules.penalty(boxes[i], heads, view)
            if (v > bestValue) {
                best = i
                bestValue = v
            }
        }
        val gain = bestValue - current
        return (if (gain >= FRAMING_GAIN) boxes[best] else null) to gain
    }

    private fun bestPan(frame: Bitmap, viewSize: Float, heads: List<Box>): Pair<Box?, Float> {
        val boxes = Geometry.panCandidates(PAN_PROBE * viewSize)
        val scores = model.score(frame, boxes)
        // Each probe stands for shifting the whole viewfinder so its center lands on the probe's center.
        val targets = boxes.map { Box(it.cx, it.cy, viewSize, viewSize) }
        val center = boxes.size / 2
        val view = targets[center]
        val current = scores[center] - SubjectRules.penalty(view, heads, view)
        var best = center
        var bestValue = current
        for (i in boxes.indices) {
            val v = scores[i] - PAN_MOVE_PENALTY * hypot(boxes[i].cx - 0.5f, boxes[i].cy - 0.5f) / viewSize -
                SubjectRules.penalty(targets[i], heads, view)
            if (v > bestValue) {
                best = i
                bestValue = v
            }
        }
        val gain = bestValue - current
        return (if (gain >= PAN_GAIN) targets[best] else null) to gain
    }

    /** A framing already within alignment tolerance of the viewfinder gives nothing to do. */
    private fun needsMove(b: Box, viewSize: Float) =
        abs(b.cx - 0.5f) / viewSize >= GuidanceEngine.PAN_TOL || abs(b.cy - 0.5f) / viewSize >= GuidanceEngine.PAN_TOL ||
            abs(b.w / viewSize - 1f) >= GuidanceEngine.ZOOM_TOL

    private fun upright(image: ImageProxy): Bitmap {
        val src = image.toBitmap()
        return Bitmap.createBitmap(src, 0, 0, src.width, src.height,
            Matrix().apply { postRotate(image.imageInfo.rotationDegrees.toFloat()) }, false)
    }

    private companion object {
        /** Face detection input; faces of 8% frame width stay ~30 px, enough for ML Kit. */
        const val DETECT_W = 360
        const val DETECT_H = 480

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
