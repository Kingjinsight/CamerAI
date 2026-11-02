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
 * current viewfinder. Candidates may zoom in, shift, and (when the camera frame is wider than the
 * viewfinder) zoom out. The locked target, if any, is scored alongside them so the guidance engine
 * can tell whether something clearly better has appeared. Detected people add penalties for
 * framings that cut or drop them (see [SubjectRules]).
 */
class CompositionAnalyzer(
    private val model: CompositionModel,
    private val subjects: SubjectDetector,
    private val zoom: () -> ZoomSnapshot?,
    private val motion: MotionTracker,
    /** Where the locked target sits in a frame taken at this camera zoom and rotation, if one is locked. */
    private val locked: (cameraZoom: Float, rotation: FloatArray) -> Box?,
    private val onResult: (Suggestion, inferenceMs: Long) -> Unit,
) : ImageAnalysis.Analyzer {

    private class Scored(val best: Box?, val gain: Float, val lockedDeficit: Float?)

    /** Scoring settings for a camera frame [context] times wider than the viewfinder. */
    private class Mode(val context: Float, val priorWeight: Float, val minGain: Float)

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
            val lockedBox = locked(z.camera, rotation)
            val result = if (viewSize * MIN_CONTEXT <= 1f) {
                bestFraming(frame, viewSize, heads, lockedBox, WIDE)
            } else {
                bestFraming(frame, viewSize, heads, lockedBox, ONE_X)
                    .takeIf { it.best != null || lockedBox != null }
                    ?: bestPan(frame, viewSize, heads)
            }
            val elapsed = SystemClock.elapsedRealtime() - start
            val actionable = result.best?.takeIf { needsMove(it, viewSize) }
            onResult(Suggestion(actionable, result.gain, z.camera, rotation, steady, heads, result.lockedDeficit), elapsed)
        }
    }

    private fun bestFraming(frame: Bitmap, viewSize: Float, heads: List<Box>, lockedBox: Box?, mode: Mode): Scored {
        val candidates = Geometry.candidates(viewSize)
        val scoreLocked = lockedBox != null && inFrame(lockedBox)
        val boxes = if (scoreLocked) candidates + lockedBox!! else candidates
        val view = boxes[0]
        val scores = model.score(frame, boxes)
        val values = FloatArray(boxes.size) { i ->
            val b = boxes[i]
            if (i == 0) {
                scores[0] - SubjectRules.penalty(view, heads, view)
            } else {
                scores[i] - mode.priorWeight * model.sizePrior(b.w / viewSize, mode.context) -
                    MOVE_PENALTY * Geometry.moveCost(b, viewSize) - SubjectRules.penalty(b, heads, view)
            }
        }
        val best = values.indices.maxBy { values[it] }
        val gain = values[best] - values[0]
        val deficit = if (scoreLocked) values[best] - values.last() else null
        return Scored(if (gain >= mode.minGain) boxes[best] else null, gain, deficit)
    }

    private fun bestPan(frame: Bitmap, viewSize: Float, heads: List<Box>): Scored {
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
        return Scored(if (gain >= PAN_GAIN) targets[best] else null, gain, null)
    }

    /** A framing already within alignment tolerance of the viewfinder gives nothing to do. */
    private fun needsMove(b: Box, viewSize: Float) =
        abs(b.cx - 0.5f) / viewSize >= GuidanceEngine.PAN_TOL || abs(b.cy - 0.5f) / viewSize >= GuidanceEngine.PAN_TOL ||
            abs(b.w / viewSize - 1f) >= GuidanceEngine.ZOOM_TOL

    /** The model can only score framings the camera frame actually contains. */
    private fun inFrame(b: Box) =
        b.left >= -FRAME_SLACK && b.top >= -FRAME_SLACK && b.left + b.w <= 1f + FRAME_SLACK && b.top + b.h <= 1f + FRAME_SLACK

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
        const val FRAME_SLACK = 0.02f

        /**
         * Removed share of the size bias (training/calibrate.py) and minimum gain (MOS std) to guide.
         * With 1.5x context: ~68% of GAICD frames guided. At 1x: ~72% guided, median zoom 1.5x, and
         * the chosen framing stays put after a 3% frame shift 89% of the time.
         */
        val WIDE = Mode(context = 1.5f, priorWeight = 0.5f, minGain = 0.4f)
        val ONE_X = Mode(context = 1.0f, priorWeight = 0.75f, minGain = 0.3f)

        /** Score (MOS std) a framing must beat per view-width of movement, so nearby framings win ties. */
        const val MOVE_PENALTY = 0.5f

        /** Fallback pan mode compares windows of this fraction of the viewfinder at 5x5 positions. */
        const val PAN_PROBE = 0.7f
        const val PAN_MOVE_PENALTY = 0.5f

        /** On GAICD, pan hints at this threshold point back toward the original framing 82% of the time. */
        const val PAN_GAIN = 0.6f
    }
}
