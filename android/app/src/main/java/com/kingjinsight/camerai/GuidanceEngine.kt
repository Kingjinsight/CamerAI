package com.kingjinsight.camerai

import kotlin.math.abs

/** Model output for one analysis frame, in camera-frame coordinates. */
class Suggestion(
    /** Framing worth guiding toward, or null when the current one is as good as it gets nearby. */
    val best: Box?,
    /** Predicted score gain of the best framing (MOS std), shown for tuning. */
    val gain: Float,
    val cameraZoom: Float,
    val rotation: FloatArray,
    val steady: Boolean,
    /** Detected heads in camera-frame coordinates. */
    val heads: List<Box> = emptyList(),
    /** How much better [best] scores than the locked target, when the target was in frame. */
    val lockedDeficit: Float? = null,
)

enum class Phase { Searching, Good, Guiding, Aligned }

data class Guidance(
    val phase: Phase,
    /** Target framing in view-normalized coordinates. */
    val box: Box? = null,
    val pan: String? = null,
    val zoom: String? = null,
    val targetZoom: Float? = null,
    /** 0..1 while holding an aligned framing before auto capture. */
    val hold: Float = 0f,
)

/**
 * Turns per-frame crop suggestions into a world-anchored target and per-tick guidance (pan
 * direction, zoom, alignment) for the viewfinder overlay. Once locked, the target stays until it
 * is captured, reset, out of view for [LOST_MS], or beaten by a clearly better framing for
 * [RELOCK_FRAMES] frames in a row; a single quiet frame never drops it.
 */
class GuidanceEngine(var lens: Lens = Lens.DEFAULT) {
    private val recent = ArrayDeque<Target>()

    @Volatile
    private var target: Target? = null
    private var disagree = 0
    private var goodStreak = 0
    private var alignedSince = 0L
    private var lostSince = -1L

    fun reset() {
        recent.clear()
        target = null
        disagree = 0
        goodStreak = 0
        lostSince = -1L
    }

    /** The locked target in a camera frame taken at [cameraZoom] with device [rotation]; safe off the main thread. */
    fun lockedFrameBox(cameraZoom: Float, rotation: FloatArray): Box? =
        target?.let { Geometry.project(it, cameraZoom, lens, rotation) }

    fun onSuggestion(s: Suggestion) {
        val best = s.best
        val current = target
        if (current != null) {
            val fresh = best?.let { Geometry.anchor(it, s.cameraZoom, lens, s.rotation) }
            val beaten = fresh != null && (s.lockedDeficit ?: 0f) >= REPLACE_MARGIN && !Geometry.similar(current, fresh, lens)
            disagree = if (beaten) disagree + 1 else 0
            if (disagree >= RELOCK_FRAMES) {
                target = fresh
                disagree = 0
            }
            return
        }
        if (best == null) {
            recent.clear()
            goodStreak++
            return
        }
        goodStreak = 0
        val fresh = Geometry.anchor(best, s.cameraZoom, lens, s.rotation)
        if (recent.isNotEmpty() && !Geometry.similar(recent.last(), fresh, lens)) recent.clear()
        recent.addLast(fresh)
        if (recent.size >= LOCK_FRAMES && s.steady) {
            target = Geometry.average(recent.toList())
            recent.clear()
        }
    }

    fun update(userZoom: Float, minZoom: Float, rotation: FloatArray, steady: Boolean, nowMs: Long): Guidance {
        val t = target ?: return Guidance(if (goodStreak > 0) Phase.Good else Phase.Searching)
        val box = Geometry.project(t, userZoom, lens, rotation)
        val visible = box != null && abs(box.cx - 0.5f) < 1.5f && abs(box.cy - 0.5f) < 1.5f
        if (visible) {
            lostSince = -1L
        } else {
            if (lostSince < 0) lostSince = nowMs
            if (nowMs - lostSince > LOST_MS) {
                reset()
                return Guidance(Phase.Searching)
            }
        }
        if (box == null) return Guidance(Phase.Guiding, pan = "转回拍摄方向")
        val dx = box.cx - 0.5f
        val dy = box.cy - 0.5f
        val panOk = abs(dx) < PAN_TOL && abs(dy) < PAN_TOL
        val zoomOk = box.w in (1f - ZOOM_TOL)..(1f + ZOOM_TOL)
        val canZoomOut = t.zoom >= minZoom * 0.98f

        val aligned = panOk && (zoomOk || (!canZoomOut && userZoom <= minZoom * 1.02f && box.w > 1f))
        if (!aligned || !steady) alignedSince = nowMs
        if (aligned) {
            return Guidance(Phase.Aligned, box, targetZoom = t.zoom,
                hold = ((nowMs - alignedSince) / HOLD_MS.toFloat()).coerceIn(0f, 1f))
        }
        val pan = if (panOk) null else panText(dx, dy)
        val zoom = when {
            zoomOk -> null
            box.w < 1f -> "放大到 %.1f×".format(t.zoom)
            canZoomOut -> "缩小到 %.1f×".format(t.zoom)
            else -> "后退一步"
        }
        return Guidance(Phase.Guiding, box, pan, zoom, t.zoom)
    }

    private fun panText(dx: Float, dy: Float): String {
        val h = if (abs(dx) >= PAN_TOL) (if (dx > 0) "右" else "左") else ""
        val v = if (abs(dy) >= PAN_TOL) (if (dy > 0) "下" else "上") else ""
        return "向$h$v" + "移动"
    }

    companion object {
        const val LOCK_FRAMES = 3
        const val RELOCK_FRAMES = 6

        /** Score (MOS std) by which a new framing must beat the locked one before it can take over. */
        const val REPLACE_MARGIN = 0.5f

        /** A target out of view this long is given up, e.g. after turning to a different scene. */
        const val LOST_MS = 8000L
        const val PAN_TOL = 0.05f
        const val ZOOM_TOL = 0.08f
        const val HOLD_MS = 800L
    }
}
