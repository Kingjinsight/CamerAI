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
 * Turns per-frame crop suggestions into a stable, world-anchored target and per-tick
 * guidance (pan direction, zoom, alignment) for the viewfinder overlay.
 */
class GuidanceEngine(var lens: Lens = Lens.DEFAULT) {
    private val recent = ArrayDeque<Target>()
    private var target: Target? = null
    private var disagree = 0
    private var goodStreak = 0
    private var aligned = false
    private var alignedSince = 0L

    fun reset() {
        recent.clear()
        target = null
        disagree = 0
        goodStreak = 0
        aligned = false
    }

    fun onSuggestion(s: Suggestion) {
        val best = s.best
        if (best == null) {
            recent.clear()
            goodStreak++
            if (goodStreak >= 2 && !aligned) target = null
            return
        }
        goodStreak = 0
        val fresh = Geometry.anchor(best, s.cameraZoom, lens, s.rotation)
        val current = target
        if (current == null) {
            if (recent.isNotEmpty() && !Geometry.similar(recent.last(), fresh, lens)) recent.clear()
            recent.addLast(fresh)
            if (recent.size >= LOCK_FRAMES && s.steady) {
                target = Geometry.average(recent.toList())
                recent.clear()
            }
        } else if (Geometry.similar(current, fresh, lens)) {
            disagree = 0
        } else if (++disagree >= RELOCK_FRAMES) {
            target = fresh
            disagree = 0
        }
    }

    fun update(userZoom: Float, minZoom: Float, rotation: FloatArray, steady: Boolean, nowMs: Long): Guidance {
        val t = target
        if (t == null) {
            aligned = false
            return Guidance(if (goodStreak > 0) Phase.Good else Phase.Searching)
        }
        val box = Geometry.project(t, userZoom, lens, rotation)
            ?: return Guidance(Phase.Guiding, pan = "转回拍摄方向")
        val dx = box.cx - 0.5f
        val dy = box.cy - 0.5f
        val panOk = abs(dx) < PAN_TOL && abs(dy) < PAN_TOL
        val zoomOk = box.w in (1f - ZOOM_TOL)..(1f + ZOOM_TOL)
        val canZoomOut = t.zoom >= minZoom * 0.98f

        aligned = panOk && (zoomOk || (!canZoomOut && userZoom <= minZoom * 1.02f && box.w > 1f))
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
        const val RELOCK_FRAMES = 5
        const val PAN_TOL = 0.05f
        const val ZOOM_TOL = 0.08f
        const val HOLD_MS = 800L
    }
}
