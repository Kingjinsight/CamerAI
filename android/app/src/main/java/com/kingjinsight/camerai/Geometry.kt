package com.kingjinsight.camerai

import kotlin.math.abs
import kotlin.math.ceil
import kotlin.math.exp
import kotlin.math.hypot
import kotlin.math.ln
import kotlin.math.sqrt

/** Normalized focal lengths (focal / frame side) of the upright portrait frame at 1x zoom. */
data class Lens(val fx1: Float, val fy1: Float) {
    companion object {
        /** ~24 mm equivalent main camera, 3:4 portrait frame. */
        val DEFAULT = Lens(1f, 0.75f)
    }
}

/** Axis-aligned box in normalized coordinates, by center and size. */
data class Box(val cx: Float, val cy: Float, val w: Float, val h: Float) {
    val left get() = cx - w / 2
    val top get() = cy - h / 2
}

/**
 * A framing anchored in the world: unit direction of its center in world coordinates and
 * the user zoom at which the viewfinder exactly covers it.
 */
class Target(val dir: FloatArray, val zoom: Float)

object Geometry {

    /**
     * Anchors [box] (camera-frame coordinates, same aspect as the frame) seen at [cameraZoom]
     * while the device had rotation matrix [rot] (device -> world, row-major).
     */
    fun anchor(box: Box, cameraZoom: Float, lens: Lens, rot: FloatArray): Target {
        val tx = (box.cx - 0.5f) / (lens.fx1 * cameraZoom)
        val ty = (box.cy - 0.5f) / (lens.fy1 * cameraZoom)
        // Back camera in portrait: image x = device +X, image y = device -Y, optical axis = device -Z.
        val world = mul(rot, floatArrayOf(tx, -ty, -1f))
        return Target(normalize(world), cameraZoom / box.w)
    }

    /** Where [target] sits in the viewfinder (view-normalized coordinates); null when behind the camera. */
    fun project(target: Target, userZoom: Float, lens: Lens, rot: FloatArray): Box? {
        val dev = mulTransposed(rot, target.dir)
        val depth = -dev[2]
        if (depth < 1e-3f) return null
        val tx = dev[0] / depth
        val ty = -dev[1] / depth
        val size = userZoom / target.zoom
        return Box(0.5f + tx * lens.fx1 * userZoom, 0.5f + ty * lens.fy1 * userZoom, size, size)
    }

    /** True when two targets describe nearly the same framing (one candidate-grid step apart at most). */
    fun similar(a: Target, b: Target, lens: Lens): Boolean {
        val dot = (a.dir[0] * b.dir[0] + a.dir[1] * b.dir[1] + a.dir[2] * b.dir[2]).coerceIn(-1f, 1f)
        val angle = sqrt(2f * (1f - dot))
        val width = 1f / (lens.fx1 * minOf(a.zoom, b.zoom))
        return angle < 0.1f * width && abs(ln(a.zoom / b.zoom)) < 0.2f
    }

    fun average(targets: List<Target>): Target {
        val sum = FloatArray(3)
        var logZoom = 0f
        for (t in targets) {
            for (i in 0..2) sum[i] += t.dir[i]
            logZoom += ln(t.zoom)
        }
        return Target(normalize(sum), exp(logZoom / targets.size))
    }

    /**
     * Candidate framings in camera-frame coordinates, all with the frame's aspect ratio.
     * Index 0 is the current viewfinder, a centered box of side [viewSize].
     */
    fun candidates(viewSize: Float, scales: Int = 7, maxPositions: Int = 6): List<Box> {
        val out = mutableListOf(Box(0.5f, 0.5f, viewSize, viewSize))
        val lo = maxOf(0.42f, viewSize * 0.6f)
        for (i in 0 until scales) {
            val s = lo * Math.pow((1.0 / lo), i / (scales - 1.0)).toFloat()
            val n = ceil((1f - s) / 0.05f).toInt().plus(1).coerceIn(1, maxPositions)
            for (yi in 0 until n) for (xi in 0 until n) {
                val step = if (n == 1) 0f else (1f - s) / (n - 1)
                val x = if (n == 1) (1f - s) / 2 else xi * step
                val y = if (n == 1) (1f - s) / 2 else yi * step
                out += Box(x + s / 2, y + s / 2, s, s)
            }
        }
        return out
    }

    /** Equal-size windows of side [size] on an n x n grid over the frame; the middle one is centered. */
    fun panCandidates(size: Float, n: Int = 5): List<Box> {
        val step = (1f - size) / (n - 1)
        return List(n * n) { i -> Box(size / 2 + (i % n) * step, size / 2 + (i / n) * step, size, size) }
    }

    /** Distance of [box] from the current view, in view widths plus log zoom change. */
    fun moveCost(box: Box, viewSize: Float): Float =
        hypot(box.cx - 0.5f, box.cy - 0.5f) / viewSize + abs(ln(box.w / viewSize))

    private fun mul(m: FloatArray, v: FloatArray) = floatArrayOf(
        m[0] * v[0] + m[1] * v[1] + m[2] * v[2],
        m[3] * v[0] + m[4] * v[1] + m[5] * v[2],
        m[6] * v[0] + m[7] * v[1] + m[8] * v[2],
    )

    private fun mulTransposed(m: FloatArray, v: FloatArray) = floatArrayOf(
        m[0] * v[0] + m[3] * v[1] + m[6] * v[2],
        m[1] * v[0] + m[4] * v[1] + m[7] * v[2],
        m[2] * v[0] + m[5] * v[1] + m[8] * v[2],
    )

    private fun normalize(v: FloatArray): FloatArray {
        val n = sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
        return floatArrayOf(v[0] / n, v[1] / n, v[2] / n)
    }
}
