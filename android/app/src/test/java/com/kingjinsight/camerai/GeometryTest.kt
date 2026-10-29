package com.kingjinsight.camerai

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.cos
import kotlin.math.sin

class GeometryTest {
    private val lens = Lens.DEFAULT
    private val identity = floatArrayOf(1f, 0f, 0f, 0f, 1f, 0f, 0f, 0f, 1f)

    /** Device rotated by [angle] about its own Y (up) axis; positive turns the back camera to the right. */
    private fun yawRight(angle: Float): FloatArray {
        val c = cos(angle)
        val s = sin(angle)
        // Columns are device axes in world coordinates: rotating -Z toward +X.
        return floatArrayOf(c, 0f, -s, 0f, 1f, 0f, s, 0f, c)
    }

    @Test
    fun anchorThenProjectMapsFrameToView() {
        val box = Box(0.7f, 0.4f, 0.5f, 0.5f)
        val cameraZoom = 0.6f
        val userZoom = 0.9f
        val t = Geometry.anchor(box, cameraZoom, lens, identity)
        val v = Geometry.project(t, userZoom, lens, identity)!!
        val d = userZoom / cameraZoom
        assertEquals(0.5f + (box.cx - 0.5f) * d, v.cx, 1e-4f)
        assertEquals(0.5f + (box.cy - 0.5f) * d, v.cy, 1e-4f)
        assertEquals(box.w * d, v.w, 1e-4f)
        assertEquals(cameraZoom / box.w, t.zoom, 1e-4f)
    }

    @Test
    fun turningTowardTargetCentersIt() {
        val t = Geometry.anchor(Box(0.8f, 0.5f, 0.5f, 0.5f), 1f, lens, identity)
        val before = Geometry.project(t, 1f, lens, identity)!!
        assertTrue("target starts right of center", before.cx > 0.5f)
        // Tangent offset (0.3 / fx) -> yaw angle that brings it to the center.
        val angle = kotlin.math.atan(0.3f / lens.fx1)
        val after = Geometry.project(t, 1f, lens, yawRight(angle))!!
        assertEquals(0.5f, after.cx, 1e-3f)
        assertEquals(0.5f, after.cy, 1e-3f)
        val over = Geometry.project(t, 1f, lens, yawRight(angle * 2))!!
        assertTrue("overshoot puts target left", over.cx < 0.5f)
    }

    @Test
    fun targetBehindCameraIsNull() {
        val t = Geometry.anchor(Box(0.5f, 0.5f, 0.5f, 0.5f), 1f, lens, identity)
        assertNull(Geometry.project(t, 1f, lens, yawRight(Math.PI.toFloat())))
    }

    @Test
    fun candidatesStayInsideFrameAndKeepCurrentViewFirst() {
        val boxes = Geometry.candidates(1f / 1.5f)
        assertEquals(Box(0.5f, 0.5f, 1f / 1.5f, 1f / 1.5f), boxes[0])
        assertTrue(boxes.size in 50..400)
        for (b in boxes) {
            assertTrue(b.left >= -1e-5f && b.top >= -1e-5f)
            assertTrue(b.left + b.w <= 1f + 1e-5f && b.top + b.h <= 1f + 1e-5f)
        }
    }

    @Test
    fun panCandidatesCenterTheMiddleWindow() {
        val boxes = Geometry.panCandidates(0.7f)
        assertEquals(25, boxes.size)
        assertEquals(Box(0.5f, 0.5f, 0.7f, 0.7f), boxes[12])
        assertEquals(0.35f, boxes[0].cx, 1e-6f)
        assertEquals(0.65f, boxes[24].cy, 1e-6f)
    }

    @Test
    fun engineLocksAfterConsistentSuggestionsAndGuides() {
        val engine = GuidanceEngine(lens)
        val s = Suggestion(Box(0.75f, 0.5f, 0.6f, 0.6f), gain = 1f, cameraZoom = 1f, rotation = identity, steady = true)
        repeat(GuidanceEngine.LOCK_FRAMES) { engine.onSuggestion(s) }
        val g = engine.update(userZoom = 1f, minZoom = 1f, rotation = identity, steady = true, nowMs = 0)
        assertEquals(Phase.Guiding, g.phase)
        assertEquals("向右移动", g.pan)
        assertEquals("放大到 1.7×", g.zoom)
    }

    @Test
    fun engineHoldsWhenAligned() {
        val engine = GuidanceEngine(lens)
        val s = Suggestion(Box(0.5f, 0.5f, 0.98f, 0.98f), gain = 1f, cameraZoom = 1f, rotation = identity, steady = true)
        repeat(GuidanceEngine.LOCK_FRAMES) { engine.onSuggestion(s) }
        assertEquals(0f, engine.update(1f, 1f, identity, true, 0).hold, 1e-6f)
        val g = engine.update(1f, 1f, identity, true, GuidanceEngine.HOLD_MS)
        assertEquals(Phase.Aligned, g.phase)
        assertEquals(1f, g.hold, 1e-6f)
    }
}
