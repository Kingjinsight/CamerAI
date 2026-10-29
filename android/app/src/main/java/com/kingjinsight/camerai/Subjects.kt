package com.kingjinsight.camerai

import android.graphics.Bitmap
import com.google.android.gms.tasks.Tasks
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.face.FaceDetection
import com.google.mlkit.vision.face.FaceDetectorOptions
import java.io.Closeable
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min

/** Finds people's heads in analysis frames with ML Kit's bundled face detector. */
class SubjectDetector : Closeable {
    private val detector = FaceDetection.getClient(
        FaceDetectorOptions.Builder()
            .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_FAST)
            .setMinFaceSize(0.08f)
            .build()
    )

    /** Head boxes in normalized coordinates of [frame]; call off the main thread. */
    fun heads(frame: Bitmap): List<Box> =
        Tasks.await(detector.process(InputImage.fromBitmap(frame, 0))).map { face ->
            val r = face.boundingBox
            // Face box spans brows to chin; grow it to cover hair and ears.
            val w = r.width() * 1.4f / frame.width
            val h = r.height() * 1.6f / frame.height
            Box(r.exactCenterX() / frame.width, (r.exactCenterY() - 0.15f * r.height()) / frame.height, w, h)
        }

    override fun close() = detector.close()
}

/** Photo rules for people that the crop model does not know, as score penalties (MOS std). */
object SubjectRules {
    private const val CUT = 1.5f
    private const val DROP = 1.0f
    private const val PLACEMENT = 1.5f
    private const val EDGE = 0.5f

    /** Penalty for framing [box] given [heads], all in camera-frame coordinates; [view] is the current viewfinder. */
    fun penalty(box: Box, heads: List<Box>, view: Box): Float {
        var p = 0f
        var main: Box? = null
        for (head in heads) {
            val inside = coverage(head, box)
            when {
                inside in 0.02f..0.98f -> p += CUT
                inside < 0.02f && coverage(head, view) > 0.98f -> p += DROP
                inside > 0.98f && (main == null || head.w > main.w) -> main = head
            }
        }
        main?.let { head ->
            // Eyes sit ~45% down the head box; put them near the upper third line.
            val eyes = (head.top + 0.45f * head.h - box.top) / box.h
            p += PLACEMENT * max(0f, abs(eyes - 1f / 3f) - 0.08f)
            val x = (head.cx - box.left) / box.w
            if (x < 0.15f || x > 0.85f) p += EDGE
        }
        return p
    }

    /** Fraction of [a]'s area inside [b]. */
    private fun coverage(a: Box, b: Box): Float {
        val w = min(a.left + a.w, b.left + b.w) - max(a.left, b.left)
        val h = min(a.top + a.h, b.top + b.h) - max(a.top, b.top)
        return if (w <= 0f || h <= 0f) 0f else w * h / (a.w * a.h)
    }
}
