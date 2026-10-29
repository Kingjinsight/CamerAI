package com.kingjinsight.camerai

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.graphics.Bitmap
import org.json.JSONObject
import java.io.Closeable
import java.nio.FloatBuffer

/** GAIC crop scorer (training/model.py) running on ONNX Runtime. Scores are MOS z-scores. */
class CompositionModel(context: Context) : Closeable {
    private val env = OrtEnvironment.getEnvironment()
    private val session: OrtSession = env.createSession(
        context.assets.open("gaic.onnx").use { it.readBytes() },
        OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(4)
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        },
    )
    private val prior = JSONObject(context.assets.open("gaic.json").bufferedReader().use { it.readText() })
        .getJSONArray("size_prior")
        .let { a -> Array(a.length()) { i -> a.getJSONArray(i).let { floatArrayOf(it.getDouble(0).toFloat(), it.getDouble(1).toFloat()) } } }
    private val pixels = IntArray(INPUT_W * INPUT_H)
    private val image = FloatBuffer.allocate(3 * INPUT_W * INPUT_H)

    /** [frame] must be INPUT_W x INPUT_H; [boxes] are in normalized frame coordinates. */
    fun score(frame: Bitmap, boxes: List<Box>): FloatArray {
        frame.getPixels(pixels, 0, INPUT_W, 0, 0, INPUT_W, INPUT_H)
        val plane = INPUT_W * INPUT_H
        for (i in 0 until plane) {
            val p = pixels[i]
            image.put(i, ((p shr 16 and 0xFF) / 255f - MEAN[0]) / STD[0])
            image.put(plane + i, ((p shr 8 and 0xFF) / 255f - MEAN[1]) / STD[1])
            image.put(2 * plane + i, ((p and 0xFF) / 255f - MEAN[2]) / STD[2])
        }
        val coords = FloatBuffer.allocate(boxes.size * 4)
        for (b in boxes) {
            coords.put(b.left * INPUT_W).put(b.top * INPUT_H)
                .put((b.left + b.w) * INPUT_W).put((b.top + b.h) * INPUT_H)
        }
        coords.rewind()
        image.rewind()
        OnnxTensor.createTensor(env, image, longArrayOf(1, 3, INPUT_H.toLong(), INPUT_W.toLong())).use { img ->
            OnnxTensor.createTensor(env, coords, longArrayOf(boxes.size.toLong(), 4)).use { bx ->
                session.run(mapOf("image" to img, "boxes" to bx)).use { out ->
                    return (out[0].value as FloatArray)
                }
            }
        }
    }

    /** Mean score change the model gives a framing of [relativeSize] x the viewfinder (training/calibrate.py). */
    fun sizePrior(relativeSize: Float): Float {
        if (relativeSize <= prior.first()[0]) return prior.first()[1]
        for (i in 1 until prior.size) {
            val (r1, b1) = prior[i]
            if (relativeSize <= r1) {
                val (r0, b0) = prior[i - 1]
                return b0 + (b1 - b0) * (relativeSize - r0) / (r1 - r0)
            }
        }
        return prior.last()[1]
    }

    override fun close() {
        session.close()
    }

    companion object {
        const val INPUT_W = 240
        const val INPUT_H = 320
        private val MEAN = floatArrayOf(0.485f, 0.456f, 0.406f)
        private val STD = floatArrayOf(0.229f, 0.224f, 0.225f)
    }
}
