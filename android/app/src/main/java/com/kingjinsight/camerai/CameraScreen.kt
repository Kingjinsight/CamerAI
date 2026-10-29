package com.kingjinsight.camerai

import android.content.ContentValues
import android.content.Context
import android.content.Intent
import android.graphics.Bitmap
import android.hardware.camera2.CameraCharacteristics
import android.net.Uri
import android.os.SystemClock
import android.provider.MediaStore
import android.util.Size
import android.widget.Toast
import androidx.annotation.OptIn
import androidx.camera.camera2.interop.Camera2CameraInfo
import androidx.camera.camera2.interop.ExperimentalCamera2Interop
import androidx.camera.core.Camera
import androidx.camera.core.CameraInfo
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.Preview
import androidx.camera.core.resolutionselector.AspectRatioStrategy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.animation.core.Animatable
import androidx.compose.animation.core.tween
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.gestures.detectTransformGestures
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.systemBarsPadding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableLongStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.clipToBounds
import androidx.compose.ui.geometry.CornerRadius
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size as GSize
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.PathEffect
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.drawscope.DrawScope
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.concurrent.futures.await
import androidx.core.content.ContextCompat
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.LifecycleEventObserver
import androidx.lifecycle.compose.LocalLifecycleOwner
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.suspendCancellableCoroutine
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.concurrent.Executors
import kotlin.coroutines.resume
import kotlin.coroutines.resumeWithException
import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.hypot
import kotlin.math.min
import kotlin.math.sin

/** Camera frame width relative to the viewfinder, so the model sees what lies outside it. */
private const val CONTEXT = 1.5f
private const val ZOOM_SETTLE_MS = 600L

private val Amber = Color(0xFFFFC83D)
private val Green = Color(0xFF4CD964)

@Composable
fun CameraScreen() {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    val scope = rememberCoroutineScope()

    val previewView = remember {
        PreviewView(context).apply {
            implementationMode = PreviewView.ImplementationMode.COMPATIBLE
            scaleType = PreviewView.ScaleType.FIT_CENTER
        }
    }
    val imageCapture = remember {
        ImageCapture.Builder()
            .setResolutionSelector(ratio4x3().build())
            .setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)
            .build()
    }
    val engine = remember { GuidanceEngine() }
    var tick by remember { mutableLongStateOf(0L) }
    val motion = remember { MotionTracker(context) { tick = SystemClock.elapsedRealtime() } }

    var camera by remember { mutableStateOf<Camera?>(null) }
    var minZoom by remember { mutableFloatStateOf(1f) }
    var maxZoom by remember { mutableFloatStateOf(1f) }
    var userZoom by remember { mutableFloatStateOf(1f) }
    var cameraZoom by remember { mutableFloatStateOf(1f) }
    var zoomChangedAt by remember { mutableLongStateOf(0L) }
    var inferenceMs by remember { mutableLongStateOf(0L) }
    var gain by remember { mutableFloatStateOf(0f) }
    var heads by remember { mutableStateOf<List<Box>>(emptyList()) }
    var autoCapture by remember { mutableStateOf(true) }
    var capturing by remember { mutableStateOf(false) }
    var frozen by remember { mutableStateOf<Bitmap?>(null) }
    var lastPhoto by remember { mutableStateOf<Uri?>(null) }
    val zoomAnim = remember { Animatable(1f) }

    fun setZoom(requested: Float) {
        val cam = camera ?: return
        userZoom = requested.coerceIn(minZoom, maxZoom)
        val digital = userZoom / cameraZoom
        val atWidest = cameraZoom <= minZoom * 1.001f
        // Keep the camera ~CONTEXT wider than the viewfinder; only retune outside a band to avoid lens hopping.
        if (!(digital in 1.25f..1.8f || (atWidest && digital in 1f..1.8f))) {
            cameraZoom = (userZoom / CONTEXT).coerceIn(minZoom, maxZoom)
            cam.cameraControl.setZoomRatio(cameraZoom)
            zoomChangedAt = SystemClock.elapsedRealtime()
        }
    }

    fun capture() {
        val cam = camera ?: return
        if (capturing) return
        capturing = true
        frozen = previewView.bitmap
        scope.launch {
            try {
                // Shoot through the lens that natively matches the viewfinder instead of cropping the wide frame.
                if (abs(userZoom - cameraZoom) > 0.01f) {
                    cam.cameraControl.setZoomRatio(userZoom).await()
                    delay(400)
                }
                lastPhoto = takePicture(context, imageCapture)
                Toast.makeText(context, "已保存到 相册/CamerAI", Toast.LENGTH_SHORT).show()
            } catch (e: Exception) {
                Toast.makeText(context, "拍摄失败：${e.message}", Toast.LENGTH_LONG).show()
            } finally {
                runCatching { cam.cameraControl.setZoomRatio(cameraZoom).await() }
                zoomChangedAt = SystemClock.elapsedRealtime()
                delay(300)
                frozen = null
                capturing = false
                engine.reset()
            }
        }
    }

    DisposableEffect(lifecycleOwner) {
        val model = CompositionModel(context)
        val subjects = SubjectDetector()
        val executor = Executors.newSingleThreadExecutor()
        val analysis = ImageAnalysis.Builder()
            .setResolutionSelector(
                ratio4x3().setResolutionStrategy(
                    ResolutionStrategy(Size(640, 480), ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER)
                ).build()
            )
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
            .build()
        val main = ContextCompat.getMainExecutor(context)
        analysis.setAnalyzer(executor, CompositionAnalyzer(
            model,
            subjects,
            zoom = {
                val settled = SystemClock.elapsedRealtime() - zoomChangedAt > ZOOM_SETTLE_MS
                if (settled && !capturing && camera != null) ZoomSnapshot(cameraZoom, userZoom) else null
            },
            motion = motion,
        ) { suggestion, ms ->
            main.execute {
                engine.onSuggestion(suggestion)
                inferenceMs = ms
                gain = suggestion.gain
                // Frame -> viewfinder coordinates: the viewfinder is a centered digital crop of the frame.
                val d = userZoom / suggestion.cameraZoom
                heads = suggestion.heads.map { Box(0.5f + (it.cx - 0.5f) * d, 0.5f + (it.cy - 0.5f) * d, it.w * d, it.h * d) }
                tick = SystemClock.elapsedRealtime()
            }
        })

        val providerFuture = ProcessCameraProvider.getInstance(context)
        providerFuture.addListener({
            val provider = providerFuture.get()
            val preview = Preview.Builder().setResolutionSelector(ratio4x3().build()).build()
            preview.surfaceProvider = previewView.surfaceProvider
            provider.unbindAll()
            val cam = provider.bindToLifecycle(lifecycleOwner, widestBackCamera(provider), preview, analysis, imageCapture)
            camera = cam
            engine.lens = lensOf(cam.cameraInfo)
            cameraZoom = 1f
            cam.cameraInfo.zoomState.observe(lifecycleOwner) {
                if (it.minZoomRatio != minZoom || it.maxZoomRatio != maxZoom) {
                    minZoom = it.minZoomRatio
                    maxZoom = it.maxZoomRatio
                    setZoom(userZoom)
                }
            }
            setZoom(1f)
        }, main)

        val observer = LifecycleEventObserver { _, event ->
            when (event) {
                Lifecycle.Event.ON_RESUME -> motion.start()
                Lifecycle.Event.ON_PAUSE -> motion.stop()
                else -> Unit
            }
        }
        lifecycleOwner.lifecycle.addObserver(observer)
        onDispose {
            lifecycleOwner.lifecycle.removeObserver(observer)
            motion.stop()
            analysis.clearAnalyzer()
            executor.execute {
                model.close()
                subjects.close()
            }
            executor.shutdown()
            runCatching { providerFuture.get().unbindAll() }
        }
    }

    // tick is the timestamp of the latest sensor/analysis update and drives recomposition.
    val guidance = engine.update(userZoom, minZoom, motion.rotation, motion.steady, tick)

    LaunchedEffect(guidance.hold >= 1f) {
        if (guidance.hold >= 1f && autoCapture) capture()
    }

    Column(
        Modifier
            .fillMaxSize()
            .background(Color.Black)
            .systemBarsPadding()
    ) {
        Box(
            Modifier
                .fillMaxWidth()
                .aspectRatio(3f / 4f)
                .clipToBounds()
                .pointerInput(Unit) {
                    detectTransformGestures { _, _, zoom, _ -> if (!capturing) setZoom(userZoom * zoom) }
                }
        ) {
            val scale = userZoom / cameraZoom
            Box(Modifier.fillMaxSize().graphicsLayer { scaleX = scale; scaleY = scale }) {
                AndroidView({ previewView }, Modifier.fillMaxSize())
                frozen?.let {
                    Image(it.asImageBitmap(), null, Modifier.fillMaxSize(), contentScale = ContentScale.FillBounds)
                }
            }
            Canvas(Modifier.fillMaxSize()) { drawOverlay(guidance, heads) }
            HintChip(guidance, autoCapture, Modifier.align(Alignment.TopCenter).padding(top = 12.dp))
            Text(
                "镜头 %.1f–%.0f× · 取景 %.2f× · 相机 %.2f× · %dms · Δ%.2f".format(minZoom, maxZoom, userZoom, cameraZoom, inferenceMs, gain),
                color = Color.White.copy(alpha = 0.6f), fontSize = 10.sp,
                modifier = Modifier.align(Alignment.BottomStart).padding(6.dp)
            )
        }

        Spacer(Modifier.height(12.dp))
        ZoomRow(minZoom, maxZoom, userZoom, guidance.targetZoom) { target ->
            scope.launch {
                zoomAnim.snapTo(userZoom)
                zoomAnim.animateTo(target.coerceIn(minZoom, maxZoom), tween(450)) { setZoom(value) }
            }
        }
        Spacer(Modifier.weight(1f))
        Row(
            Modifier.fillMaxWidth().padding(horizontal = 28.dp, vertical = 20.dp),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Thumbnail(lastPhoto, context)
            Shutter(if (guidance.phase == Phase.Aligned && autoCapture) guidance.hold else 0f) { capture() }
            Text(
                if (autoCapture) "自动拍\n开" else "自动拍\n关",
                color = if (autoCapture) Amber else Color.White,
                fontSize = 13.sp, lineHeight = 16.sp,
                modifier = Modifier
                    .width(56.dp)
                    .clip(RoundedCornerShape(10.dp))
                    .clickable { autoCapture = !autoCapture }
                    .padding(8.dp),
            )
        }
    }
}

private fun DrawScope.drawOverlay(g: Guidance, heads: List<Box>) {
    val w = size.width
    val h = size.height
    val faint = Color.White.copy(alpha = 0.25f)
    for (i in 1..2) {
        drawLine(faint, Offset(w * i / 3, 0f), Offset(w * i / 3, h), 1.dp.toPx())
        drawLine(faint, Offset(0f, h * i / 3), Offset(w, h * i / 3), 1.dp.toPx())
    }
    for (head in heads) {
        drawRoundRect(
            Color.White.copy(alpha = 0.7f),
            topLeft = Offset(head.left * w, head.top * h),
            size = GSize(head.w * w, head.h * h),
            cornerRadius = CornerRadius(8.dp.toPx()),
            style = Stroke(1.5.dp.toPx()),
        )
    }
    val center = Offset(w / 2, h / 2)
    val box = g.box
    val color = if (g.phase == Phase.Aligned) Green else Amber
    drawCircle(Color.White, 16.dp.toPx(), center, style = Stroke(2.dp.toPx()))
    if (box == null || g.phase == Phase.Good || g.phase == Phase.Searching) return

    drawRect(
        color,
        topLeft = Offset(box.left * w, box.top * h),
        size = GSize(box.w * w, box.h * h),
        style = Stroke(3.dp.toPx()),
    )
    val dot = Offset(box.cx * w, box.cy * h)
    val inside = dot.x in 0f..w && dot.y in 0f..h
    if (inside) {
        if (g.pan != null) {
            drawLine(color, center, dot, 2.dp.toPx(), pathEffect = PathEffect.dashPathEffect(floatArrayOf(14f, 10f)))
        }
        drawCircle(color, 7.dp.toPx(), dot)
    } else {
        // Target center left the viewfinder: arrow on the border pointing at it.
        val angle = atan2(dot.y - center.y, dot.x - center.x)
        val inset = 36.dp.toPx()
        val reach = min(
            if (cos(angle) != 0f) (w / 2 - inset) / abs(cos(angle)) else Float.MAX_VALUE,
            if (sin(angle) != 0f) (h / 2 - inset) / abs(sin(angle)) else Float.MAX_VALUE,
        )
        val tip = Offset(center.x + cos(angle) * reach, center.y + sin(angle) * reach)
        val back = 26.dp.toPx()
        val side = 14.dp.toPx()
        val path = Path().apply {
            moveTo(tip.x, tip.y)
            lineTo(tip.x - back * cos(angle) - side * sin(angle), tip.y - back * sin(angle) + side * cos(angle))
            lineTo(tip.x - back * cos(angle) + side * sin(angle), tip.y - back * sin(angle) - side * cos(angle))
            close()
        }
        drawPath(path, color)
    }
}

@Composable
private fun HintChip(g: Guidance, autoCapture: Boolean, modifier: Modifier) {
    val text = when (g.phase) {
        Phase.Searching -> "正在分析构图…"
        Phase.Good -> "取景合适，可以拍摄"
        Phase.Guiding -> listOfNotNull(g.pan, g.zoom).joinToString(" · ")
        Phase.Aligned -> if (autoCapture) "保持稳定，即将拍摄" else "已对齐，按快门"
    }
    val color = when (g.phase) {
        Phase.Aligned, Phase.Good -> Green
        Phase.Guiding -> Amber
        Phase.Searching -> Color.White
    }
    Text(
        text, color = color, fontSize = 16.sp, fontWeight = FontWeight.Medium,
        modifier = modifier
            .background(Color.Black.copy(alpha = 0.55f), RoundedCornerShape(18.dp))
            .padding(horizontal = 16.dp, vertical = 8.dp)
    )
}

@Composable
private fun ZoomRow(minZoom: Float, maxZoom: Float, current: Float, target: Float?, onZoom: (Float) -> Unit) {
    val presets = buildList {
        if (minZoom < 0.95f) add(minZoom)
        addAll(listOf(1f, 2f, 3.7f, 5f).filter { it <= maxZoom })
    }
    Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.Center, verticalAlignment = Alignment.CenterVertically) {
        for (p in presets) {
            val active = abs(current - p) < 0.05f
            Text(
                if (active) "%.1f×".format(current) else "%.1f".format(p),
                color = if (active) Amber else Color.White,
                fontSize = 13.sp,
                modifier = Modifier
                    .padding(horizontal = 6.dp)
                    .size(40.dp)
                    .background(Color.White.copy(alpha = 0.12f), CircleShape)
                    .clip(CircleShape)
                    .clickable { onZoom(p) }
                    .padding(top = 11.dp),
                textAlign = TextAlign.Center,
            )
        }
        val reachable = target?.coerceIn(minZoom, maxZoom)
        if (reachable != null && abs(reachable - current) / current > 0.08f) {
            Text(
                "对准 %.1f×".format(reachable),
                color = Color.Black, fontSize = 13.sp, fontWeight = FontWeight.Medium,
                modifier = Modifier
                    .padding(start = 10.dp)
                    .background(Amber, RoundedCornerShape(20.dp))
                    .clip(RoundedCornerShape(20.dp))
                    .clickable { onZoom(reachable) }
                    .padding(horizontal = 12.dp, vertical = 9.dp),
            )
        }
    }
}

@Composable
private fun Shutter(progress: Float, onClick: () -> Unit) {
    Box(Modifier.size(78.dp).clickable(onClick = onClick), contentAlignment = Alignment.Center) {
        Canvas(Modifier.fillMaxSize()) {
            drawCircle(Color.White, size.minDimension / 2 - 3.dp.toPx(), style = Stroke(4.dp.toPx()))
            drawCircle(Color.White, size.minDimension / 2 - 10.dp.toPx())
            if (progress > 0f) {
                drawArc(Green, -90f, 360f * progress, false, style = Stroke(5.dp.toPx()))
            }
        }
    }
}

@Composable
private fun Thumbnail(uri: Uri?, context: Context) {
    val bitmap = remember(uri) {
        uri?.let { runCatching { context.contentResolver.loadThumbnail(it, Size(160, 160), null) }.getOrNull() }
    }
    Box(
        Modifier
            .size(52.dp)
            .clip(RoundedCornerShape(10.dp))
            .border(1.dp, Color.White.copy(alpha = 0.5f), RoundedCornerShape(10.dp))
            .clickable(enabled = uri != null) {
                context.startActivity(Intent(Intent.ACTION_VIEW, uri).addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION))
            }
    ) {
        bitmap?.let { Image(it.asImageBitmap(), null, Modifier.fillMaxSize(), contentScale = ContentScale.Crop) }
    }
}

/**
 * The back camera reaching the lowest zoom. Phones often list the main lens alone first (1x only),
 * with the logical multi-camera that includes the ultra-wide (e.g. 0.6x) under another id.
 */
private fun widestBackCamera(provider: ProcessCameraProvider): CameraSelector =
    provider.availableCameraInfos
        .filter { it.lensFacing == CameraSelector.LENS_FACING_BACK }
        .minByOrNull { it.zoomState.value?.minZoomRatio ?: 1f }
        ?.cameraSelector
        ?: CameraSelector.DEFAULT_BACK_CAMERA

private fun ratio4x3() = ResolutionSelector.Builder()
    .setAspectRatioStrategy(AspectRatioStrategy.RATIO_4_3_FALLBACK_AUTO_STRATEGY)

@OptIn(ExperimentalCamera2Interop::class)
private fun lensOf(info: CameraInfo): Lens {
    val c = Camera2CameraInfo.from(info)
    val focals = c.getCameraCharacteristic(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS)
    val sensor = c.getCameraCharacteristic(CameraCharacteristics.SENSOR_INFO_PHYSICAL_SIZE)
    if (focals == null || focals.isEmpty() || sensor == null) return Lens.DEFAULT
    // 1x is the main camera, ~24 mm full-frame equivalent; distrust values far from a phone main lens.
    val diagonal = hypot(sensor.width, sensor.height)
    val focal = focals.minBy { abs(it * 43.27f / diagonal - 24f) }
    val fx = focal / min(sensor.width, sensor.height)
    return if (fx in 0.7f..1.4f) Lens(fx, fx * 0.75f) else Lens.DEFAULT
}

private suspend fun takePicture(context: Context, capture: ImageCapture): Uri =
    suspendCancellableCoroutine { cont ->
        val name = "CamerAI_" + SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
        val values = ContentValues().apply {
            put(MediaStore.MediaColumns.DISPLAY_NAME, name)
            put(MediaStore.MediaColumns.MIME_TYPE, "image/jpeg")
            put(MediaStore.MediaColumns.RELATIVE_PATH, "Pictures/CamerAI")
        }
        val options = ImageCapture.OutputFileOptions
            .Builder(context.contentResolver, MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values)
            .build()
        capture.takePicture(options, ContextCompat.getMainExecutor(context), object : ImageCapture.OnImageSavedCallback {
            override fun onImageSaved(output: ImageCapture.OutputFileResults) {
                val uri = output.savedUri
                if (uri != null) cont.resume(uri) else cont.resumeWithException(IllegalStateException("no uri"))
            }

            override fun onError(exception: ImageCaptureException) = cont.resumeWithException(exception)
        })
    }
