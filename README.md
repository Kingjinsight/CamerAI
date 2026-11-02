# CamerAI

Real-time composition guidance for Android, in the spirit of Huawei's AI framing assistant.
The camera looks at a wider field of view than the viewfinder. An on-device cropping model
scores possible framings around the current shot, and the app guides you to the best one by
telling you which way to move the phone and how far to zoom in or out. When the shot is lined
up and steady, it can take the photo for you.

## How it works

1. **Wide context.** The camera runs about 1.5× wider than the viewfinder, using the
   ultra-wide lens when the phone exposes it (zoom below 1×). The viewfinder is a digital crop
   of that stream, so the model can see what lies just outside the shot.
2. **Scoring.** Every analysis frame (240×320) goes through a GAIC model
   ([Grid Anchor based Image Cropping](Papers/Grid_Anchor_based_Image_Cropping__A_New_Benchmark_and_An_Efficient_Model.pdf),
   MobileNetV2, RoI + RoD alignment). The model scores about 200 candidate framings: the
   current view plus shifted, larger (zoom out) and smaller (zoom in) boxes, all at the
   photo's 3:4 aspect ratio.
3. **Decision.** GAIC prefers bigger crops, so part of its measured size bias (`size_prior`, from
   `training/calibrate.py`, calibrated separately for 1.5× context and for 1× with no context) is
   subtracted, along with a penalty for distance from the current view.
   - With context, the best candidate must beat the current view by 0.4 MOS standard deviations
     (~68% of GAICD frames get guidance).
   - At 1× with no ultra-wide, candidates zoom in and shift inside the frame, and the threshold is
     0.3. About 72% of frames get a "zoom to ~1.5× and move" target, and the chosen framing stays
     put after a 3% frame shift 89% of the time.
   - If nothing passes, a pan-only fallback compares equal-size windows at 5×5 positions.
4. **People.** ML Kit's bundled face detector finds heads in each frame. Framings that cut through a
   head (−1.5) or drop someone who is in the current shot (−1.0) are penalized, and the main
   person's eye line is pulled toward the upper third. The current view gets the same penalties,
   so a shot that clips someone triggers guidance even when the crop model is happy.
5. **World anchoring.** After 3 consistent frames, the target is locked as a direction in world
   space, using the rotation-vector sensor. It stays in place on screen while you move or zoom,
   and the hints are recomputed from the gyroscope at 60 Hz, not just at model rate. The lock
   persists. The locked target is re-scored with every frame and is replaced only when another
   framing beats it by 0.5 for 6 frames in a row. It is dropped after capture, after 8 s out of
   view, or when you tap **重新取景** ("re-frame").
6. **Guidance.** The overlay shows the target box, a target dot to bring into the center ring,
   an edge arrow when the target is off screen, and a level line.
   Zooming is digital at once. The camera follows about 1.5× wider in small per-frame steps, and the
   digital crop tracks the zoom each preview frame actually arrived at (`CONTROL_ZOOM_RATIO`), so
   zooming is smooth and never shows black edges. Text hints read like `向右上移动 · 放大到 1.8×`
   ("move up and right · zoom to 1.8×"). Tap `对准 x×` ("go to x×") to zoom to the target
   automatically.
7. **Capture.** When the shot is aligned (center within 5%, size within 8%) and held steady for
   0.8 s, auto-capture fires. The app switches to the viewfinder's native zoom for the shot
   (main lens, not a crop of the ultra-wide) and saves to `Pictures/CamerAI`.

The debug line at the bottom of the preview shows `lens zoom range · viewfinder zoom · camera
zoom · inference ms`. A range starting at 1.0 means the phone does not expose its ultra-wide
lens to third-party apps. In that case the model can still suggest panning and zooming in,
but not zooming out.

## Layout

```
training/   PyTorch model, GAICD loader, training/eval, ONNX export
android/    Kotlin + Jetpack Compose + CameraX app, ONNX Runtime inference
Papers/     reference papers
```

## Train

Uses the `general-ai` conda env (PyTorch with CUDA, MPS or CPU). Unzip GAICD to `data/GAIC`
(`images/{train,test}`, `annotations/*.txt` with `y1 x1 y2 x2 MOS` per line).

```bash
python training/train.py --data data/GAIC --out checkpoints      # 80 epochs, ~25 min on an RTX 3070
python training/export.py --ckpt checkpoints/best.pt               # -> android/app/src/main/assets/gaic.{onnx,json}
python training/calibrate.py                                       # adds size_prior to gaic.json
```

100 training images are held out to pick the checkpoint. Test-set metrics (GAICD v1, 200
images) are written to `checkpoints/test_metrics.json`. The bundled model (epoch 42) scores
SRCC 0.784, Acc5 46.3 and Acc10 67.0 on that test set.

On a phone that does not expose its ultra-wide lens, the camera cannot be wider than the
viewfinder at 1×, so 1× guidance can only zoom in and shift, and zoom-out hints start at about
1.25×. vivo
X100 Ultra (OriginOS 6) is one such phone: its 0.6× logical camera is limited to the packages in
`vendor.camera.aux.packagelist`.

## Build the app

Requires JDK 17 and an Android SDK (platform 35). Set `sdk.dir` in `android/local.properties`.

```bash
cd android
./gradlew assembleDebug testDebugUnitTest
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

Minimum Android 11 (API 30). Portrait only.

## Resources

Dataset link:
- GAICD: https://drive.google.com/file/d/1KhmyjoimsQVXqPnLjKZiU4iXNKNyyxqW/view
- CADB: https://drive.google.com/file/d/1fpZoo5exRfoarqDvdLDpQVXVOKFW63vz/view
