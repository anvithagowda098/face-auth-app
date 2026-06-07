/*
 * FaceProcessorPlugin.kt — VisionCamera frame-processor plugin (Android).
 *
 * Per frame: runs ML Kit face detection, selects the largest face, derives pose
 * + liveness signals, and returns a lightweight status map. When called with
 * { capture: true } it also warps the face onto the 112x112 ArcFace template
 * (same closed-form similarity transform as src/core/geometry.ts, validated to
 * machine precision against skimage) and returns the RGB bytes as base64.
 *
 * Registered by FaceProcessorModule (Expo OnCreate). Gradle deps live in this
 * module's build.gradle:
 *   implementation 'com.google.mlkit:face-detection:16.1.6'
 *   (vision-camera + worklets-core are autolinked into the app)
 *
 * Performance: this reference converts the YUV frame via YuvImage for clarity.
 * For production, downscale the analysis frame and reuse a bitmap buffer.
 */

package com.edgefacesentinel.faceprocessor

import android.graphics.*
import android.util.Base64
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.face.Face
import com.google.mlkit.vision.face.FaceDetection
import com.google.mlkit.vision.face.FaceDetectorOptions
import com.google.mlkit.vision.face.FaceLandmark
import com.mrousavy.camera.frameprocessors.Frame
import com.mrousavy.camera.frameprocessors.FrameProcessorPlugin
import com.mrousavy.camera.frameprocessors.VisionCameraProxy
import com.google.android.gms.tasks.Tasks
import java.io.ByteArrayOutputStream

private const val INPUT = 112

// ArcFace template (leftEye, rightEye, nose, leftMouth, rightMouth).
private val TEMPLATE = arrayOf(
  floatArrayOf(38.2946f, 51.6963f),
  floatArrayOf(73.5318f, 51.5014f),
  floatArrayOf(56.0252f, 71.7366f),
  floatArrayOf(41.5493f, 92.3655f),
  floatArrayOf(70.7299f, 92.2041f),
)

class FaceProcessorPlugin(proxy: VisionCameraProxy, options: Map<String, Any>?) :
  FrameProcessorPlugin() {

  private val detector = FaceDetection.getClient(
    FaceDetectorOptions.Builder()
      .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_FAST)
      .setLandmarkMode(FaceDetectorOptions.LANDMARK_MODE_ALL)
      .setClassificationMode(FaceDetectorOptions.CLASSIFICATION_MODE_ALL)
      .setMinFaceSize(0.10f)
      .build()
  )

  override fun callback(frame: Frame, arguments: Map<String, Any>?): Any? {
    val wantCapture = (arguments?.get("capture") as? Boolean) ?: false
    val mediaImage = frame.image ?: return notFound()
    val rotation = frame.imageProxy.imageInfo.rotationDegrees
    val input = InputImage.fromMediaImage(mediaImage, rotation)

    val faces = try {
      Tasks.await(detector.process(input))
    } catch (e: Exception) {
      return notFound()
    }
    if (faces.isEmpty()) return notFound()

    // largest face by bbox area
    val face = faces.maxByOrNull { it.boundingBox.width() * it.boundingBox.height() }!!
    val lm = landmarks(face) ?: return notFound()

    val frameW = input.width.toFloat()
    val frameH = input.height.toFloat()
    val faceRatio =
      (face.boundingBox.width() * face.boundingBox.height()) / (frameW * frameH)

    val out = HashMap<String, Any?>()
    out["found"] = true
    out["confidence"] = 0.9 // ML Kit gives no per-face score; gate on landmarks+ratio
    out["faceRatio"] = faceRatio.toDouble()
    out["yaw"] = face.headEulerAngleY.toDouble()
    out["pitch"] = face.headEulerAngleX.toDouble()
    out["roll"] = face.headEulerAngleZ.toDouble()
    out["leftEyeOpen"] = (face.leftEyeOpenProbability ?: -1f).toDouble()
    out["rightEyeOpen"] = (face.rightEyeOpenProbability ?: -1f).toDouble()
    out["smiling"] = (face.smilingProbability ?: -1f).toDouble()
    out["landmarks"] = normalisedLandmarks(lm, frameW, frameH)

    if (wantCapture) {
      val bitmap = yuvToBitmap(frame, rotation)
      if (bitmap != null) {
        out["alignedRgbB64"] = warpAndEncode(bitmap, lm)
      }
    }
    return out
  }

  // ── landmarks ──────────────────────────────────────────────────────────────

  private fun landmarks(face: Face): Array<FloatArray>? {
    val le = face.getLandmark(FaceLandmark.LEFT_EYE)?.position
    val re = face.getLandmark(FaceLandmark.RIGHT_EYE)?.position
    val no = face.getLandmark(FaceLandmark.NOSE_BASE)?.position
    val ml = face.getLandmark(FaceLandmark.MOUTH_LEFT)?.position
    val mr = face.getLandmark(FaceLandmark.MOUTH_RIGHT)?.position
    if (le == null || re == null || no == null || ml == null || mr == null) return null
    return arrayOf(
      floatArrayOf(le.x, le.y), floatArrayOf(re.x, re.y), floatArrayOf(no.x, no.y),
      floatArrayOf(ml.x, ml.y), floatArrayOf(mr.x, mr.y),
    )
  }

  private fun normalisedLandmarks(lm: Array<FloatArray>, w: Float, h: Float): Map<String, Any> {
    fun p(i: Int) = mapOf("x" to (lm[i][0] / w).toDouble(), "y" to (lm[i][1] / h).toDouble())
    return mapOf(
      "leftEye" to p(0), "rightEye" to p(1), "nose" to p(2),
      "leftMouth" to p(3), "rightMouth" to p(4),
    )
  }

  // ── alignment warp (similarity transform, identical to geometry.ts) ──────────

  private fun warpAndEncode(src: Bitmap, lm: Array<FloatArray>): String {
    val m = estimateSimilarity(lm, TEMPLATE) // [a,b,c,d,tx,ty]
    val matrix = Matrix()
    // Android Matrix is [MSCALE_X, MSKEW_X, MTRANS_X, MSKEW_Y, MSCALE_Y, MTRANS_Y, 0,0,1]
    matrix.setValues(floatArrayOf(m[0], m[2], m[4], m[1], m[3], m[5], 0f, 0f, 1f))

    val dst = Bitmap.createBitmap(INPUT, INPUT, Bitmap.Config.ARGB_8888)
    val canvas = Canvas(dst)
    canvas.drawBitmap(src, matrix, Paint(Paint.FILTER_BITMAP_FLAG))

    val pixels = IntArray(INPUT * INPUT)
    dst.getPixels(pixels, 0, INPUT, 0, 0, INPUT, INPUT)
    val rgb = ByteArray(INPUT * INPUT * 3)
    for (i in pixels.indices) {
      val px = pixels[i]
      rgb[i * 3] = ((px shr 16) and 0xFF).toByte()     // R
      rgb[i * 3 + 1] = ((px shr 8) and 0xFF).toByte()  // G
      rgb[i * 3 + 2] = (px and 0xFF).toByte()          // B
    }
    return Base64.encodeToString(rgb, Base64.NO_WRAP)
  }

  /** Closed-form 2-D similarity LS (no reflection). Returns [a,b,c,d,tx,ty]. */
  private fun estimateSimilarity(src: Array<FloatArray>, dst: Array<FloatArray>): FloatArray {
    val n = src.size
    var msx = 0f; var msy = 0f; var mdx = 0f; var mdy = 0f
    for (i in 0 until n) { msx += src[i][0]; msy += src[i][1]; mdx += dst[i][0]; mdy += dst[i][1] }
    msx /= n; msy /= n; mdx /= n; mdy /= n
    var num1 = 0f; var num2 = 0f; var den = 0f
    for (i in 0 until n) {
      val x = src[i][0] - msx; val y = src[i][1] - msy
      val u = dst[i][0] - mdx; val v = dst[i][1] - mdy
      num1 += x * u + y * v
      num2 += x * v - y * u
      den += x * x + y * y
    }
    val a = num1 / den; val b = num2 / den
    val tx = mdx - (a * msx - b * msy)
    val ty = mdy - (b * msx + a * msy)
    return floatArrayOf(a, b, -b, a, tx, ty)
  }

  // ── frame -> bitmap ──────────────────────────────────────────────────────────

  private fun yuvToBitmap(frame: Frame, rotation: Int): Bitmap? {
    val image = frame.image ?: return null
    val nv21 = yuv420ToNv21(image)
    val yuv = YuvImage(nv21, ImageFormat.NV21, image.width, image.height, null)
    val out = ByteArrayOutputStream()
    yuv.compressToJpeg(Rect(0, 0, image.width, image.height), 90, out)
    val bytes = out.toByteArray()
    val bmp = BitmapFactory.decodeByteArray(bytes, 0, bytes.size) ?: return null
    if (rotation == 0) return bmp
    val m = Matrix().apply { postRotate(rotation.toFloat()) }
    return Bitmap.createBitmap(bmp, 0, 0, bmp.width, bmp.height, m, true)
  }

  private fun yuv420ToNv21(image: android.media.Image): ByteArray {
    val y = image.planes[0].buffer
    val u = image.planes[1].buffer
    val v = image.planes[2].buffer
    val ySize = y.remaining(); val uSize = u.remaining(); val vSize = v.remaining()
    val nv21 = ByteArray(ySize + uSize + vSize)
    y.get(nv21, 0, ySize)
    v.get(nv21, ySize, vSize)
    u.get(nv21, ySize + vSize, uSize)
    return nv21
  }

  private fun notFound(): Map<String, Any?> = mapOf(
    "found" to false, "confidence" to 0.0, "faceRatio" to 0.0,
    "yaw" to 0.0, "pitch" to 0.0, "roll" to 0.0,
    "leftEyeOpen" to -1.0, "rightEyeOpen" to -1.0, "smiling" to -1.0,
    "landmarks" to null,
  )
}
