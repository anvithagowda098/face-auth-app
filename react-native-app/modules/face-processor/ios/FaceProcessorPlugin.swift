/*
 * FaceProcessorPlugin.swift — VisionCamera frame-processor plugin (iOS).
 *
 * Mirror of the Android plugin: ML Kit face detection -> status map, and on
 * { capture: true } an ArcFace-aligned 112x112 RGB crop returned as base64,
 * using the same closed-form similarity transform as src/core/geometry.ts.
 *
 * Registered by FaceProcessorModule (Expo OnCreate). Pods (see podspec):
 *   GoogleMLKit/FaceDetection, VisionCamera, ExpoModulesCore
 *
 * Performance note: same as Android — for production, run detection on a
 * downscaled frame and reuse buffers.
 */

import Foundation
import VisionCamera
import MLKitVision
import MLKitFaceDetection
import CoreGraphics

private let INPUT = 112

// ArcFace template: leftEye, rightEye, nose, leftMouth, rightMouth.
private let TEMPLATE: [(CGFloat, CGFloat)] = [
  (38.2946, 51.6963), (73.5318, 51.5014), (56.0252, 71.7366),
  (41.5493, 92.3655), (70.7299, 92.2041)
]

@objc(FaceProcessorPlugin)
public class FaceProcessorPlugin: FrameProcessorPlugin {
  private let detector: FaceDetector

  public override init(proxy: VisionCameraProxyHolder, options: [AnyHashable: Any]! = [:]) {
    let opts = FaceDetectorOptions()
    opts.performanceMode = .fast
    opts.landmarkMode = .all
    opts.classificationMode = .all
    opts.minFaceSize = 0.10
    detector = FaceDetector.faceDetector(options: opts)
    super.init(proxy: proxy, options: options)
  }

  public override func callback(_ frame: Frame, withArguments arguments: [AnyHashable: Any]?) -> Any? {
    let wantCapture = (arguments?["capture"] as? Bool) ?? false
    let buffer = frame.buffer
    let image = VisionImage(buffer: buffer)
    image.orientation = frame.orientation

    guard let faces = try? detector.results(in: image), !faces.isEmpty else {
      return notFound()
    }
    let face = faces.max(by: { $0.frame.width * $0.frame.height < $1.frame.width * $1.frame.height })!
    guard let lm = landmarks(face) else { return notFound() }

    let w = CGFloat(CVPixelBufferGetWidth(CMSampleBufferGetImageBuffer(buffer)!))
    let h = CGFloat(CVPixelBufferGetHeight(CMSampleBufferGetImageBuffer(buffer)!))
    let faceRatio = (face.frame.width * face.frame.height) / (w * h)

    var out: [String: Any] = [
      "found": true,
      "confidence": 0.9,
      "faceRatio": Double(faceRatio),
      "yaw": Double(face.headEulerAngleY),
      "pitch": Double(face.headEulerAngleX),
      "roll": Double(face.headEulerAngleZ),
      "leftEyeOpen": face.hasLeftEyeOpenProbability ? Double(face.leftEyeOpenProbability) : -1.0,
      "rightEyeOpen": face.hasRightEyeOpenProbability ? Double(face.rightEyeOpenProbability) : -1.0,
      "smiling": face.hasSmilingProbability ? Double(face.smilingProbability) : -1.0,
      "landmarks": normalisedLandmarks(lm, w, h),
    ]

    if wantCapture, let cg = cgImage(from: buffer) {
      out["alignedRgbB64"] = warpAndEncode(cg, lm)
    }
    return out
  }

  private func landmarks(_ face: Face) -> [(CGFloat, CGFloat)]? {
    guard
      let le = face.landmark(ofType: .leftEye)?.position,
      let re = face.landmark(ofType: .rightEye)?.position,
      let no = face.landmark(ofType: .noseBase)?.position,
      let ml = face.landmark(ofType: .mouthLeft)?.position,
      let mr = face.landmark(ofType: .mouthRight)?.position
    else { return nil }
    return [(le.x, le.y), (re.x, re.y), (no.x, no.y), (ml.x, ml.y), (mr.x, mr.y)]
  }

  private func normalisedLandmarks(_ lm: [(CGFloat, CGFloat)], _ w: CGFloat, _ h: CGFloat) -> [String: Any] {
    func p(_ i: Int) -> [String: Double] { ["x": Double(lm[i].0 / w), "y": Double(lm[i].1 / h)] }
    return ["leftEye": p(0), "rightEye": p(1), "nose": p(2), "leftMouth": p(3), "rightMouth": p(4)]
  }

  // Closed-form similarity LS (no reflection) -> [a,b,c,d,tx,ty].
  private func estimateSimilarity(_ src: [(CGFloat, CGFloat)], _ dst: [(CGFloat, CGFloat)]) -> [CGFloat] {
    let n = CGFloat(src.count)
    let msx = src.map { $0.0 }.reduce(0,+) / n, msy = src.map { $0.1 }.reduce(0,+) / n
    let mdx = dst.map { $0.0 }.reduce(0,+) / n, mdy = dst.map { $0.1 }.reduce(0,+) / n
    var num1: CGFloat = 0, num2: CGFloat = 0, den: CGFloat = 0
    for i in 0..<src.count {
      let x = src[i].0 - msx, y = src[i].1 - msy
      let u = dst[i].0 - mdx, v = dst[i].1 - mdy
      num1 += x*u + y*v; num2 += x*v - y*u; den += x*x + y*y
    }
    let a = num1/den, b = num2/den
    return [a, b, -b, a, mdx - (a*msx - b*msy), mdy - (b*msx + a*msy)]
  }

  private func warpAndEncode(_ src: CGImage, _ lm: [(CGFloat, CGFloat)]) -> String? {
    let m = estimateSimilarity(lm, TEMPLATE)
    let space = CGColorSpaceCreateDeviceRGB()
    guard let ctx = CGContext(
      data: nil, width: INPUT, height: INPUT, bitsPerComponent: 8, bytesPerRow: INPUT * 4,
      space: space, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
    ) else { return nil }

    // CoreGraphics origin is bottom-left; flip Y so the affine matches image coords.
    ctx.translateBy(x: 0, y: CGFloat(INPUT))
    ctx.scaleBy(x: 1, y: -1)
    // x' = a*x + c*y + tx ; y' = b*x + d*y + ty
    ctx.concatenate(CGAffineTransform(a: m[0], b: m[1], c: m[2], d: m[3], tx: m[4], ty: m[5]))
    ctx.draw(src, in: CGRect(x: 0, y: 0, width: src.width, height: src.height))

    guard let data = ctx.data else { return nil }
    let ptr = data.bindMemory(to: UInt8.self, capacity: INPUT * INPUT * 4)
    var rgb = [UInt8](repeating: 0, count: INPUT * INPUT * 3)
    for i in 0..<(INPUT * INPUT) {
      rgb[i*3] = ptr[i*4]; rgb[i*3+1] = ptr[i*4+1]; rgb[i*3+2] = ptr[i*4+2]
    }
    return Data(rgb).base64EncodedString()
  }

  private func cgImage(from buffer: CMSampleBuffer) -> CGImage? {
    guard let px = CMSampleBufferGetImageBuffer(buffer) else { return nil }
    let ci = CIImage(cvPixelBuffer: px)
    return CIContext().createCGImage(ci, from: ci.extent)
  }

  private func notFound() -> [String: Any] {
    ["found": false, "confidence": 0.0, "faceRatio": 0.0, "yaw": 0.0, "pitch": 0.0, "roll": 0.0,
     "leftEyeOpen": -1.0, "rightEyeOpen": -1.0, "smiling": -1.0, "landmarks": NSNull()]
  }
}
