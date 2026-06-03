/*
 * FaceProcessorModule.swift — Expo module that registers the `faceProcessor`
 * VisionCamera frame-processor plugin at startup (OnCreate). Registering from
 * the module avoids needing an ObjC `+load` category and an app bridging
 * header — the Swift plugin compiles inside this module's pod.
 */
import ExpoModulesCore
import VisionCamera

public class FaceProcessorModule: Module {
  public func definition() -> ModuleDefinition {
    Name("FaceProcessor")

    OnCreate {
      FrameProcessorPluginRegistry.addFrameProcessorPlugin("faceProcessor") { proxy, options in
        FaceProcessorPlugin(proxy: proxy, options: options)
      }
    }
  }
}
