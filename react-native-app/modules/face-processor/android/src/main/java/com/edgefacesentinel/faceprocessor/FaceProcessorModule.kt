/*
 * FaceProcessorModule.kt — Expo module whose only job is to register the
 * `faceProcessor` VisionCamera frame-processor plugin at app start. Expo runs
 * OnCreate once when the native module graph initialises, which is before the
 * camera mounts, so the plugin is ready when FaceCamera looks it up.
 */
package com.edgefacesentinel.faceprocessor

import expo.modules.kotlin.modules.Module
import expo.modules.kotlin.modules.ModuleDefinition
import com.mrousavy.camera.frameprocessors.FrameProcessorPluginRegistry

class FaceProcessorModule : Module() {
  override fun definition() = ModuleDefinition {
    Name("FaceProcessor")

    OnCreate {
      FrameProcessorPluginRegistry.addFrameProcessorPlugin("faceProcessor") { proxy, options ->
        FaceProcessorPlugin(proxy, options)
      }
    }
  }
}
