/*
 * Register the plugin once at app start. Add this call inside your
 * MainApplication.onCreate() (after super.onCreate()), or paste the body into an
 * existing ReactPackage. VisionCamera looks the plugin up by the name passed to
 * initFrameProcessorPlugin('faceProcessor') on the JS side.
 */

package com.edgefacesentinel

import com.mrousavy.camera.frameprocessors.FrameProcessorPluginRegistry

object FaceProcessorRegistration {
  fun register() {
    FrameProcessorPluginRegistry.addFrameProcessorPlugin("faceProcessor") { proxy, options ->
      FaceProcessorPlugin(proxy, options)
    }
  }
}

// In MainApplication.onCreate():
//   FaceProcessorRegistration.register()
