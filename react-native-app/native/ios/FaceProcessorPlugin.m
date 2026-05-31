/*
 * Objective-C bridge that registers the Swift plugin with VisionCamera under the
 * name "faceProcessor" (matching initFrameProcessorPlugin on the JS side).
 * Add this file to the iOS target alongside FaceProcessorPlugin.swift.
 */

#import <VisionCamera/FrameProcessorPlugin.h>
#import <VisionCamera/FrameProcessorPluginRegistry.h>
#import "EdgeFaceSentinel-Swift.h"   // generated Swift header (use your module name)

@interface FaceProcessorPlugin (Registration)
@end

@implementation FaceProcessorPlugin (Registration)

+ (void)load {
  [FrameProcessorPluginRegistry addFrameProcessorPlugin:@"faceProcessor"
                                        withInitializer:^FrameProcessorPlugin* _Nonnull(VisionCameraProxyHolder* _Nonnull proxy,
                                                                                        NSDictionary* _Nullable options) {
    return [[FaceProcessorPlugin alloc] initWithProxy:proxy options:options];
  }];
}

@end
