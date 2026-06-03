Pod::Spec.new do |s|
  s.name           = 'FaceProcessor'
  s.version        = '0.1.0'
  s.summary        = 'ArcFace face-warp VisionCamera frame processor (local Expo module)'
  s.description    = 'MLKit detect + closed-form similarity warp to a 112x112 ArcFace crop.'
  s.author         = 'NHAI'
  s.homepage       = 'https://example.com'
  s.platforms      = { :ios => '15.1', :tvos => '15.1' }
  s.source         = { git: '' }
  s.static_framework = true

  s.dependency 'ExpoModulesCore'
  s.dependency 'VisionCamera'
  s.dependency 'GoogleMLKit/FaceDetection'

  # Swift compiles inside this pod; no app bridging header needed.
  s.source_files = "**/*.{h,m,mm,swift}"
end
