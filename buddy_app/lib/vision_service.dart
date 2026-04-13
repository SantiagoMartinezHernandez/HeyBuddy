import 'dart:js_interop';
import 'dart:convert';

// Bindings to the JavaScript functions in mediapipe_bridge.js
@JS('startCamera')
external void _startCamera();

@JS('registerDartCallback')
external void _registerDartCallback(JSFunction callback);

class VisionService {
  // We will expose this stream to our Flutter UI
  Function(List<dynamic>)? onLandmarksUpdated;

  void initialize() {
    // 1. Tell JS to send data to our Dart function
    _registerDartCallback(_handleJSResults.toJS);
    
    // 2. Fire up the camera!
    _startCamera();
  }

  // This is the function JS will call 30 times a second
  void _handleJSResults(JSString jsonString) {
    if (onLandmarksUpdated != null) {
      final String data = jsonString.toDart;
      final List<dynamic> landmarks = jsonDecode(data);
      onLandmarksUpdated!(landmarks);
    }
  }
}