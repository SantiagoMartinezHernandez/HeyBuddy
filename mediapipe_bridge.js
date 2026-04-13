// ... [Start of mediapipe_bridge.js] ...

// 1. Create the video element
const videoElement = document.createElement('video');
videoElement.style.display = 'block';
videoElement.style.position = 'absolute';
videoElement.style.width = '100vw';
videoElement.style.height = '100vh';
videoElement.style.objectFit = 'cover'; // We keep this for proper responsiveness math!
videoElement.style.transform = 'scaleX(-1)'; // Mirror effect
videoElement.style.zIndex = '-1'; // Push behind Flutter
videoElement.style.border = 'none'; // CRITICAL FIX: Ensure no browser borders
videoElement.setAttribute('playsinline', ''); // Essential for iOS
videoElement.setAttribute('webkit-playsinline', ''); 
document.body.appendChild(videoElement);

// PATCH: The Polished Multimodal Black Background
// This ensures any letterboxing area (due to 'contain') is clean black.
document.body.style.backgroundColor = 'black'; 
document.body.style.margin = '0'; // Removes clean edges
document.body.style.overflow = 'hidden'; // Prevents unnecessary scrollbars

let dartCallback = null;

// Flutter will call this to pass its listening function
window.registerDartCallback = function(callback) {
    dartCallback = callback;
};

const pose = new Pose({locateFile: (file) => {
  return `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${file}`;
}});

pose.setOptions({
  modelComplexity: 1, 
  smoothLandmarks: true,
  minDetectionConfidence: 0.6,
  minTrackingConfidence: 0.6
});

pose.onResults((results) => {
    if (results.poseLandmarks && dartCallback) {
        // Map the results to a simple JSON string to easily pass to Dart
        const landmarks = results.poseLandmarks.map(lm => ({x: lm.x, y: lm.y, z: lm.z, visibility: lm.visibility}));
        dartCallback(JSON.stringify(landmarks));
    }
});

// Determine if the device is in portrait mode
const isPortrait = window.innerHeight > window.innerWidth;

// Dynamically set resolution based on orientation
const idealWidth = isPortrait ? 480 : 640;
const idealHeight = isPortrait ? 640 : 480;

const camera = new Camera(videoElement, {
  onFrame: async () => {
    await pose.send({image: videoElement});
  },
  width: idealWidth,
  height: idealHeight,
  facingMode: 'user' // Ensures we get the front selfie camera on mobile!
});

window.startCamera = function() {
    camera.start();
}