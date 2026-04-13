import 'package:flutter/material.dart';
import 'vision_service.dart';
import 'package:speech_to_text/speech_to_text.dart'; // AUDIO: Import the package
import 'dart:math' as math;

enum AppState { calibrating, menu, exercise, stats }

void main() {
  runApp(const BuddyApp());
}

class BuddyApp extends StatelessWidget {
  const BuddyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Hey Buddy Demo',
      theme: ThemeData.dark().copyWith(
        // Make the background transparent so the JS video shows through!
        scaffoldBackgroundColor: Colors.transparent,
      ),
      home: const CalibrationScreen(),
    );
  }
}

class CalibrationScreen extends StatefulWidget {
  const CalibrationScreen({super.key});

  @override
  State<CalibrationScreen> createState() => _CalibrationScreenState();
}

// ... (Keep your existing SkeletonPainter and BuddyApp classes exactly the same)

class _CalibrationScreenState extends State<CalibrationScreen> {
  final VisionService _visionService = VisionService();
  List<dynamic> _currentLandmarks = [];

  // App State Management
  AppState _currentState = AppState.calibrating;

  // Calibration Variables
  bool _isCalibrated = false;
  String _instructionText = "Step back.\nAlign your full body in the frame.";

  // Audio Variables
  final SpeechToText _speechToText = SpeechToText();
  bool _speechEnabled = false;
  String _lastWords = '';
  bool _isMicActive = false;

  //Hover UI Variables
  String? _activeHover; // 'exercise' or 'stats'
  DateTime? _hoverStartTime;
  double _hoverProgress = 0.0;

  //Exercise Tracking Variables
  int _repCount = 0;
  bool _isSquattingDown = false;
  double _currentKneeAngle = 180.0; // Start standing straight

  @override
  void initState() {
    super.initState();
    _initSpeech();

    _visionService.onLandmarksUpdated = (landmarks) {
      if (landmarks.length >= 33) {
        setState(() {
          _currentLandmarks = landmarks;
        });

        if (_currentState == AppState.calibrating) {
          _checkCalibration(landmarks);
        } else if (_currentState == AppState.menu) {
          _processHover(landmarks);
        } else if (_currentState == AppState.exercise) {
          _processExercise(landmarks); // NEW!
        }
      }
    };

    _visionService.initialize();
  }

  void _initSpeech() async {
    _speechEnabled = await _speechToText.initialize(
      onError: (val) => print('Speech Error: $val'),
      onStatus: (val) {
        if (val == 'done' && _isMicActive) {
          _startListening();
        }
      },
    );
    setState(() {});
  }

  void _startListening() async {
    await _speechToText.listen(
      localeId: 'en_US',
      onResult: (result) {
        setState(() {
          _lastWords = result.recognizedWords.toLowerCase();

          if (_lastWords.contains("bye bye")) {
            _isMicActive = false;
            _currentState = AppState.calibrating;
            _instructionText = "Buddy resting.\nStep into frame to wake.";
            _stopListening();
            return; // Stop evaluating!
          }

          // State Machine Routing
          if (_currentState == AppState.calibrating &&
              _lastWords.contains("hey buddy")) {
            print("🟢 Moving to Menu!");
            _currentState = AppState.menu;
            _lastWords = '';
          } else if (_currentState == AppState.menu) {
            if (_lastWords.contains("exercise")) {
              _triggerMenuAction('exercise');
            } else if (_lastWords.contains("stats") ||
                _lastWords.contains("statistics")) {
              _triggerMenuAction('stats');
            }
          } else if (_currentState == AppState.exercise) {
            // FIX: Check for the commands anywhere in the string, and forcefully reset everything
            if (_lastWords.contains("stop") ||
                _lastWords.contains("menu") ||
                _lastWords.contains("back")) {
              print("🟢 Voice Command: Stopping exercise!");
              _currentState = AppState.menu;
              _lastWords = '';
              _activeHover = null;
              _hoverProgress = 0.0;
            }
          }
        });
      },
      listenFor: const Duration(seconds: 60),
      pauseFor: const Duration(seconds: 10),
      partialResults: true,
      cancelOnError: false,
    );
  }

  void _stopListening() async {
    await _speechToText.stop();
  }

  // --- CALIBRATION LOGIC ---
  void _checkCalibration(List<dynamic> landmarks) {
    // ... (Keep your exact same calibration logic here)
    final shouldersVisible =
        landmarks[11]['visibility'] > 0.7 && landmarks[12]['visibility'] > 0.7;
    final hipsVisible =
        landmarks[23]['visibility'] > 0.7 && landmarks[24]['visibility'] > 0.7;
    final anklesVisible =
        landmarks[27]['visibility'] > 0.7 && landmarks[28]['visibility'] > 0.7;

    final isNowCalibrated = shouldersVisible && hipsVisible && anklesVisible;

    if (isNowCalibrated != _isCalibrated) {
      setState(() {
        _isCalibrated = isNowCalibrated;
        if (_isCalibrated) {
          _instructionText = "Perfect.\nSay 'Hey Buddy' to begin.";
          if (_speechEnabled && !_isMicActive) {
            _isMicActive = true;
            _startListening();
          }
        } else {
          _instructionText = "Step back.\nAlign your full body in the frame.";
        }
      });
    }
  }

  // --- NEW: HOVER UI LOGIC ---
  void _processHover(List<dynamic> landmarks) {
    bool hoveringExercise = false;
    bool hoveringStats = false;

    for (int node in [15, 16]) {
      if (landmarks[node]['visibility'] > 0.6) {
        double mappedX = 1.0 - landmarks[node]['x'];
        double mappedY = landmarks[node]['y'];

        // WIDENED HITBOXES FOR MOBILE:
        // Top 45% of the screen height, and outer 45% of the width
        if (mappedX < 0.45 && mappedY < 0.45) hoveringExercise = true;
        if (mappedX > 0.55 && mappedY < 0.45) hoveringStats = true;
      }
    }

    if (hoveringExercise) {
      _updateHoverState('exercise');
    } else if (hoveringStats) {
      _updateHoverState('stats');
    } else {
      // User pulled hand away, reset timer!
      if (_hoverProgress > 0.0) {
        setState(() {
          _activeHover = null;
          _hoverStartTime = null;
          _hoverProgress = 0.0;
        });
      }
    }
  }

  void _updateHoverState(String target) {
    if (_activeHover != target) {
      _activeHover = target;
      _hoverStartTime = DateTime.now();
    } else {
      // Calculate how long they've been hovering
      final elapsed =
          DateTime.now().difference(_hoverStartTime!).inMilliseconds;
      setState(() {
        _hoverProgress = (elapsed / 3000.0).clamp(
          0.0,
          1.0,
        ); // 3 seconds = 3000ms
      });

      if (_hoverProgress >= 1.0) {
        _triggerMenuAction(target);
      }
    }
  }

  void _triggerMenuAction(String action) {
    setState(() {
      _hoverProgress = 0.0;
      _activeHover = null;
      _lastWords = ''; // clear mic buffer

      if (action == 'exercise') {
        _currentState = AppState.exercise; // Move to Screen 3!
      } else {
        _currentState = AppState.stats; // Move to Screen 4!
      }
    });
  }

  // --- NEW: EXERCISE LOGIC ---
  void _processExercise(List<dynamic> landmarks) {
    // We will use the Left leg for this example: Hip (23), Knee (25), Ankle (27)
    // You can easily mirror this for the right leg or average them out later!
    var hip = landmarks[23];
    var knee = landmarks[25];
    var ankle = landmarks[27];

    if (hip['visibility'] > 0.6 &&
        knee['visibility'] > 0.6 &&
        ankle['visibility'] > 0.6) {
      double angle = Biomechanics.calculateAngle(hip, knee, ankle);

      setState(() {
        _currentKneeAngle = angle;

        // The Squat State Machine
        if (angle < 90.0 && !_isSquattingDown) {
          // User hit the bottom of the squat!
          _isSquattingDown = true;
          print("🔽 Squat depth reached!");
        } else if (angle > 160.0 && _isSquattingDown) {
          // User stood back up!
          _isSquattingDown = false;
          _repCount++;
          print("✅ Rep Completed! Total: $_repCount");
        }
      });
    }
  }

  // ... (build method continues in next step)
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.transparent,
      body: Stack(
        fit: StackFit.expand,
        children: [
          // The background skeleton
          CustomPaint(painter: SkeletonPainter(_currentLandmarks)),

          // 1. THE CALIBRATION SCREEN
          if (_currentState == AppState.calibrating)
            Center(
              child: AnimatedContainer(
                duration: const Duration(milliseconds: 500),
                curve: Curves.easeInOut,
                width: 400,
                height: 400,
                decoration: BoxDecoration(
                  shape: BoxShape.circle,
                  color: const Color(0xFF121212).withOpacity(0.8),
                  boxShadow: [
                    BoxShadow(
                      color:
                          _isCalibrated
                              ? Colors.greenAccent.withOpacity(0.6)
                              : Colors.blueAccent.withOpacity(0.2),
                      blurRadius: _isCalibrated ? 50 : 20,
                      spreadRadius: _isCalibrated ? 20 : 5,
                    ),
                  ],
                  border: Border.all(
                    color:
                        _isCalibrated ? Colors.greenAccent : Colors.blueAccent,
                    width: 4,
                  ),
                ),
                child: Column(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    Icon(
                      _isCalibrated ? Icons.mic : Icons.person_search,
                      size: 80,
                      color:
                          _isCalibrated
                              ? Colors.greenAccent
                              : Colors.blueAccent,
                    ),
                    const SizedBox(height: 20),
                    Text(
                      _instructionText,
                      textAlign: TextAlign.center,
                      style: TextStyle(
                        fontSize: 22,
                        fontWeight: FontWeight.w500,
                        color: _isCalibrated ? Colors.white : Colors.white70,
                      ),
                    ),
                  ],
                ),
              ),
            ),

          // 2. THE HOVER MENU (SCREEN 2)
          if (_currentState == AppState.menu) _buildHoverMenu(),

          // 3. THE EXERCISE HUD (SCREEN 3)
          if (_currentState == AppState.exercise) _buildExerciseScreen(),
          // Persistent Mic Indicator (FIXED!)
          Positioned(
            bottom: 30,
            left: 30,
            child: AnimatedOpacity(
              opacity: _isMicActive ? 1.0 : 0.0,
              duration: const Duration(milliseconds: 300),
              child: Container(
                padding: const EdgeInsets.symmetric(
                  horizontal: 16,
                  vertical: 12,
                ),
                decoration: BoxDecoration(
                  color: Colors.black54,
                  borderRadius: BorderRadius.circular(30),
                  border: Border.all(color: Colors.greenAccent, width: 2),
                ),
                child: const Row(
                  children: [
                    Icon(Icons.mic, color: Colors.greenAccent, size: 24),
                    SizedBox(width: 10),
                    Text(
                      "Buddy is Active",
                      style: TextStyle(
                        color: Colors.greenAccent,
                        fontWeight: FontWeight.bold,
                        fontSize: 16,
                      ),
                    ),
                  ],
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }

  // --- NEW: THE EXERCISE UI ---
  Widget _buildExerciseScreen() {
    // Calculate squat progress for the visual bar.
    // 160 degrees = Standing (0% depth). 90 degrees = Full Squat (100% depth).
    double squatProgress = ((160.0 - _currentKneeAngle) / (160.0 - 90.0)).clamp(
      0.0,
      1.0,
    );

    return Stack(
      children: [
        // Top Left: Huge Rep Counter
        Positioned(
          top: 50,
          left: 50,
          child: Container(
            padding: const EdgeInsets.symmetric(horizontal: 40, vertical: 20),
            decoration: BoxDecoration(
              color: Colors.black87,
              borderRadius: BorderRadius.circular(20),
              border: Border.all(
                // Pulse green when a rep is successfully hit!
                color:
                    _isSquattingDown ? Colors.greenAccent : Colors.blueAccent,
                width: 3,
              ),
            ),
            child: Column(
              children: [
                const Text(
                  "SQUATS",
                  style: TextStyle(
                    color: Colors.blueAccent,
                    fontSize: 24,
                    fontWeight: FontWeight.bold,
                    letterSpacing: 2,
                  ),
                ),
                Text(
                  "$_repCount",
                  style: const TextStyle(
                    color: Colors.white,
                    fontSize: 100,
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ],
            ),
          ),
        ),

        // Right Side: Dynamic Depth Gauge
        Positioned(
          top: 50,
          right: 50,
          bottom: 100,
          child: Container(
            width: 80,
            decoration: BoxDecoration(
              color: Colors.black54,
              borderRadius: BorderRadius.circular(40),
              border: Border.all(color: Colors.white30, width: 2),
            ),
            child: Column(
              mainAxisAlignment: MainAxisAlignment.end,
              children: [
                Expanded(
                  child: Padding(
                    padding: const EdgeInsets.all(8.0),
                    child: RotatedBox(
                      quarterTurns: -1, // Make the progress bar vertical
                      child: ClipRRect(
                        borderRadius: BorderRadius.circular(10),
                        child: LinearProgressIndicator(
                          value: squatProgress,
                          backgroundColor: Colors.transparent,
                          valueColor: AlwaysStoppedAnimation<Color>(
                            squatProgress >= 1.0
                                ? Colors.greenAccent
                                : Colors.blueAccent,
                          ),
                        ),
                      ),
                    ),
                  ),
                ),
                const SizedBox(height: 15),
                // Live Angle Readout
                Text(
                  "${_currentKneeAngle.toInt()}°",
                  style: const TextStyle(
                    color: Colors.white,
                    fontSize: 22,
                    fontWeight: FontWeight.bold,
                  ),
                ),
                const SizedBox(height: 20),
              ],
            ),
          ),
        ),

        // Bottom Center: Voice Instructions & Live Subtitles
        Align(
          alignment: Alignment.bottomCenter,
          child: Padding(
            padding: const EdgeInsets.only(bottom: 40.0),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                const Text(
                  "Say 'stop', 'menu', or 'back' to exit",
                  style: TextStyle(
                    color: Colors.white70,
                    fontSize: 22,
                    fontStyle: FontStyle.italic,
                  ),
                ),
                // NEW: Show what the mic is currently hearing so you can debug!
                if (_lastWords.isNotEmpty) ...[
                  const SizedBox(height: 10),
                  Text(
                    '"$_lastWords"',
                    style: const TextStyle(
                      color: Colors.greenAccent,
                      fontSize: 18,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                ],
              ],
            ),
          ),
        ),
      ],
    );
  }

  // --- NEW: THE HOVER MENU UI ---
  Widget _buildHoverMenu() {
    return Stack(
      children: [
        // Top Left Button: Exercise
        Positioned(
          top: 50,
          left: 50,
          child: _buildHoverButton(
            title: "EXERCISE",
            icon: Icons.fitness_center,
            progress: _activeHover == 'exercise' ? _hoverProgress : 0.0,
            color: Colors.blueAccent,
          ),
        ),

        // Top Right Button: Stats
        Positioned(
          top: 50,
          right: 50,
          child: _buildHoverButton(
            title: "STATS",
            icon: Icons.bar_chart,
            progress: _activeHover == 'stats' ? _hoverProgress : 0.0,
            color: Colors.orangeAccent,
          ),
        ),

        // Instructions in the middle
        const Align(
          alignment: Alignment.center,
          child: Text(
            "Hover hands over corners for 3 seconds\nor say 'Hey Buddy, I want to exercise'",
            textAlign: TextAlign.center,
            style: TextStyle(color: Colors.white70, fontSize: 24),
          ),
        ),
      ],
    );
  }

  Widget _buildHoverButton({
    required String title,
    required IconData icon,
    required double progress,
    required Color color,
  }) {
    return Container(
      width: 200,
      height: 200,
      decoration: BoxDecoration(
        color: Colors.black54,
        borderRadius: BorderRadius.circular(20),
        border: Border.all(
          color: progress > 0 ? color : Colors.white30,
          width: progress > 0 ? 4 : 2,
        ),
      ),
      child: Stack(
        alignment: Alignment.center,
        children: [
          // The radial progress filling up!
          SizedBox(
            width: 180,
            height: 180,
            child: CircularProgressIndicator(
              value: progress,
              strokeWidth: 10,
              backgroundColor: Colors.transparent,
              valueColor: AlwaysStoppedAnimation<Color>(color),
            ),
          ),
          Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Icon(icon, size: 60, color: progress > 0 ? color : Colors.white),
              const SizedBox(height: 10),
              Text(
                title,
                style: const TextStyle(
                  fontSize: 24,
                  fontWeight: FontWeight.bold,
                  color: Colors.white,
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }
}

class SkeletonPainter extends CustomPainter {
  final List<dynamic> landmarks;

  SkeletonPainter(this.landmarks);

  @override
  void paint(Canvas canvas, Size size) {
    if (landmarks.isEmpty) return;

    final paintPoint =
        Paint()
          ..color = Colors.greenAccent
          ..strokeWidth = 6
          ..strokeCap = StrokeCap.round;

    final paintLine =
        Paint()
          ..color = Colors.greenAccent.withOpacity(0.5)
          ..strokeWidth = 3;

    // Define the pairs of landmarks to connect (Torso, Arms, Legs)
    final connections = [
      [11, 12], [11, 23], [12, 24], [23, 24], // Torso
      [11, 13], [13, 15], // Right Arm
      [12, 14], [14, 16], // Left Arm
      [23, 25], [25, 27], // Right Leg
      [24, 26], [26, 28], // Left Leg
    ];

    // Draw lines
    for (var connection in connections) {
      final start = landmarks[connection[0]];
      final end = landmarks[connection[1]];

      // Only draw if both points are reasonably visible
      if (start['visibility'] > 0.5 && end['visibility'] > 0.5) {
        // FLIP THE X AXIS for the mirror effect: (1.0 - x)
        final startOffset = Offset(
          (1.0 - start['x']) * size.width,
          start['y'] * size.height,
        );
        final endOffset = Offset(
          (1.0 - end['x']) * size.width,
          end['y'] * size.height,
        );
        canvas.drawLine(startOffset, endOffset, paintLine);
      }
    }

    // Draw joint points
    for (var i = 11; i <= 28; i++) {
      // Only draw body points, skip face for now
      final lm = landmarks[i];
      if (lm['visibility'] > 0.5) {
        final offset = Offset(
          (1.0 - lm['x']) * size.width,
          lm['y'] * size.height,
        );
        // FIX: Use drawCircle instead of drawPoint.
        // The 4.0 is the radius of the joint node.
        canvas.drawCircle(offset, 4.0, paintPoint);
      }
    }
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => true;
}

class Biomechanics {
  // Calculates the 2D angle between three points (A, B, C) where B is the vertex (e.g., Knee)
  static double calculateAngle(
    Map<String, dynamic> a,
    Map<String, dynamic> b,
    Map<String, dynamic> c,
  ) {
    // We only need X and Y for a 2D angle projection
    double ax = a['x'];
    double ay = a['y'];
    double bx = b['x'];
    double by = b['y'];
    double cx = c['x'];
    double cy = c['y'];

    // Vector 1 (B to A)
    double v1x = ax - bx;
    double v1y = ay - by;

    // Vector 2 (B to C)
    double v2x = cx - bx;
    double v2y = cy - by;

    // Dot product and magnitudes
    double dotProduct = (v1x * v2x) + (v1y * v2y);
    double magnitude1 = math.sqrt((v1x * v1x) + (v1y * v1y));
    double magnitude2 = math.sqrt((v2x * v2x) + (v2y * v2y));

    if (magnitude1 == 0 || magnitude2 == 0) return 0.0;

    // Calculate angle in radians, then convert to degrees
    double angleRad = math.acos(dotProduct / (magnitude1 * magnitude2));
    double angleDeg = angleRad * (180.0 / math.pi);

    return angleDeg;
  }
}
