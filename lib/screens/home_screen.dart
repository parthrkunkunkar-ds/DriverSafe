/*
 * Home Screen
 */
import 'dart:async';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:camera/camera.dart';
import 'package:google_mlkit_face_detection/google_mlkit_face_detection.dart';
import '../theme.dart';
import '../services/detection_service.dart';
import '../services/cnn_service.dart';
import '../services/analytics_service.dart';

class HomeScreen extends StatefulWidget {
  final VoidCallback? onDrowsinessDetected;
  final VoidCallback? onSessionSaved;
  final double earThreshold;
  const HomeScreen({
    super.key,
    this.onDrowsinessDetected,
    this.onSessionSaved,
    this.earThreshold = 0.20,
  });

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  CameraController? _cameraController;
  bool _isCameraInitialized = false;
  bool _isMonitoring = false;
  double _earValue = 0.320;
  double _cnnScore = -1.0;
  bool _cnnExpanded = false;
  int _fps = 0;

  final DetectionService _detectionService = DetectionService();
  final CnnService _cnnService = CnnService();
  bool _faceDetected = false;
  int _drowsyFrameCount = 0;
  int _noFaceFrameCount = 0;
  static const int _drowsyFrameThreshold = 20;
  static const int _noFaceFrameThreshold = 20;
  double get _earThreshold => widget.earThreshold;

  int _frameCount = 0;
  DateTime _lastFpsTime = DateTime.now();

  DateTime? _sessionStart;
  int _sessionAlerts = 0;
  double _earSum = 0.0;
  int _earCount = 0;

  Future<void> _startMonitoring() async {
    final cameras = await availableCameras();
    final front = cameras.firstWhere(
      (c) => c.lensDirection == CameraLensDirection.front,
      orElse: () => cameras.first,
    );
    _cameraController = CameraController(
      front,
      ResolutionPreset.medium,
      enableAudio: false,
      imageFormatGroup: ImageFormatGroup.nv21,
    );
    await _cameraController!.initialize();
    await _cnnService.init();
    _cameraController!.startImageStream(_onCameraFrame);

    if (mounted) {
      setState(() {
        _isCameraInitialized = true;
        _isMonitoring = true;
        _drowsyFrameCount = 0;
        _noFaceFrameCount = 0;
        _frameCount = 0;
        _lastFpsTime = DateTime.now();
        _sessionStart = DateTime.now();
        _sessionAlerts = 0;
        _earSum = 0.0;
        _earCount = 0;
      });
    }
  }

  void _onCameraFrame(CameraImage image) async {
    _frameCount++;
    final now = DateTime.now();
    final elapsed = now.difference(_lastFpsTime).inMilliseconds;
    if (elapsed >= 1000) {
      if (mounted) {
        setState(() {
          _fps = (_frameCount * 1000 / elapsed).round();
        });
      }
      _frameCount = 0;
      _lastFpsTime = now;
    }

    final yBytes = Uint8List.fromList(image.planes[0].bytes);
    final uBytes = Uint8List.fromList(image.planes[1].bytes);
    final vBytes = Uint8List.fromList(image.planes[2].bytes);
    final yRowStride = image.planes[0].bytesPerRow;
    final uvRowStride = image.planes[1].bytesPerRow;
    final uvPixelStride = image.planes[1].bytesPerPixel ?? 1;
    final imgWidth = image.width;
    final imgHeight = image.height;

    final rotation = _getRotation();
    final result = await _detectionService.processFrame(image, rotation);
    if (result == null || !mounted) return;

    if (result.faceDetected && result.faceBox != null) {
      _cnnService.submitFrame(
        yBytes: yBytes,
        uBytes: uBytes,
        vBytes: vBytes,
        yRowStride: yRowStride,
        uvRowStride: uvRowStride,
        uvPixelStride: uvPixelStride,
        width: imgWidth,
        height: imgHeight,
        faceLeft: result.faceBox!.left,
        faceTop: result.faceBox!.top,
        faceRight: result.faceBox!.right,
        faceBottom: result.faceBox!.bottom,
        leftEyeLeft: result.leftEyeBox?.left,
        leftEyeTop: result.leftEyeBox?.top,
        leftEyeRight: result.leftEyeBox?.right,
        leftEyeBottom: result.leftEyeBox?.bottom,
        rightEyeLeft: result.rightEyeBox?.left,
        rightEyeTop: result.rightEyeBox?.top,
        rightEyeRight: result.rightEyeBox?.right,
        rightEyeBottom: result.rightEyeBox?.bottom,
      );
    }

    setState(() {
      if (result.faceDetected) {
        _faceDetected = true;
        _noFaceFrameCount = 0;
        _cnnScore = _cnnService.lastScore;
        if (result.ear > 0) {
          _earValue = result.ear;
          _earSum += result.ear;
          _earCount++;
          if (_earValue < _earThreshold) {
            _drowsyFrameCount++;
          } else {
            _drowsyFrameCount = 0;
          }
        }
      } else {
        _noFaceFrameCount++;
        if (_noFaceFrameCount >= _noFaceFrameThreshold) {
          _faceDetected = false;
          _drowsyFrameCount = 0;
          _cnnScore = -1.0;
        }
      }
    });

    if (_drowsyFrameCount >= _drowsyFrameThreshold && mounted) {
      _drowsyFrameCount = 0;
      _triggerAlert();
    }
  }

  InputImageRotation _getRotation() {
    return InputImageRotation.rotation0deg;
  }

  void _triggerAlert() {
    _sessionAlerts++;
    AnalyticsService.recordAlarmEvent();
    widget.onDrowsinessDetected?.call();
  }

  Future<void> _saveSession() async {
    if (_sessionStart == null) return;

    final duration = DateTime.now().difference(_sessionStart!).inSeconds;
    final avgEar = _earCount > 0 ? _earSum / _earCount : 0.0;

    final now = DateTime.now();
    const months = [
      'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'
    ];
    final date = '${months[now.month - 1]} ${now.day}, ${now.year}';

    await AnalyticsService.saveSession(DriveSession(
      date: date,
      durationSeconds: duration,
      alerts: _sessionAlerts,
      avgEar: double.parse(avgEar.toStringAsFixed(2)),
    ));

    _sessionStart = null;
    widget.onSessionSaved?.call();
  }

  void _stopMonitoring() async {
    await _saveSession();
    _cameraController?.stopImageStream();
    _cameraController?.dispose();
    _cameraController = null;
    setState(() {
      _isCameraInitialized = false;
      _isMonitoring = false;
      _earValue = 0.320;
      _cnnScore = -1.0;
      _fps = 0;
      _drowsyFrameCount = 0;
      _noFaceFrameCount = 0;
      _faceDetected = false;
    });
  }

  @override
  void dispose() {
    _cameraController?.stopImageStream();
    _cameraController?.dispose();
    _detectionService.dispose();
    _cnnService.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppColors.bg(context),
      body: SafeArea(
        child: SingleChildScrollView(
          padding: const EdgeInsets.all(16),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        'SurakshaDrive',
                        style: GoogleFonts.inter(
                          fontSize: 22,
                          fontWeight: FontWeight.bold,
                          color: AppColors.text(context),
                        ),
                      ),
                      Text(
                        _isMonitoring
                            ? (_faceDetected
                                ? 'Face detected'
                                : 'Searching for face...')
                            : 'Ready to monitor',
                        style: GoogleFonts.inter(
                          fontSize: 13,
                          color: AppColors.subText(context),
                        ),
                      ),
                    ],
                  ),
                  Container(
                    padding: const EdgeInsets.symmetric(
                        horizontal: 12, vertical: 6),
                    decoration: BoxDecoration(
                      color: AppColors.green.withOpacity(0.12),
                      borderRadius: BorderRadius.circular(20),
                    ),
                    child: Text(
                      'ON-DEVICE',
                      style: GoogleFonts.inter(
                        fontSize: 11,
                        fontWeight: FontWeight.w600,
                        color: AppColors.green,
                      ),
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 20),
              Container(
                width: double.infinity,
                decoration: BoxDecoration(
                  color: Colors.black,
                  borderRadius: BorderRadius.circular(20),
                  border: _isMonitoring
                      ? Border.all(
                          color: _drowsyFrameCount > 10
                              ? AppColors.red.withOpacity(0.8)
                              : AppColors.primary.withOpacity(0.5),
                          width: 2,
                        )
                      : null,
                  boxShadow: [
                    BoxShadow(
                      color: Colors.black.withOpacity(0.06),
                      blurRadius: 20,
                      offset: const Offset(0, 4),
                    ),
                  ],
                ),
                child: ClipRRect(
                  borderRadius: BorderRadius.circular(20),
                  child: AspectRatio(
                    aspectRatio: 3 / 4,
                    child: _isCameraInitialized
                        ? Stack(
                            fit: StackFit.expand,
                            children: [
                              CameraPreview(_cameraController!),
                              Positioned(
                                top: 12,
                                left: 12,
                                child: Container(
                                  padding: const EdgeInsets.symmetric(
                                      horizontal: 8, vertical: 4),
                                  decoration: BoxDecoration(
                                    color: Colors.black54,
                                    borderRadius: BorderRadius.circular(8),
                                  ),
                                  child: Row(
                                    mainAxisSize: MainAxisSize.min,
                                    children: [
                                      Container(
                                        width: 6,
                                        height: 6,
                                        decoration: const BoxDecoration(
                                          color: Color(0xFFFF453A),
                                          shape: BoxShape.circle,
                                        ),
                                      ),
                                      const SizedBox(width: 4),
                                      Text(
                                        'LIVE',
                                        style: GoogleFonts.inter(
                                          color: Colors.white,
                                          fontSize: 11,
                                          fontWeight: FontWeight.w600,
                                        ),
                                      ),
                                    ],
                                  ),
                                ),
                              ),
                              Positioned(
                                top: 12,
                                right: 12,
                                child: Container(
                                  padding: const EdgeInsets.symmetric(
                                      horizontal: 8, vertical: 4),
                                  decoration: BoxDecoration(
                                    color: Colors.black54,
                                    borderRadius: BorderRadius.circular(8),
                                  ),
                                  child: Text(
                                    '$_fps FPS',
                                    style: GoogleFonts.inter(
                                      color: Colors.white,
                                      fontSize: 11,
                                      fontWeight: FontWeight.w600,
                                    ),
                                  ),
                                ),
                              ),
                              if (_isMonitoring)
                                Positioned(
                                  bottom: 50,
                                  left: 12,
                                  child: Container(
                                    padding: const EdgeInsets.symmetric(
                                        horizontal: 8, vertical: 4),
                                    decoration: BoxDecoration(
                                      color: Colors.black54,
                                      borderRadius: BorderRadius.circular(8),
                                    ),
                                    child: Row(
                                      mainAxisSize: MainAxisSize.min,
                                      children: [
                                        Container(
                                          width: 6,
                                          height: 6,
                                          decoration: BoxDecoration(
                                            color: _faceDetected
                                                ? AppColors.green
                                                : AppColors.subText(context),
                                            shape: BoxShape.circle,
                                          ),
                                        ),
                                        const SizedBox(width: 4),
                                        Text(
                                          _faceDetected ? 'FACE' : 'NO FACE',
                                          style: GoogleFonts.inter(
                                            color: Colors.white,
                                            fontSize: 11,
                                            fontWeight: FontWeight.w600,
                                          ),
                                        ),
                                      ],
                                    ),
                                  ),
                                ),
                              Positioned(
                                bottom: 12,
                                right: 12,
                                child: GestureDetector(
                                  onTap: _stopMonitoring,
                                  child: Container(
                                    padding: const EdgeInsets.symmetric(
                                        horizontal: 10, vertical: 6),
                                    decoration: BoxDecoration(
                                      color: AppColors.red,
                                      borderRadius: BorderRadius.circular(8),
                                    ),
                                    child: Row(
                                      mainAxisSize: MainAxisSize.min,
                                      children: [
                                        const Icon(Icons.stop,
                                            color: Colors.white, size: 14),
                                        const SizedBox(width: 4),
                                        Text(
                                          'Stop',
                                          style: GoogleFonts.inter(
                                            color: Colors.white,
                                            fontSize: 11,
                                            fontWeight: FontWeight.w600,
                                          ),
                                        ),
                                      ],
                                    ),
                                  ),
                                ),
                              ),
                            ],
                          )
                        : GestureDetector(
                            onTap: _startMonitoring,
                            child: Column(
                              mainAxisAlignment: MainAxisAlignment.center,
                              children: [
                                Container(
                                  width: 72,
                                  height: 72,
                                  decoration: BoxDecoration(
                                    color:
                                        AppColors.primary.withOpacity(0.15),
                                    shape: BoxShape.circle,
                                  ),
                                  child: Icon(
                                    Icons.play_arrow_rounded,
                                    color: AppColors.primary,
                                    size: 40,
                                  ),
                                ),
                                const SizedBox(height: 16),
                                Text(
                                  'Tap to start monitoring',
                                  style: GoogleFonts.inter(
                                    color: AppColors.subText(context),
                                    fontSize: 14,
                                  ),
                                ),
                              ],
                            ),
                          ),
                  ),
                ),
              ),
              const SizedBox(height: 16),
              Container(
                padding: const EdgeInsets.all(16),
                decoration: BoxDecoration(
                  color: AppColors.card(context),
                  borderRadius: BorderRadius.circular(16),
                  boxShadow: [
                    BoxShadow(
                      color: Colors.black.withOpacity(0.05),
                      blurRadius: 10,
                      offset: const Offset(0, 2),
                    ),
                  ],
                ),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(
                      mainAxisAlignment: MainAxisAlignment.spaceBetween,
                      children: [
                        Text(
                          'EYE ASPECT RATIO',
                          style: GoogleFonts.inter(
                            fontSize: 11,
                            fontWeight: FontWeight.w600,
                            color: AppColors.subText(context),
                            letterSpacing: 0.5,
                          ),
                        ),
                        Text(
                          _earValue.toStringAsFixed(3),
                          style: GoogleFonts.inter(
                            fontSize: 28,
                            fontWeight: FontWeight.bold,
                            color: _isMonitoring && _faceDetected
                                ? (_earValue < _earThreshold
                                    ? AppColors.red
                                    : AppColors.green)
                                : AppColors.text(context),
                          ),
                        ),
                      ],
                    ),
                    const SizedBox(height: 10),
                    ClipRRect(
                      borderRadius: BorderRadius.circular(4),
                      child: LinearProgressIndicator(
                        value: (_earValue / 0.4).clamp(0.0, 1.0),
                        backgroundColor: AppColors.divider(context),
                        valueColor: AlwaysStoppedAnimation<Color>(
                          _isMonitoring && _faceDetected
                              ? (_earValue < _earThreshold
                                  ? AppColors.red
                                  : AppColors.green)
                              : AppColors.green,
                        ),
                        minHeight: 6,
                      ),
                    ),
                    const SizedBox(height: 8),
                    Row(
                      mainAxisAlignment: MainAxisAlignment.spaceBetween,
                      children: [
                        Text(
                          'Normal range: 0.28 - 0.40 • Lower = drowsy',
                          style: GoogleFonts.inter(
                            fontSize: 12,
                            color: AppColors.subText(context),
                          ),
                        ),
                        if (_isMonitoring && _drowsyFrameCount > 0)
                          Text(
                            '${_drowsyFrameCount}/$_drowsyFrameThreshold',
                            style: GoogleFonts.inter(
                              fontSize: 12,
                              fontWeight: FontWeight.w600,
                              color: AppColors.red,
                            ),
                          ),
                      ],
                    ),
                    if (_isMonitoring && _faceDetected && _cnnScore >= 0) ...[
                      Divider(color: AppColors.divider(context), height: 24),
                      GestureDetector(
                        onTap: () =>
                            setState(() => _cnnExpanded = !_cnnExpanded),
                        child: Row(
                          mainAxisAlignment: MainAxisAlignment.spaceBetween,
                          children: [
                            Row(
                              children: [
                                Text(
                                  'CNN MODEL',
                                  style: GoogleFonts.inter(
                                    fontSize: 11,
                                    fontWeight: FontWeight.w600,
                                    color: AppColors.subText(context),
                                    letterSpacing: 0.5,
                                  ),
                                ),
                                const SizedBox(width: 4),
                                Container(
                                  padding: const EdgeInsets.symmetric(
                                      horizontal: 5, vertical: 2),
                                  decoration: BoxDecoration(
                                    color:
                                        AppColors.primary.withOpacity(0.12),
                                    borderRadius: BorderRadius.circular(4),
                                  ),
                                  child: Text(
                                    'Experimental',
                                    style: GoogleFonts.inter(
                                      fontSize: 9,
                                      fontWeight: FontWeight.w600,
                                      color: AppColors.primary,
                                    ),
                                  ),
                                ),
                              ],
                            ),
                            Row(
                              children: [
                                Text(
                                  _cnnScore > 0.5 ? 'OPEN' : 'CLOSED',
                                  style: GoogleFonts.inter(
                                    fontSize: 12,
                                    fontWeight: FontWeight.w600,
                                    color: _cnnScore > 0.5
                                        ? AppColors.green
                                        : AppColors.red,
                                  ),
                                ),
                                const SizedBox(width: 6),
                                Text(
                                  '(${(_cnnScore * 100).toStringAsFixed(0)}%)',
                                  style: GoogleFonts.inter(
                                    fontSize: 12,
                                    color: AppColors.subText(context),
                                  ),
                                ),
                                const SizedBox(width: 4),
                                Icon(
                                  _cnnExpanded
                                      ? Icons.keyboard_arrow_up
                                      : Icons.keyboard_arrow_down,
                                  size: 18,
                                  color: AppColors.subText(context),
                                ),
                              ],
                            ),
                          ],
                        ),
                      ),
                      if (_cnnExpanded) ...[
                        const SizedBox(height: 8),
                        ClipRRect(
                          borderRadius: BorderRadius.circular(4),
                          child: LinearProgressIndicator(
                            value: _cnnScore.clamp(0.0, 1.0),
                            backgroundColor: AppColors.divider(context),
                            valueColor: AlwaysStoppedAnimation<Color>(
                              _cnnScore > 0.5
                                  ? AppColors.green
                                  : AppColors.red,
                            ),
                            minHeight: 4,
                          ),
                        ),
                        const SizedBox(height: 6),
                        Text(
                          'Secondary signal • EAR drives detection • Results may vary',
                          style: GoogleFonts.inter(
                            fontSize: 11,
                            color: AppColors.subText(context),
                          ),
                        ),
                      ],
                    ],
                  ],
                ),
              ),
              const SizedBox(height: 16),
              SizedBox(
                width: double.infinity,
                height: 56,
                child: ElevatedButton.icon(
                  onPressed: () {
                    ScaffoldMessenger.of(context).showSnackBar(
                      SnackBar(
                        content: Text(
                          'Google Maps integration coming soon!',
                          style: GoogleFonts.inter(),
                        ),
                        backgroundColor: const Color(0xFF1C1C1E),
                        behavior: SnackBarBehavior.floating,
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(12),
                        ),
                      ),
                    );
                  },
                  icon: const Icon(Icons.map_outlined),
                  label: Text(
                    'Connect with Maps',
                    style: GoogleFonts.inter(
                      fontSize: 17,
                      fontWeight: FontWeight.w600,
                    ),
                  ),
                  style: ElevatedButton.styleFrom(
                    backgroundColor: const Color(0xFF1C1C1E),
                    foregroundColor: Colors.white,
                    elevation: 0,
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(16),
                    ),
                  ),
                ),
              ),
              const SizedBox(height: 12),
              Container(
                padding: const EdgeInsets.all(14),
                decoration: BoxDecoration(
                  color: AppColors.primary.withOpacity(0.08),
                  borderRadius: BorderRadius.circular(12),
                ),
                child: Row(
                  children: [
                    Icon(Icons.warning_amber_rounded,
                        color: AppColors.primary, size: 18),
                    const SizedBox(width: 10),
                    Expanded(
                      child: Text(
                        'If you feel drowsy, pull over immediately and rest',
                        style: GoogleFonts.inter(
                          fontSize: 13,
                          color: AppColors.primary,
                          fontWeight: FontWeight.w500,
                        ),
                      ),
                    ),
                  ],
                ),
              ),
              const SizedBox(height: 16),
            ],
          ),
        ),
      ),
    );
  }
}