/*
 * Alert Screen
 * Displays an alert when drowsiness is detected, with options for sound and vibration.
 * Also shows nearby rest stops for the driver to pull over safely.
 */
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:vibration/vibration.dart';
import 'package:audioplayers/audioplayers.dart';
import '../theme.dart';

class AlertScreen extends StatefulWidget {
  final bool isActive;
  final VoidCallback? onDismiss;
  final bool soundEnabled;
  final bool vibrationEnabled;
  const AlertScreen({
    super.key,
    this.isActive = false,
    this.onDismiss,
    this.soundEnabled = true,
    this.vibrationEnabled = true,
  });

  @override
  State<AlertScreen> createState() => _AlertScreenState();
}

class _AlertScreenState extends State<AlertScreen>
    with SingleTickerProviderStateMixin {
  late AnimationController _pulseController;
  late Animation<double> _pulseAnim;
  final AudioPlayer _audioPlayer = AudioPlayer();
  bool _alertActive = false;

  final List<Map<String, dynamic>> _restStops = [
    {
      'name': 'Highway Dhaba',
      'distance': '2.3 km away',
      'icon': Icons.restaurant_outlined,
    },
    {
      'name': 'Petrol Pump Rest Area',
      'distance': '4.1 km away',
      'icon': Icons.local_gas_station_outlined,
    },
    {
      'name': 'Motel Paradise',
      'distance': '5.8 km away',
      'icon': Icons.hotel_outlined,
    },
  ];

  @override
  void initState() {
    super.initState();
    _pulseController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 800),
    )..repeat(reverse: true);
    _pulseAnim = Tween<double>(begin: 0.95, end: 1.05).animate(
      CurvedAnimation(parent: _pulseController, curve: Curves.easeInOut),
    );
    if (widget.isActive) _triggerAlert();
  }

  Future<void> _triggerAlert() async {
    setState(() => _alertActive = true);
   if (widget.vibrationEnabled) {
  final hasVibrator = await Vibration.hasVibrator() ?? false;
  if (hasVibrator) {
    Vibration.vibrate(pattern: [0, 500, 200, 500, 200, 1000], repeat: 0);
  }
}
if (widget.soundEnabled) {
      try {
        await _audioPlayer.setReleaseMode(ReleaseMode.loop);
        await _audioPlayer.play(AssetSource('audio/alarm.mp3'));
      } catch (e) {
        debugPrint('Audio error: $e');
      }
    }
  }

  Future<void> _dismissAlert() async {
    Vibration.cancel();
    await _audioPlayer.stop();
    setState(() => _alertActive = false);
    widget.onDismiss?.call();
  }

  @override
  void dispose() {
    _pulseController.dispose();
    _audioPlayer.dispose();
    Vibration.cancel();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: _alertActive
          ? AppColors.red
          : AppColors.bg(context),
      body: SafeArea(
        child: _alertActive ? _buildActiveAlert() : _buildIdleState(context),
      ),
    );
  }

  Widget _buildActiveAlert() {
    return Column(
      children: [
        Align(
          alignment: Alignment.topLeft,
          child: IconButton(
            icon: const Icon(Icons.arrow_back, color: Colors.white),
            onPressed: _dismissAlert,
          ),
        ),
        Expanded(
          child: SingleChildScrollView(
            padding: const EdgeInsets.symmetric(horizontal: 24),
            child: Column(
              children: [
                const SizedBox(height: 16),
                ScaleTransition(
                  scale: _pulseAnim,
                  child: Container(
                    width: 80,
                    height: 80,
                    decoration: BoxDecoration(
                      color: Colors.white.withOpacity(0.2),
                      shape: BoxShape.circle,
                    ),
                    child: const Icon(
                      Icons.warning_amber_rounded,
                      color: Colors.white,
                      size: 48,
                    ),
                  ),
                ),
                const SizedBox(height: 24),
                Text(
                  'जागो! रुको!',
                  style: GoogleFonts.inter(
                    fontSize: 42,
                    fontWeight: FontWeight.bold,
                    color: Colors.white,
                  ),
                ),
                const SizedBox(height: 8),
                Text(
                  'Wake Up! Pull Over!',
                  style: GoogleFonts.inter(
                    fontSize: 20,
                    fontWeight: FontWeight.w600,
                    color: Colors.white,
                  ),
                ),
                const SizedBox(height: 12),
                Text(
                  'Drowsiness detected. Find a safe place to stop and rest immediately.',
                  textAlign: TextAlign.center,
                  style: GoogleFonts.inter(
                    fontSize: 15,
                    color: Colors.white.withOpacity(0.85),
                    height: 1.5,
                  ),
                ),
                const SizedBox(height: 20),
                Container(
                  padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
                  decoration: BoxDecoration(
                    color: Colors.white.withOpacity(0.2),
                    borderRadius: BorderRadius.circular(20),
                  ),
                  child: Row(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      const Icon(Icons.vibration, color: Colors.white, size: 16),
                      const SizedBox(width: 6),
                      Text(
                        'Vibration active',
                        style: GoogleFonts.inter(
                          color: Colors.white,
                          fontSize: 13,
                          fontWeight: FontWeight.w500,
                        ),
                      ),
                    ],
                  ),
                ),
                const SizedBox(height: 28),
                Align(
                  alignment: Alignment.centerLeft,
                  child: Row(
                    children: [
                      const Icon(Icons.location_on_outlined,
                          color: Colors.white, size: 16),
                      const SizedBox(width: 6),
                      Text(
                        'Nearby Rest Stops',
                        style: GoogleFonts.inter(
                          color: Colors.white,
                          fontSize: 13,
                          fontWeight: FontWeight.w600,
                        ),
                      ),
                    ],
                  ),
                ),
                const SizedBox(height: 12),
                ...(_restStops.map((stop) => Container(
                      margin: const EdgeInsets.only(bottom: 10),
                      padding: const EdgeInsets.all(14),
                      decoration: BoxDecoration(
                        color: Colors.white.withOpacity(0.15),
                        borderRadius: BorderRadius.circular(14),
                      ),
                      child: Row(
                        children: [
                          Container(
                            width: 36,
                            height: 36,
                            decoration: BoxDecoration(
                              color: Colors.white.withOpacity(0.2),
                              shape: BoxShape.circle,
                            ),
                            child: Icon(
                              stop['icon'] as IconData,
                              color: Colors.white,
                              size: 18,
                            ),
                          ),
                          const SizedBox(width: 12),
                          Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              Text(
                                stop['name'] as String,
                                style: GoogleFonts.inter(
                                  color: Colors.white,
                                  fontSize: 14,
                                  fontWeight: FontWeight.w600,
                                ),
                              ),
                              Text(
                                stop['distance'] as String,
                                style: GoogleFonts.inter(
                                  color: Colors.white.withOpacity(0.7),
                                  fontSize: 12,
                                ),
                              ),
                            ],
                          ),
                        ],
                      ),
                    ))),
                const SizedBox(height: 20),
              ],
            ),
          ),
        ),
        Padding(
          padding: const EdgeInsets.all(16),
          child: SizedBox(
            width: double.infinity,
            height: 56,
            child: ElevatedButton(
              onPressed: _dismissAlert,
              style: ElevatedButton.styleFrom(
                backgroundColor: Colors.white,
                foregroundColor: AppColors.red,
                elevation: 0,
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(16),
                ),
              ),
              child: Text(
                "I'm Awake — Dismiss Alert",
                style: GoogleFonts.inter(
                  fontSize: 16,
                  fontWeight: FontWeight.bold,
                ),
              ),
            ),
          ),
        ),
      ],
    );
  }

  Widget _buildIdleState(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Padding(
          padding: const EdgeInsets.all(16),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                'Alert Center',
                style: GoogleFonts.inter(
                  fontSize: 28,
                  fontWeight: FontWeight.bold,
                  color: AppColors.text(context),
                ),
              ),
              Text(
                'No active alerts',
                style: GoogleFonts.inter(
                  fontSize: 14,
                  color: AppColors.subText(context),
                ),
              ),
            ],
          ),
        ),
        Expanded(
          child: Center(
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                Container(
                  width: 80,
                  height: 80,
                  decoration: BoxDecoration(
                    color: AppColors.green.withOpacity(0.12),
                    shape: BoxShape.circle,
                  ),
                  child: const Icon(
                    Icons.check_circle_outline,
                    color: AppColors.green,
                    size: 40,
                  ),
                ),
                const SizedBox(height: 16),
                Text(
                  'You\'re doing great!',
                  style: GoogleFonts.inter(
                    fontSize: 18,
                    fontWeight: FontWeight.w600,
                    color: AppColors.text(context),
                  ),
                ),
                const SizedBox(height: 8),
                Text(
                  'No drowsiness detected.\nKeep driving safely.',
                  textAlign: TextAlign.center,
                  style: GoogleFonts.inter(
                    fontSize: 14,
                    color: AppColors.subText(context),
                    height: 1.5,
                  ),
                ),
                const SizedBox(height: 32),
                ElevatedButton.icon(
                  onPressed: _triggerAlert,
                  icon: const Icon(Icons.warning_amber_rounded),
                  label: Text(
                    'Test Alert',
                    style: GoogleFonts.inter(fontWeight: FontWeight.w600),
                  ),
                  style: ElevatedButton.styleFrom(
                    backgroundColor: AppColors.red,
                    foregroundColor: Colors.white,
                    elevation: 0,
                    padding: const EdgeInsets.symmetric(
                        horizontal: 24, vertical: 14),
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(14),
                    ),
                  ),
                ),
              ],
            ),
          ),
        ),
      ],
    );
  }
}