/*
 * Main Screen
 */
import 'package:flutter/material.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'home_screen.dart';
import 'alert_screen.dart';
import 'analytics_screen.dart';
import 'settings_screen.dart';

class MainScreen extends StatefulWidget {
  const MainScreen({super.key});

  @override
  State<MainScreen> createState() => _MainScreenState();
}

class _MainScreenState extends State<MainScreen> {
  bool _alertActive = false;
  double _earSensitivity = 0.20;
  bool _alertSound = true;
  bool _vibration = true;
  int _analyticsKey = 0;
  int _currentIndex = 0;

  @override
  void initState() {
    super.initState();
    _loadSettings();
  }

  Future<void> _loadSettings() async {
    final prefs = await SharedPreferences.getInstance();
    setState(() {
      _earSensitivity = prefs.getDouble('earSensitivity') ?? 0.20;
      _alertSound = prefs.getBool('alertSound') ?? true;
      _vibration = prefs.getBool('vibration') ?? true;
    });
  }

  void _triggerAlert() {
    setState(() {
      _alertActive = true;
      _currentIndex = 1;
    });
  }

  void _dismissAlert() {
    setState(() {
      _alertActive = false;
    });
  }

  @override
  Widget build(BuildContext context) {
    final screens = [
      HomeScreen(
        onDrowsinessDetected: _triggerAlert,
        onSessionSaved: () => setState(() => _analyticsKey++),
        earThreshold: _earSensitivity,
      ),
      AlertScreen(
        isActive: _alertActive,
        onDismiss: _dismissAlert,
        soundEnabled: _alertSound,
        vibrationEnabled: _vibration,
      ),
      AnalyticsScreen(key: ValueKey(_analyticsKey)),
      SettingsScreen(onSettingsChanged: _loadSettings),
    ];

    return Scaffold(
      body: screens[_currentIndex],
      bottomNavigationBar: NavigationBar(
        selectedIndex: _currentIndex,
        onDestinationSelected: (i) {
          if (i != 1) _dismissAlert();
          if (i == 0) _loadSettings();
          setState(() => _currentIndex = i);
        },
        backgroundColor: Theme.of(context).colorScheme.surface,
        indicatorColor: const Color(0xFFFF9500).withOpacity(0.15),
        destinations: [
          NavigationDestination(
            icon: const Icon(Icons.home_outlined),
            selectedIcon: const Icon(Icons.home, color: Color(0xFFFF9500)),
            label: 'Home',
          ),
          NavigationDestination(
            icon: const Icon(Icons.warning_amber_outlined),
            selectedIcon:
                const Icon(Icons.warning_amber, color: Color(0xFFFF9500)),
            label: 'Alert',
          ),
          NavigationDestination(
            icon: const Icon(Icons.bar_chart_outlined),
            selectedIcon:
                const Icon(Icons.bar_chart, color: Color(0xFFFF9500)),
            label: 'Analytics',
          ),
          NavigationDestination(
            icon: const Icon(Icons.settings_outlined),
            selectedIcon:
                const Icon(Icons.settings, color: Color(0xFFFF9500)),
            label: 'Settings',
          ),
        ],
      ),
    );
  }
}