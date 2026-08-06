/*
 * Analytics Service
 */
import 'dart:convert';
import 'package:flutter/foundation.dart';
import 'package:shared_preferences/shared_preferences.dart';

class DriveSession {
  final String date;
  final int durationSeconds;
  final int alerts;
  final double avgEar;

  DriveSession({
    required this.date,
    required this.durationSeconds,
    required this.alerts,
    required this.avgEar,
  });

  String get durationFormatted {
    final h = durationSeconds ~/ 3600;
    final m = (durationSeconds % 3600) ~/ 60;
    return '${h}h ${m.toString().padLeft(2, '0')}m';
  }

  Map<String, dynamic> toJson() => {
        'date': date,
        'durationSeconds': durationSeconds,
        'alerts': alerts,
        'avgEar': avgEar,
      };

  factory DriveSession.fromJson(Map<String, dynamic> json) => DriveSession(
        date: json['date'],
        durationSeconds: json['durationSeconds'],
        alerts: json['alerts'],
        avgEar: (json['avgEar'] as num).toDouble(),
      );
}

class AnalyticsService {
  static const _sessionsKey = 'drive_sessions';
  static const _alarmCountKey = 'total_alarm_count';

  static Future<void> saveSession(DriveSession session) async {
    try {
      final prefs = await SharedPreferences.getInstance();
      final sessions = await loadSessions();
      sessions.insert(0, session);
      final trimmed = sessions.take(50).toList();
      await prefs.setString(
        _sessionsKey,
        jsonEncode(trimmed.map((s) => s.toJson()).toList()),
      );
    } catch (e) {
      debugPrint('Analytics save error: $e');
    }
  }

  static Future<List<DriveSession>> loadSessions() async {
    try {
      final prefs = await SharedPreferences.getInstance();
      final raw = prefs.getString(_sessionsKey);
      if (raw == null || raw.isEmpty) return [];
      final list = jsonDecode(raw) as List;
      return list.map((e) => DriveSession.fromJson(e)).toList();
    } catch (e) {
      debugPrint('Analytics load error: $e');
      return [];
    }
  }

  static Future<void> recordAlarmEvent() async {
    try {
      final prefs = await SharedPreferences.getInstance();
      final current = prefs.getInt(_alarmCountKey) ?? 0;
      await prefs.setInt(_alarmCountKey, current + 1);
    } catch (e) {
      debugPrint('Alarm event save error: $e');
    }
  }

  static Future<int> getTotalAlarmCount() async {
    final prefs = await SharedPreferences.getInstance();
    return prefs.getInt(_alarmCountKey) ?? 0;
  }

  static Future<Map<String, dynamic>> loadSummary() async {
    final sessions = await loadSessions();
    final totalAlarmCount = await getTotalAlarmCount();

    if (sessions.isEmpty && totalAlarmCount == 0) {
      return {
        'totalSeconds': 0,
        'totalAlerts': 0,
        'avgEar': 0.0,
        'totalSessions': 0,
      };
    }

    final totalSeconds =
        sessions.fold<int>(0, (sum, s) => sum + s.durationSeconds);
    final totalAlerts = totalAlarmCount > 0
        ? totalAlarmCount
        : sessions.fold<int>(0, (sum, s) => sum + s.alerts);
    final avgEar = sessions.isEmpty
        ? 0.0
        : sessions.fold<double>(0, (sum, s) => sum + s.avgEar) /
            sessions.length;

    return {
      'totalSeconds': totalSeconds,
      'totalAlerts': totalAlerts,
      'avgEar': avgEar,
      'totalSessions': sessions.length,
    };
  }

  static String formatTotalTime(int seconds) {
    final h = seconds ~/ 3600;
    final m = (seconds % 3600) ~/ 60;
    return '${h}h ${m.toString().padLeft(2, '0')}m';
  }
}