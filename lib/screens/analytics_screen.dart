/*
 * Analytics Screen
 * Displays driving analytics, including total drive time, alerts, average EAR value, and recent sessions.
 * Fetches data from the AnalyticsService and allows refreshing the data.
 */
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import '../theme.dart';
import '../services/analytics_service.dart';

class AnalyticsScreen extends StatefulWidget {
  const AnalyticsScreen({super.key});

  @override
  State<AnalyticsScreen> createState() => _AnalyticsScreenState();
}

class _AnalyticsScreenState extends State<AnalyticsScreen> {
  Map<String, dynamic> _summary = {
    'totalSeconds': 0,
    'totalAlerts': 0,
    'avgEar': 0.0,
    'totalSessions': 0,
  };
  List<DriveSession> _sessions = [];
  bool _loading = true;

  @override
  void initState() {
    super.initState();
    _loadData();
  }

  @override
  void didChangeDependencies() {
    super.didChangeDependencies();
    _loadData();
  }

  Future<void> _loadData() async {
    final summary = await AnalyticsService.loadSummary();
    final sessions = await AnalyticsService.loadSessions();
    if (mounted) {
      setState(() {
        _summary = summary;
        _sessions = sessions;
        _loading = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppColors.bg(context),
      body: SafeArea(
        child: RefreshIndicator(
          onRefresh: _loadData,
          color: AppColors.primary,
          child: SingleChildScrollView(
            physics: const AlwaysScrollableScrollPhysics(),
            padding: const EdgeInsets.all(16),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  'Analytics',
                  style: GoogleFonts.inter(
                    fontSize: 28,
                    fontWeight: FontWeight.bold,
                    color: AppColors.text(context),
                  ),
                ),
                Text(
                  'Your driving safety insights',
                  style: GoogleFonts.inter(
                    fontSize: 14,
                    color: AppColors.subText(context),
                  ),
                ),

                const SizedBox(height: 20),

                if (_loading)
                  const Center(child: CircularProgressIndicator())
                else ...[
                  Row(
                    children: [
                      _statCard(context,
                        icon: Icons.timer_outlined,
                        label: 'Total Drive Time',
                        value: AnalyticsService.formatTotalTime(
                            _summary['totalSeconds']),
                        sub: '${_summary['totalSessions']} sessions',
                        subColor: AppColors.green,
                        iconColor: AppColors.primary,
                      ),
                      const SizedBox(width: 12),
                      _statCard(context,
                        icon: Icons.warning_amber_outlined,
                        label: 'Total Alerts',
                        value: '${_summary['totalAlerts']}',
                        sub: 'All time',
                        subColor: AppColors.subText(context),
                        iconColor: AppColors.red,
                      ),
                    ],
                  ),

                  const SizedBox(height: 12),

                  Row(
                    children: [
                      _statCard(context,
                        icon: Icons.remove_red_eye_outlined,
                        label: 'Avg EAR Value',
                        value: (_summary['avgEar'] as double)
                            .toStringAsFixed(2),
                        sub: (_summary['avgEar'] as double) >= 0.28
                            ? 'Within safe range'
                            : 'Below safe range',
                        subColor: (_summary['avgEar'] as double) >= 0.28
                            ? AppColors.green
                            : AppColors.red,
                        iconColor: AppColors.green,
                      ),
                      const SizedBox(width: 12),
                      _statCard(context,
                        icon: Icons.calendar_today_outlined,
                        label: 'Sessions',
                        value: '${_summary['totalSessions']}',
                        sub: 'All time',
                        subColor: AppColors.subText(context),
                        iconColor: AppColors.primary,
                      ),
                    ],
                  ),

                  const SizedBox(height: 24),

                  Text(
                    'Recent Sessions',
                    style: GoogleFonts.inter(
                      fontSize: 18,
                      fontWeight: FontWeight.bold,
                      color: AppColors.text(context),
                    ),
                  ),

                  const SizedBox(height: 12),

                  if (_sessions.isEmpty)
                    Center(
                      child: Padding(
                        padding: const EdgeInsets.all(32),
                        child: Column(
                          children: [
                            Icon(Icons.directions_car_outlined,
                                size: 48,
                                color: AppColors.subText(context)),
                            const SizedBox(height: 12),
                            Text(
                              'No sessions yet.\nStart monitoring to track your drives.',
                              textAlign: TextAlign.center,
                              style: GoogleFonts.inter(
                                fontSize: 14,
                                color: AppColors.subText(context),
                                height: 1.5,
                              ),
                            ),
                          ],
                        ),
                      ),
                    )
                  else
                    ...(_sessions
                        .take(10)
                        .map((s) => _sessionCard(context,
                              date: s.date,
                              duration: s.durationFormatted,
                              alerts: s.alerts,
                              avgEar: s.avgEar,
                            ))),
                ],
              ],
            ),
          ),
        ),
      ),
    );
  }

  Widget _statCard(BuildContext context, {
    required IconData icon,
    required String label,
    required String value,
    required String sub,
    required Color subColor,
    required Color iconColor,
  }) {
    return Expanded(
      child: Container(
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
            Icon(icon, color: iconColor, size: 22),
            const SizedBox(height: 10),
            Text(label,
                style: GoogleFonts.inter(
                    fontSize: 11, color: AppColors.subText(context))),
            const SizedBox(height: 4),
            Text(value,
                style: GoogleFonts.inter(
                    fontSize: 24,
                    fontWeight: FontWeight.bold,
                    color: AppColors.text(context))),
            const SizedBox(height: 4),
            Text(sub,
                style: GoogleFonts.inter(
                    fontSize: 11,
                    color: subColor,
                    fontWeight: FontWeight.w500)),
          ],
        ),
      ),
    );
  }

  Widget _sessionCard(BuildContext context, {
    required String date,
    required String duration,
    required int alerts,
    required double avgEar,
  }) {
    return Container(
      margin: const EdgeInsets.only(bottom: 12),
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
              Text(date,
                  style: GoogleFonts.inter(
                      fontSize: 15,
                      fontWeight: FontWeight.w600,
                      color: AppColors.text(context))),
              Container(
                padding:
                    const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
                decoration: BoxDecoration(
                  color: AppColors.green.withOpacity(0.12),
                  borderRadius: BorderRadius.circular(20),
                ),
                child: Text('Completed',
                    style: GoogleFonts.inter(
                        fontSize: 11,
                        fontWeight: FontWeight.w600,
                        color: AppColors.green)),
              ),
            ],
          ),
          const SizedBox(height: 4),
          Text(duration,
              style: GoogleFonts.inter(
                  fontSize: 13, color: AppColors.subText(context))),
          const SizedBox(height: 12),
          Row(
            children: [
              Icon(Icons.warning_amber_outlined,
                  color: AppColors.primary, size: 16),
              const SizedBox(width: 4),
              Text('Alerts  $alerts',
                  style: GoogleFonts.inter(
                      fontSize: 13,
                      color: AppColors.text(context),
                      fontWeight: FontWeight.w500)),
              const SizedBox(width: 20),
              Icon(Icons.remove_red_eye_outlined,
                  color: AppColors.green, size: 16),
              const SizedBox(width: 4),
              Text('Avg EAR  ${avgEar.toStringAsFixed(2)}',
                  style: GoogleFonts.inter(
                      fontSize: 13,
                      color: AppColors.text(context),
                      fontWeight: FontWeight.w500)),
            ],
          ),
        ],
      ),
    );
  }
}