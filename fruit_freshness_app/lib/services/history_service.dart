import 'dart:convert';
import 'package:shared_preferences/shared_preferences.dart';

class HistoryItem {
  final String imagePath;
  final String result;
  final double confidence;
  final DateTime timestamp;

  HistoryItem({
    required this.imagePath,
    required this.result,
    required this.confidence,
    required this.timestamp,
  });

  Map<String, dynamic> toJson() => {
        'imagePath': imagePath,
        'result': result,
        'confidence': confidence,
        'timestamp': timestamp.toIso8601String(),
      };

  factory HistoryItem.fromJson(Map<String, dynamic> json) => HistoryItem(
        imagePath: json['imagePath'] as String,
        result: json['result'] as String,
        confidence: (json['confidence'] as num).toDouble(),
        timestamp: DateTime.parse(json['timestamp'] as String),
      );
}

class HistoryService {
  static const String _key = 'prediction_history';

  static Future<List<HistoryItem>> getHistory() async {
    final prefs = await SharedPreferences.getInstance();
    final data = prefs.getStringList(_key) ?? [];
    return data
        .map((e) => HistoryItem.fromJson(jsonDecode(e) as Map<String, dynamic>))
        .toList()
      ..sort((a, b) => b.timestamp.compareTo(a.timestamp));
  }

  static Future<void> addHistory(HistoryItem item) async {
    final prefs = await SharedPreferences.getInstance();
    final data = prefs.getStringList(_key) ?? [];
    data.add(jsonEncode(item.toJson()));
    await prefs.setStringList(_key, data);
  }

  static Future<void> clearHistory() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.remove(_key);
  }
}
