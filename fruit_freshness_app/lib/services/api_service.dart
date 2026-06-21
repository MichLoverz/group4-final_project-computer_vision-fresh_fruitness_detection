import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';
import 'package:http/http.dart' as http;

/// Response dari API predict
class PredictResult {
  final bool hasFruit;
  final String? result;
  final double confidence;
  final String? message;
  final Uint8List? preprocessedImage;
  final Uint8List? keypointsImage;

  PredictResult({
    required this.hasFruit,
    this.result,
    this.confidence = 0,
    this.message,
    this.preprocessedImage,
    this.keypointsImage,
  });
}

/// Service untuk komunikasi dengan backend FastAPI
class ApiService {
  // URL backend
  // Lokal (HP fisik via WiFi yang sama): pakai IP PC
  // Production (setelah deploy ke Render): ganti ke URL Render
  static const String baseUrl = 'http://192.168.0.107:8000';
  static String _activeUrl = baseUrl;

  /// Set URL backend (untuk switch antara lokal/production)
  static void setBaseUrl(String url) {
    _activeUrl = url;
  }

  /// Cek apakah backend reachable
  static Future<bool> healthCheck() async {
    try {
      final resp = await http.get(
        Uri.parse('$_activeUrl/health'),
      ).timeout(const Duration(seconds: 5));
      return resp.statusCode == 200;
    } catch (_) {
      return false;
    }
  }

  /// Kirim foto ke backend, terima hasil prediksi
  static Future<PredictResult> predict(File imageFile) async {
    try {
      final request = http.MultipartRequest(
        'POST',
        Uri.parse('$_activeUrl/predict'),
      );
      request.files.add(
        await http.MultipartFile.fromPath('file', imageFile.path),
      );

      final streamedResponse = await request.send()
          .timeout(const Duration(seconds: 30));
      final response = await http.Response.fromStream(streamedResponse);

      if (response.statusCode != 200) {
        final data = jsonDecode(response.body);
        throw Exception(data['error'] ?? 'Server error ${response.statusCode}');
      }

      final data = jsonDecode(response.body);

      Uint8List? preprocessedBytes;
      if (data['preprocessed_image'] != null) {
        preprocessedBytes = base64Decode(data['preprocessed_image']);
      }

      Uint8List? keypointsBytes;
      if (data['keypoints_image'] != null) {
        keypointsBytes = base64Decode(data['keypoints_image']);
      }

      return PredictResult(
        hasFruit: data['has_fruit'] ?? false,
        result: data['result'],
        confidence: (data['confidence'] ?? 0).toDouble(),
        message: data['message'],
        preprocessedImage: preprocessedBytes,
        keypointsImage: keypointsBytes,
      );
    } on SocketException {
      throw Exception('Tidak dapat terhubung ke server. Pastikan backend berjalan.');
    } catch (e) {
      if (e is Exception) rethrow;
      throw Exception('Error: $e');
    }
  }
}
