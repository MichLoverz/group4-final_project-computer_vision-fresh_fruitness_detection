import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:image_cropper/image_cropper.dart';
import '../services/api_service.dart';
import '../services/supabase_service.dart';
import '../services/history_service.dart';
import 'history_screen.dart';
import 'login_screen.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  final ImagePicker _picker = ImagePicker();

  File? _selectedImage;
  String? _result;
  double? _confidence;
  bool _isProcessing = false;
  String? _errorMessage;

  // Visualizations from backend
  Uint8List? _preprocessedImageBytes;
  Uint8List? _keypointsImageBytes;

  @override
  void initState() {
    super.initState();
    _checkBackend();
  }

  Future<void> _checkBackend() async {
    final ok = await ApiService.healthCheck();
    if (!ok && mounted) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text('Backend tidak terhubung. Pastikan server berjalan.'),
          backgroundColor: Colors.orange,
          duration: Duration(seconds: 5),
        ),
      );
    }
  }

  Future<void> _pickImage(ImageSource source) async {
    final XFile? image = await _picker.pickImage(
      source: source,
      maxWidth: 1024,
      maxHeight: 1024,
      imageQuality: 85,
    );

    if (image == null) return;

    final croppedFile = await ImageCropper().cropImage(
      sourcePath: image.path,
      uiSettings: [
        AndroidUiSettings(
          toolbarTitle: 'Crop Buah',
          toolbarColor: Colors.green.shade700,
          toolbarWidgetColor: Colors.white,
          activeControlsWidgetColor: Colors.green.shade700,
          initAspectRatio: CropAspectRatioPreset.square,
          lockAspectRatio: false,
        ),
      ],
    );

    if (croppedFile == null) return;

    setState(() {
      _selectedImage = File(croppedFile.path);
      _result = null;
      _confidence = null;
      _errorMessage = null;
      _preprocessedImageBytes = null;
      _keypointsImageBytes = null;
    });

    await _processImage();
  }

  Future<void> _processImage() async {
    if (_selectedImage == null) return;

    setState(() {
      _isProcessing = true;
      _errorMessage = null;
    });

    try {
      final predictResult = await ApiService.predict(_selectedImage!);

      if (!predictResult.hasFruit) {
        // Buah tidak terdeteksi
        setState(() {
          _isProcessing = false;
          _result = null;
          _confidence = null;
        });

        if (mounted) {
          showDialog(
            context: context,
            builder: (context) => AlertDialog(
              shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
              title: const Row(
                children: [
                  Icon(Icons.info_outline, color: Colors.orange, size: 28),
                  SizedBox(width: 8),
                  Text('Buah Tidak Terdeteksi'),
                ],
              ),
              content: const Text(
                'Tidak dapat mendeteksi buah pada foto ini.\n\n'
                'Tips:\n'
                '• Pastikan buah terlihat jelas di dalam frame\n'
                '• Ambil foto close-up dengan pencahayaan cukup\n'
                '• Hindari background yang terlalu ramai\n'
                '• Coba crop bagian buah saja',
              ),
              actions: [
                ElevatedButton(
                  onPressed: () => Navigator.pop(context),
                  style: ElevatedButton.styleFrom(
                    backgroundColor: Colors.green.shade700,
                    foregroundColor: Colors.white,
                  ),
                  child: const Text('OK, Coba Lagi'),
                ),
              ],
            ),
          );
        }
        return;
      }

      setState(() {
        _result = predictResult.result;
        _confidence = predictResult.confidence;
        _preprocessedImageBytes = predictResult.preprocessedImage;
        _keypointsImageBytes = predictResult.keypointsImage;
        _isProcessing = false;
      });

      if (mounted) {
        await HistoryService.addHistory(HistoryItem(
          imagePath: _selectedImage!.path,
          result: predictResult.result ?? 'unknown',
          confidence: predictResult.confidence,
          timestamp: DateTime.now(),
        ));

        if (SupabaseService.isLoggedIn) {
          _showConsentDialog();
        }
      }
    } catch (e) {
      setState(() {
        _isProcessing = false;
        _errorMessage = e.toString().replaceFirst('Exception: ', '');
      });
    }
  }

  Future<void> _showConsentDialog() async {
    if (_selectedImage == null || !mounted) return;

    final consent = await showDialog<bool>(
      context: context,
      builder: (context) => AlertDialog(
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
        title: const Row(
          children: [
            Icon(Icons.cloud_upload, color: Colors.green),
            SizedBox(width: 8),
            Text('Simpan ke Database?'),
          ],
        ),
        content: const Text(
          'Apakah Anda ingin mengunggah foto ini ke database kami?\n\n'
          'Foto akan digunakan untuk meningkatkan akurasi model '
          'dan tidak akan dibagikan ke pihak ketiga.',
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context, false),
            child: const Text('Tidak, Terima Kasih'),
          ),
          ElevatedButton.icon(
            onPressed: () => Navigator.pop(context, true),
            icon: const Icon(Icons.cloud_upload, size: 18),
            label: const Text('Ya, Unggah'),
            style: ElevatedButton.styleFrom(
              backgroundColor: Colors.green.shade700,
              foregroundColor: Colors.white,
            ),
          ),
        ],
      ),
    );

    if (consent == true && mounted) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Mengunggah foto...')),
      );
      final url = await SupabaseService.uploadImage(_selectedImage!);
      if (mounted) {
        final errorMsg = SupabaseService.lastUploadError;
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text(url != null
                ? 'Foto berhasil diunggah!'
                : 'Gagal: ${errorMsg ?? "Unknown error"}'),
            backgroundColor: url != null ? Colors.green : Colors.red,
            duration: const Duration(seconds: 5),
          ),
        );
      }
    }
  }

  Future<void> _handleLogout() async {
    await SupabaseService.signOut();
    if (mounted) {
      Navigator.pushReplacement(context, MaterialPageRoute(builder: (_) => const LoginScreen()));
    }
  }

  @override
  Widget build(BuildContext context) {
    final isLoggedIn = SupabaseService.isLoggedIn;
    final userEmail = SupabaseService.currentUser?.email;

    return Scaffold(
      appBar: AppBar(
        title: const Text('Fruit Freshness Detector'),
        centerTitle: true,
        backgroundColor: Colors.green.shade700,
        foregroundColor: Colors.white,
        elevation: 2,
        actions: [
          IconButton(
            icon: const Icon(Icons.history),
            tooltip: 'Riwayat',
            onPressed: () => Navigator.push(context, MaterialPageRoute(builder: (_) => const HistoryScreen())),
          ),
          if (isLoggedIn)
            PopupMenuButton<String>(
              icon: const Icon(Icons.account_circle),
              onSelected: (value) {
                if (value == 'logout') _handleLogout();
              },
              itemBuilder: (context) => [
                PopupMenuItem(enabled: false, child: Text(userEmail ?? 'User', style: const TextStyle(fontWeight: FontWeight.bold))),
                const PopupMenuDivider(),
                const PopupMenuItem(value: 'logout', child: Text('Logout')),
              ],
            )
          else
            TextButton.icon(
              onPressed: () => Navigator.pushReplacement(context, MaterialPageRoute(builder: (_) => const LoginScreen())),
              icon: const Icon(Icons.login, color: Colors.white, size: 20),
              label: const Text('Login', style: TextStyle(color: Colors.white)),
            ),
        ],
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            // Image display
            Card(
              clipBehavior: Clip.antiAlias,
              shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
              elevation: 2,
              child: Container(
                height: 280,
                color: Colors.grey.shade100,
                child: _selectedImage != null
                    ? Image.file(_selectedImage!, fit: BoxFit.contain)
                    : Center(
                        child: Column(
                          mainAxisAlignment: MainAxisAlignment.center,
                          children: [
                            Icon(Icons.add_photo_alternate, size: 64, color: Colors.grey.shade400),
                            const SizedBox(height: 12),
                            Text('Ambil foto atau pilih dari galeri', style: TextStyle(color: Colors.grey.shade600, fontSize: 16)),
                            const SizedBox(height: 4),
                            Text('Pastikan buah terlihat jelas & close-up', style: TextStyle(color: Colors.grey.shade400, fontSize: 13)),
                          ],
                        ),
                      ),
              ),
            ),

            const SizedBox(height: 16),

            // Buttons
            Row(
              children: [
                Expanded(
                  child: ElevatedButton.icon(
                    onPressed: !_isProcessing ? () => _pickImage(ImageSource.camera) : null,
                    icon: const Icon(Icons.camera_alt),
                    label: const Text('Kamera'),
                    style: ElevatedButton.styleFrom(
                      padding: const EdgeInsets.symmetric(vertical: 16),
                      backgroundColor: Colors.green.shade700,
                      foregroundColor: Colors.white,
                      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
                      textStyle: const TextStyle(fontSize: 15, fontWeight: FontWeight.w600),
                    ),
                  ),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: ElevatedButton.icon(
                    onPressed: !_isProcessing ? () => _pickImage(ImageSource.gallery) : null,
                    icon: const Icon(Icons.photo_library),
                    label: const Text('Galeri'),
                    style: ElevatedButton.styleFrom(
                      padding: const EdgeInsets.symmetric(vertical: 16),
                      backgroundColor: Colors.blue.shade700,
                      foregroundColor: Colors.white,
                      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
                      textStyle: const TextStyle(fontSize: 15, fontWeight: FontWeight.w600),
                    ),
                  ),
                ),
              ],
            ),

            const SizedBox(height: 20),

            // Processing
            if (_isProcessing)
              Card(
                shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
                child: const Padding(
                  padding: EdgeInsets.all(20),
                  child: Row(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      SizedBox(width: 24, height: 24, child: CircularProgressIndicator(strokeWidth: 3)),
                      SizedBox(width: 12),
                      Text('Menganalisis gambar...', style: TextStyle(fontSize: 15)),
                    ],
                  ),
                ),
              ),

            // Error
            if (_errorMessage != null && !_isProcessing)
              Card(
                color: Colors.red.shade50,
                shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
                child: Padding(
                  padding: const EdgeInsets.all(16),
                  child: Row(
                    children: [
                      const Icon(Icons.error_outline, color: Colors.red),
                      const SizedBox(width: 12),
                      Expanded(child: Text(_errorMessage!, style: const TextStyle(color: Colors.red))),
                    ],
                  ),
                ),
              ),

            // === RESULTS ===
            if (_result != null && !_isProcessing) ...[
              _buildResultCard(),
              const SizedBox(height: 12),
              if (_confidence != null) _buildConfidenceBar(),
              const SizedBox(height: 12),
              if (_preprocessedImageBytes != null)
                _buildImageCard('Hasil Preprocessing & Segmentasi', _preprocessedImageBytes!, 'Gambar setelah resize, Gaussian blur, CLAHE, dan segmentasi buah'),
              const SizedBox(height: 12),
              if (_keypointsImageBytes != null)
                _buildImageCard('ORB Keypoints (Fitur Tekstur)', _keypointsImageBytes!, 'Titik merah = fitur tekstur yang terdeteksi oleh ORB pada gambar original'),
              const SizedBox(height: 12),
              _buildFeatureInfoCard(),
            ],
          ],
        ),
      ),
    );
  }

  Widget _buildResultCard() {
    final isFresh = _result == 'fresh';
    return Card(
      elevation: 4,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
      child: Container(
        decoration: BoxDecoration(
          borderRadius: BorderRadius.circular(16),
          gradient: LinearGradient(colors: isFresh ? [Colors.green.shade50, Colors.green.shade100] : [Colors.red.shade50, Colors.red.shade100]),
        ),
        padding: const EdgeInsets.all(24),
        child: Column(
          children: [
            Icon(isFresh ? Icons.check_circle : Icons.warning_rounded, size: 56, color: isFresh ? Colors.green.shade700 : Colors.red.shade700),
            const SizedBox(height: 8),
            Text(isFresh ? 'SEGAR' : 'BUSUK', style: TextStyle(fontSize: 32, fontWeight: FontWeight.bold, color: isFresh ? Colors.green.shade800 : Colors.red.shade800)),
            if (_confidence != null) ...[
              const SizedBox(height: 4),
              Text('Confidence: ${_confidence!.toStringAsFixed(1)}%', style: TextStyle(fontSize: 18, fontWeight: FontWeight.w500, color: isFresh ? Colors.green.shade700 : Colors.red.shade700)),
            ],
          ],
        ),
      ),
    );
  }

  Widget _buildConfidenceBar() {
    final freshConf = _result == 'fresh' ? _confidence! : 100 - _confidence!;
    final rottenConf = _result == 'rotten' ? _confidence! : 100 - _confidence!;
    return Card(
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text('Detail Confidence', style: TextStyle(fontWeight: FontWeight.bold, fontSize: 15)),
            const SizedBox(height: 12),
            _buildProgressRow('Segar (Fresh)', freshConf, Colors.green),
            const SizedBox(height: 8),
            _buildProgressRow('Busuk (Rotten)', rottenConf, Colors.red),
          ],
        ),
      ),
    );
  }

  Widget _buildProgressRow(String label, double value, Color color) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(mainAxisAlignment: MainAxisAlignment.spaceBetween, children: [
          Text(label, style: const TextStyle(fontSize: 13)),
          Text('${value.toStringAsFixed(1)}%', style: TextStyle(fontWeight: FontWeight.bold, color: color)),
        ]),
        const SizedBox(height: 4),
        ClipRRect(
          borderRadius: BorderRadius.circular(6),
          child: LinearProgressIndicator(value: value / 100, minHeight: 10, backgroundColor: Colors.grey.shade200, valueColor: AlwaysStoppedAnimation(color)),
        ),
      ],
    );
  }

  Widget _buildImageCard(String title, Uint8List imageBytes, String caption) {
    return Card(
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
      clipBehavior: Clip.antiAlias,
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Padding(padding: const EdgeInsets.fromLTRB(16, 12, 16, 8), child: Text(title, style: const TextStyle(fontWeight: FontWeight.bold, fontSize: 15))),
          Image.memory(imageBytes, fit: BoxFit.contain, width: double.infinity),
          Padding(padding: const EdgeInsets.fromLTRB(16, 8, 16, 12), child: Text(caption, style: TextStyle(color: Colors.grey.shade600, fontSize: 12))),
        ],
      ),
    );
  }

  Widget _buildFeatureInfoCard() {
    return Card(
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Row(children: [
              Icon(Icons.analytics, size: 20, color: Colors.green),
              SizedBox(width: 8),
              Text('Pipeline Analisis', style: TextStyle(fontWeight: FontWeight.bold, fontSize: 15)),
            ]),
            const Divider(height: 20),
            _buildInfoRow(Icons.crop, 'Preprocessing', 'Resize 128×128 → Gaussian Blur → CLAHE'),
            _buildInfoRow(Icons.center_focus_strong, 'Segmentasi', 'HSV Threshold + Contour Detection'),
            _buildInfoRow(Icons.palette, 'Fitur Warna', 'HSV Histogram 8×8×8 (512) + Mean HSV (3)'),
            _buildInfoRow(Icons.texture, 'Fitur Tekstur', 'ORB Descriptor mean (32)'),
            _buildInfoRow(Icons.numbers, 'Total Features', '547 dimensi'),
            _buildInfoRow(Icons.memory, 'Classifier', 'SVM (RBF kernel) via ONNX'),
          ],
        ),
      ),
    );
  }

  Widget _buildInfoRow(IconData icon, String label, String value) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 5),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Icon(icon, size: 16, color: Colors.grey.shade500),
          const SizedBox(width: 8),
          SizedBox(width: 100, child: Text(label, style: TextStyle(color: Colors.grey.shade700, fontWeight: FontWeight.w500, fontSize: 13))),
          Expanded(child: Text(value, style: const TextStyle(fontSize: 13))),
        ],
      ),
    );
  }
}
