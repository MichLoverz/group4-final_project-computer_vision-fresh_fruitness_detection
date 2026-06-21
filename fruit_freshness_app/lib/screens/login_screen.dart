import 'package:flutter/material.dart';
import 'package:shared_preferences/shared_preferences.dart';
import '../services/supabase_service.dart';
import 'home_screen.dart';

class LoginScreen extends StatefulWidget {
  const LoginScreen({super.key});

  @override
  State<LoginScreen> createState() => _LoginScreenState();
}

class _LoginScreenState extends State<LoginScreen> {
  bool _loading = false;
  bool _skipAlways = false;
  static const String _skipKey = 'skip_login_always';

  @override
  void initState() {
    super.initState();
    // Delay agar context dan Navigator siap
    WidgetsBinding.instance.addPostFrameCallback((_) {
      _checkAutoSkip();
    });
  }

  Future<void> _checkAutoSkip() async {
    // Kalau sudah login, langsung ke home
    if (SupabaseService.isLoggedIn) {
      _goHome();
      return;
    }

    // Kalau user sudah pilih "selalu lewati", langsung ke home
    final prefs = await SharedPreferences.getInstance();
    if (prefs.getBool(_skipKey) == true) {
      _goHome();
    }
  }

  void _goHome() {
    if (!mounted) return;
    Navigator.pushReplacement(
      context,
      MaterialPageRoute(builder: (_) => const HomeScreen()),
    );
  }

  Future<void> _handleLogin() async {
    setState(() => _loading = true);
    final success = await SupabaseService.signInWithGoogle();
    setState(() => _loading = false);

    if (success && mounted) {
      // Tampilkan user agreement
      final agreed = await _showUserAgreement();
      if (agreed) {
        _goHome();
      } else {
        // Kalau tidak setuju, logout
        await SupabaseService.signOut();
        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            const SnackBar(
              content: Text('Anda harus menyetujui ketentuan untuk melanjutkan dengan akun Google'),
              backgroundColor: Colors.orange,
            ),
          );
        }
      }
    } else if (mounted) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text('Login gagal, coba lagi'),
          backgroundColor: Colors.red,
        ),
      );
    }
  }

  Future<bool> _showUserAgreement() async {
    bool checked = false;
    final result = await showDialog<bool>(
      context: context,
      barrierDismissible: false,
      builder: (context) => StatefulBuilder(
        builder: (context, setDialogState) => AlertDialog(
          title: const Text('Ketentuan Penggunaan'),
          content: SingleChildScrollView(
            child: Column(
              mainAxisSize: MainAxisSize.min,
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Text(
                  'Dengan login, Anda menyetujui bahwa:',
                  style: TextStyle(fontWeight: FontWeight.bold),
                ),
                const SizedBox(height: 12),
                const Text('1. Foto buah yang Anda prediksi dapat disimpan ke server kami untuk keperluan pengembangan dan peningkatan akurasi model.'),
                const SizedBox(height: 8),
                const Text('2. Foto tidak akan dibagikan ke pihak ketiga atau digunakan untuk tujuan komersial.'),
                const SizedBox(height: 8),
                const Text('3. Anda dapat memilih untuk tidak mengunggah foto pada setiap prediksi.'),
                const SizedBox(height: 16),
                Row(
                  children: [
                    Checkbox(
                      value: checked,
                      onChanged: (v) => setDialogState(() => checked = v ?? false),
                      activeColor: Colors.green.shade700,
                    ),
                    const Expanded(
                      child: Text(
                        'Saya telah membaca dan menyetujui ketentuan di atas',
                        style: TextStyle(fontSize: 13),
                      ),
                    ),
                  ],
                ),
              ],
            ),
          ),
          actions: [
            TextButton(
              onPressed: () => Navigator.pop(context, false),
              child: const Text('Batal'),
            ),
            ElevatedButton(
              onPressed: checked ? () => Navigator.pop(context, true) : null,
              style: ElevatedButton.styleFrom(
                backgroundColor: Colors.green.shade700,
                foregroundColor: Colors.white,
              ),
              child: const Text('Setuju & Lanjut'),
            ),
          ],
        ),
      ),
    );
    return result ?? false;
  }

  Future<void> _handleSkip() async {
    if (_skipAlways) {
      final prefs = await SharedPreferences.getInstance();
      await prefs.setBool(_skipKey, true);
    }
    _goHome();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Container(
        decoration: BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topCenter,
            end: Alignment.bottomCenter,
            colors: [Colors.green.shade800, Colors.green.shade400],
          ),
        ),
        child: SafeArea(
          child: Center(
            child: SingleChildScrollView(
              padding: const EdgeInsets.all(32),
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  // Logo / Icon
                  Container(
                    padding: const EdgeInsets.all(24),
                    decoration: BoxDecoration(
                      color: Colors.white.withValues(alpha: 0.2),
                      shape: BoxShape.circle,
                    ),
                    child: const Icon(
                      Icons.eco,
                      size: 80,
                      color: Colors.white,
                    ),
                  ),
                  const SizedBox(height: 24),

                  const Text(
                    'Fruit Freshness\nDetector',
                    textAlign: TextAlign.center,
                    style: TextStyle(
                      fontSize: 32,
                      fontWeight: FontWeight.bold,
                      color: Colors.white,
                      height: 1.2,
                    ),
                  ),
                  const SizedBox(height: 8),
                  Text(
                    'Deteksi kesegaran buah menggunakan\nComputer Vision & Machine Learning',
                    textAlign: TextAlign.center,
                    style: TextStyle(
                      fontSize: 14,
                      color: Colors.white.withValues(alpha: 0.9),
                    ),
                  ),

                  const SizedBox(height: 48),

                  // Login Button
                  SizedBox(
                    width: double.infinity,
                    child: ElevatedButton.icon(
                      onPressed: _loading ? null : _handleLogin,
                      icon: _loading
                          ? const SizedBox(
                              width: 20,
                              height: 20,
                              child: CircularProgressIndicator(strokeWidth: 2),
                            )
                          : const Icon(Icons.login),
                      label: Text(_loading ? 'Memproses...' : 'Login dengan Google'),
                      style: ElevatedButton.styleFrom(
                        padding: const EdgeInsets.symmetric(vertical: 16),
                        backgroundColor: Colors.white,
                        foregroundColor: Colors.green.shade800,
                        textStyle: const TextStyle(fontSize: 16, fontWeight: FontWeight.w600),
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(12),
                        ),
                      ),
                    ),
                  ),

                  const SizedBox(height: 16),

                  // Info text
                  Text(
                    'Login untuk menyimpan foto ke database\ndan membantu pengembangan model',
                    textAlign: TextAlign.center,
                    style: TextStyle(
                      fontSize: 12,
                      color: Colors.white.withValues(alpha: 0.7),
                    ),
                  ),

                  const SizedBox(height: 32),

                  // Divider
                  Row(
                    children: [
                      Expanded(child: Divider(color: Colors.white.withValues(alpha: 0.3))),
                      Padding(
                        padding: const EdgeInsets.symmetric(horizontal: 16),
                        child: Text('atau', style: TextStyle(color: Colors.white.withValues(alpha: 0.7))),
                      ),
                      Expanded(child: Divider(color: Colors.white.withValues(alpha: 0.3))),
                    ],
                  ),

                  const SizedBox(height: 24),

                  // Skip checkbox
                  Row(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      Checkbox(
                        value: _skipAlways,
                        onChanged: (v) => setState(() => _skipAlways = v ?? false),
                        side: const BorderSide(color: Colors.white),
                        checkColor: Colors.green.shade800,
                        fillColor: WidgetStateProperty.resolveWith(
                          (states) => states.contains(WidgetState.selected)
                              ? Colors.white
                              : Colors.transparent,
                        ),
                      ),
                      Text(
                        'Jangan tampilkan halaman ini lagi',
                        style: TextStyle(
                          color: Colors.white.withValues(alpha: 0.9),
                          fontSize: 13,
                        ),
                      ),
                    ],
                  ),

                  const SizedBox(height: 8),

                  // Skip Button
                  TextButton(
                    onPressed: _loading ? null : _handleSkip,
                    child: const Text(
                      'Lewati, lanjut tanpa login →',
                      style: TextStyle(
                        color: Colors.white,
                        fontSize: 15,
                        decoration: TextDecoration.underline,
                        decorationColor: Colors.white,
                      ),
                    ),
                  ),
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }
}
