import 'dart:io';
import 'package:flutter/foundation.dart';
import 'package:supabase_flutter/supabase_flutter.dart';
import 'package:google_sign_in/google_sign_in.dart';

class SupabaseService {
  // ============================================================
  // GANTI NILAI DI BAWAH DENGAN DATA DARI DASHBOARD SUPABASE KAMU
  // ============================================================
  static const String supabaseUrl = 'https://blzpgicmhyokbxdgzadt.supabase.co';
  static const String supabaseAnonKey = 'sb_publishable_kZhjQMw-hveOEUgCctV8ag_LBYbF9VG';

  // Google OAuth Web Client ID (dari Google Cloud Console)
  static const String googleWebClientId = '84065986059-77shrr9c808mvbii1n9cbhmoak85fsgh.apps.googleusercontent.com';

  static const String storageBucket = 'fruit-images';

  static SupabaseClient get client => Supabase.instance.client;

  /// Initialize Supabase (panggil di main.dart)
  static Future<void> initialize() async {
    await Supabase.initialize(
      url: supabaseUrl,
      anonKey: supabaseAnonKey,
    );

    // Initialize Google Sign In
    await GoogleSignIn.instance.initialize(
      serverClientId: googleWebClientId,
    );
  }

  /// Cek apakah user sudah login
  static bool get isLoggedIn => client.auth.currentUser != null;

  /// Get current user
  static User? get currentUser => client.auth.currentUser;

  /// Login dengan Google
  static Future<bool> signInWithGoogle() async {
    try {
      final googleUser = await GoogleSignIn.instance.authenticate();
      final idToken = googleUser.authentication.idToken;

      if (idToken == null) return false;

      await client.auth.signInWithIdToken(
        provider: OAuthProvider.google,
        idToken: idToken,
      );

      return true;
    } catch (e) {
      return false;
    }
  }

  /// Logout
  static Future<void> signOut() async {
    await GoogleSignIn.instance.signOut();
    await client.auth.signOut();
  }

  /// Upload foto ke Supabase Storage
  /// Returns URL publik jika berhasil, null jika gagal
  /// [errorMsg] diisi dengan pesan error jika gagal
  static String? lastUploadError;

  static Future<String?> uploadImage(File imageFile) async {
    lastUploadError = null;

    if (!isLoggedIn) {
      lastUploadError = 'Anda harus login terlebih dahulu untuk mengunggah foto';
      debugPrint('Upload error: not logged in');
      return null;
    }

    try {
      final userId = currentUser!.id;
      final timestamp = DateTime.now().millisecondsSinceEpoch;
      final pathParts = imageFile.path.split('.');
      final ext = pathParts.length > 1 ? pathParts.last : 'jpg';
      final filePath = '$userId/$timestamp.$ext';

      final bytes = await imageFile.readAsBytes();

      await client.storage.from(storageBucket).uploadBinary(
            filePath,
            bytes,
            fileOptions: const FileOptions(
              contentType: 'image/jpeg',
              upsert: true,
            ),
          );

      final url = client.storage.from(storageBucket).getPublicUrl(filePath);
      return url;
    } catch (e) {
      lastUploadError = e.toString();
      debugPrint('Upload error: $e');
      return null;
    }
  }
}
