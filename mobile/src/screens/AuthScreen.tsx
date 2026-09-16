import React, { useState } from "react";
import {
  ActivityIndicator,
  KeyboardAvoidingView,
  Platform,
  SafeAreaView,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  TouchableOpacity,
  useWindowDimensions,
  View,
} from "react-native";

export type AuthUser = {
  id: number;
  username: string;
};

type Props = {
  apiUrl: string;
  onAuthenticated: (token: string, user: AuthUser) => void;
};

export default function AuthScreen({ apiUrl, onAuthenticated }: Props) {
  const { width: screenWidth, height: screenHeight } = useWindowDimensions();
  const compactHeight = screenHeight < 700;
  const horizontalPadding = screenWidth < 360 ? 14 : 22;
  const [mode, setMode] = useState<"login" | "register">("login");
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const switchMode = (nextMode: "login" | "register") => {
    setMode(nextMode);
    setError("");
    setPassword("");
    setConfirmPassword("");
  };

  const submit = async () => {
    const cleanUsername = username.trim();

    if (!cleanUsername || !password) {
      setError("Vui lòng nhập đầy đủ tên đăng nhập và mật khẩu.");
      return;
    }
    if (cleanUsername.length < 3) {
      setError("Tên đăng nhập phải có ít nhất 3 ký tự.");
      return;
    }
    if (password.length < 6) {
      setError("Mật khẩu phải có ít nhất 6 ký tự.");
      return;
    }
    if (mode === "register" && password !== confirmPassword) {
      setError("Mật khẩu nhập lại chưa khớp.");
      return;
    }

    try {
      setLoading(true);
      setError("");

      const response = await fetch(`${apiUrl}/api/mobile/${mode}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          username: cleanUsername,
          password,
          confirm_password: confirmPassword,
        }),
      });

      const raw = await response.text();
      let data: any = {};
      try {
        data = raw ? JSON.parse(raw) : {};
      } catch {
        throw new Error(
          "Backend không trả JSON. Hãy kiểm tra đúng IP Flask và khởi động lại server.py."
        );
      }

      if (!response.ok) {
        throw new Error(data.error || "Không thể xác thực tài khoản.");
      }
      if (!data.token || !data.user?.username) {
        throw new Error("Backend chưa trả token đăng nhập hợp lệ.");
      }

      onAuthenticated(String(data.token), {
        id: Number(data.user.id),
        username: String(data.user.username),
      });
    } catch (submitError: any) {
      setError(
        submitError?.message ||
          "Không kết nối được Backend. Kiểm tra Flask, IP laptop và Wi-Fi."
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <SafeAreaView style={styles.safeArea}>
      <KeyboardAvoidingView
        style={styles.flex}
        behavior={Platform.OS === "ios" ? "padding" : "height"}
        keyboardVerticalOffset={0}
      >
        <ScrollView
          contentContainerStyle={[
            styles.container,
            {
              paddingHorizontal: horizontalPadding,
              paddingVertical: compactHeight ? 18 : 32,
            },
          ]}
          keyboardShouldPersistTaps="handled"
          keyboardDismissMode={Platform.OS === "ios" ? "interactive" : "on-drag"}
          showsVerticalScrollIndicator={false}
        >
          <View style={[styles.logoCircle, compactHeight && styles.logoCircleCompact]}>
            <Text style={styles.logo}>🚦</Text>
          </View>
          <Text style={[styles.title, compactHeight && styles.titleCompact]}>Trợ lý biển báo</Text>
          <Text style={styles.subtitle}>
            Nhận diện biển báo và tra cứu luật giao thông
          </Text>

          <View style={[styles.card, { width: "100%", maxWidth: 520, alignSelf: "center" }]}>
            <View style={styles.tabs}>
              <TouchableOpacity
                style={[styles.tab, mode === "login" && styles.tabActive]}
                onPress={() => switchMode("login")}
                disabled={loading}
              >
                <Text
                  style={[
                    styles.tabText,
                    mode === "login" && styles.tabTextActive,
                  ]}
                >
                  Đăng nhập
                </Text>
              </TouchableOpacity>
              <TouchableOpacity
                style={[styles.tab, mode === "register" && styles.tabActive]}
                onPress={() => switchMode("register")}
                disabled={loading}
              >
                <Text
                  style={[
                    styles.tabText,
                    mode === "register" && styles.tabTextActive,
                  ]}
                >
                  Đăng ký
                </Text>
              </TouchableOpacity>
            </View>

            <Text style={styles.label}>Tên đăng nhập</Text>
            <TextInput
              value={username}
              onChangeText={setUsername}
              style={styles.input}
              placeholder="Ví dụ: thao22070003"
              placeholderTextColor="#9ca3af"
              autoCapitalize="none"
              autoCorrect={false}
              editable={!loading}
              returnKeyType="next"
            />

            <Text style={styles.label}>Mật khẩu</Text>
            <TextInput
              value={password}
              onChangeText={setPassword}
              style={styles.input}
              placeholder="Tối thiểu 6 ký tự"
              placeholderTextColor="#9ca3af"
              secureTextEntry
              editable={!loading}
              returnKeyType={mode === "login" ? "done" : "next"}
              onSubmitEditing={mode === "login" ? submit : undefined}
            />

            {mode === "register" && (
              <>
                <Text style={styles.label}>Nhập lại mật khẩu</Text>
                <TextInput
                  value={confirmPassword}
                  onChangeText={setConfirmPassword}
                  style={styles.input}
                  placeholder="Nhập lại mật khẩu"
                  placeholderTextColor="#9ca3af"
                  secureTextEntry
                  editable={!loading}
                  returnKeyType="done"
                  onSubmitEditing={submit}
                />
              </>
            )}

            {!!error && (
              <View style={styles.errorBox}>
                <Text style={styles.errorText}>⚠️ {error}</Text>
              </View>
            )}

            <TouchableOpacity
              style={[styles.submitButton, loading && styles.disabledButton]}
              onPress={submit}
              disabled={loading}
            >
              {loading ? (
                <ActivityIndicator color="#fff" />
              ) : (
                <Text style={styles.submitText}>
                  {mode === "login" ? "Đăng nhập" : "Tạo tài khoản"}
                </Text>
              )}
            </TouchableOpacity>

            <Text style={styles.switchHint}>
              {mode === "login"
                ? "Chưa có tài khoản? Chọn tab Đăng ký."
                : "Đã có tài khoản? Chọn tab Đăng nhập."}
            </Text>
          </View>

          <Text style={styles.backendText}>Backend: {apiUrl}</Text>
        </ScrollView>
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safeArea: { flex: 1, backgroundColor: "#eef4fb" },
  flex: { flex: 1 },
  container: {
    flexGrow: 1,
    justifyContent: "center",
    width: "100%",
  },
  logoCircle: {
    alignSelf: "center",
    width: 78,
    height: 78,
    borderRadius: 39,
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: "#dbeafe",
    marginBottom: 14,
  },
  logoCircleCompact: { width: 62, height: 62, borderRadius: 31, marginBottom: 10 },
  logo: { fontSize: 40 },
  title: {
    color: "#111827",
    fontSize: 30,
    fontWeight: "900",
    textAlign: "center",
  },
  titleCompact: { fontSize: 26 },
  subtitle: {
    color: "#6b7280",
    textAlign: "center",
    marginTop: 8,
    marginBottom: 20,
    lineHeight: 20,
  },
  card: {
    backgroundColor: "#fff",
    borderRadius: 22,
    padding: 20,
    shadowColor: "#1e3a8a",
    shadowOpacity: 0.08,
    shadowRadius: 18,
    shadowOffset: { width: 0, height: 8 },
    elevation: 4,
  },
  tabs: {
    flexDirection: "row",
    backgroundColor: "#f1f5f9",
    borderRadius: 14,
    padding: 4,
    marginBottom: 20,
  },
  tab: { flex: 1, paddingVertical: 11, alignItems: "center", borderRadius: 11 },
  tabActive: { backgroundColor: "#2563eb" },
  tabText: { color: "#64748b", fontWeight: "800" },
  tabTextActive: { color: "#fff" },
  label: {
    color: "#374151",
    fontWeight: "800",
    marginBottom: 7,
    marginTop: 5,
  },
  input: {
    borderWidth: 1,
    borderColor: "#dbe3ef",
    backgroundColor: "#f8fafc",
    borderRadius: 13,
    paddingHorizontal: 14,
    paddingVertical: 13,
    color: "#111827",
    marginBottom: 13,
  },
  errorBox: {
    backgroundColor: "#fff1f2",
    borderWidth: 1,
    borderColor: "#fecdd3",
    borderRadius: 12,
    padding: 11,
    marginTop: 2,
    marginBottom: 12,
  },
  errorText: { color: "#be123c", lineHeight: 19, fontSize: 13 },
  submitButton: {
    backgroundColor: "#1d4ed8",
    borderRadius: 14,
    paddingVertical: 14,
    alignItems: "center",
    marginTop: 4,
  },
  disabledButton: { opacity: 0.65 },
  submitText: { color: "#fff", fontWeight: "900", fontSize: 16 },
  switchHint: {
    color: "#64748b",
    textAlign: "center",
    fontSize: 12,
    marginTop: 14,
  },
  backendText: {
    color: "#94a3b8",
    textAlign: "center",
    fontSize: 11,
    marginTop: 18,
  },
});
