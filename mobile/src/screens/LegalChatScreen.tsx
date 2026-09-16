import React, { useEffect, useRef, useState } from "react";
import {
  ActivityIndicator,
  Keyboard,
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

type ChatMessage = {
  id: string;
  role: "user" | "assistant";
  text: string;
  mode?: string;
};

type Props = {
  apiUrl: string;
  authToken: string;
  username: string;
  vehicleType: string;
  contextCode?: string;
  onBack: () => void;
  onUnauthorized: () => void | Promise<void>;
  onLogout: () => void | Promise<void>;
};

const CONTEXT_SUGGESTIONS = [
  "Biển này có ý nghĩa gì?",
  "Vi phạm biển này bị phạt bao nhiêu?",
  "Biển này có áp dụng cho xe máy không?",
  "Biển này có áp dụng cho ô tô không?",
  "Mức phạt này căn cứ điều, khoản, điểm nào?",
];

const GENERAL_SUGGESTIONS = [
  "Biển P.130 có ý nghĩa gì?",
  "Xe máy vượt đèn đỏ bị phạt bao nhiêu?",
  "Ô tô đi ngược chiều bị phạt bao nhiêu?",
  "Nghị định nào đang được dùng để tra mức phạt?",
];

const cleanMarkdown = (value: string) =>
  String(value || "")
    .replace(/\*\*(.*?)\*\*/g, "$1")
    .replace(/`([^`]+)`/g, "$1")
    .trim();

export default function LegalChatScreen({
  apiUrl,
  authToken,
  username,
  vehicleType,
  contextCode = "",
  onBack,
  onUnauthorized,
  onLogout,
}: Props) {
  const { width: screenWidth } = useWindowDimensions();
  const isCompactPhone = screenWidth < 360;
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      id: "welcome",
      role: "assistant",
      text: contextCode
        ? `Mình đang dùng biển ${contextCode} làm ngữ cảnh. Bạn có thể hỏi ý nghĩa, đối tượng áp dụng, mức phạt hoặc căn cứ pháp lý.`
        : "Xin chào! Mình có thể giải thích biển báo, đối tượng áp dụng, mức phạt và căn cứ pháp lý giao thông.",
      mode: "welcome",
    },
  ]);
  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);
  const [activeContextCode, setActiveContextCode] = useState(contextCode);
  const [suggestions, setSuggestions] = useState(
    contextCode ? CONTEXT_SUGGESTIONS : GENERAL_SUGGESTIONS
  );
  const [ragReady, setRagReady] = useState<boolean | null>(null);
  const [keyboardVisible, setKeyboardVisible] = useState(false);
  const scrollRef = useRef<ScrollView | null>(null);

  useEffect(() => {
    let mounted = true;

    fetch(`${apiUrl}/api/chatbot/status`, {
      headers: { Authorization: `Bearer ${authToken}` },
    })
      .then(async (response) => {
        if (response.status === 401) {
          await onUnauthorized();
          return null;
        }
        const raw = await response.text();
        return raw ? JSON.parse(raw) : null;
      })
      .then((data) => {
        if (mounted && data) setRagReady(Boolean(data.ready));
      })
      .catch(() => {
        if (mounted) setRagReady(false);
      });

    return () => {
      mounted = false;
    };
  }, [apiUrl, authToken, onUnauthorized]);

  useEffect(() => {
    const nextContext = String(contextCode || "").trim();
    setActiveContextCode(nextContext);
    setSuggestions(nextContext ? CONTEXT_SUGGESTIONS : GENERAL_SUGGESTIONS);
  }, [contextCode]);

  useEffect(() => {
    const showEvent = Platform.OS === "ios" ? "keyboardWillShow" : "keyboardDidShow";
    const hideEvent = Platform.OS === "ios" ? "keyboardWillHide" : "keyboardDidHide";

    const showSub = Keyboard.addListener(showEvent, () => {
      setKeyboardVisible(true);
      setTimeout(() => scrollRef.current?.scrollToEnd({ animated: true }), 100);
    });
    const hideSub = Keyboard.addListener(hideEvent, () => setKeyboardVisible(false));

    return () => {
      showSub.remove();
      hideSub.remove();
    };
  }, []);

  useEffect(() => {
    const timer = setTimeout(() => scrollRef.current?.scrollToEnd({ animated: true }), 80);
    return () => clearTimeout(timer);
  }, [messages, sending]);

  const sendQuestion = async (suggestedQuestion?: string) => {
    const question = String(suggestedQuestion ?? input).trim();
    if (!question || sending) return;

    const userMessage: ChatMessage = {
      id: `user-${Date.now()}`,
      role: "user",
      text: question,
    };
    const historyForServer = [...messages, userMessage]
      .filter((message) => message.id !== "welcome")
      .slice(-10)
      .map((message) => ({ role: message.role, text: message.text }));

    setMessages((current) => [...current, userMessage]);
    setInput("");
    setSending(true);

    try {
      const response = await fetch(`${apiUrl}/api/chatbot`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${authToken}`,
        },
        body: JSON.stringify({
          question,
          vehicle_type: vehicleType,
          context_code: activeContextCode,
          history: historyForServer,
        }),
      });

      if (response.status === 401) {
        await onUnauthorized();
        return;
      }

      const raw = await response.text();
      let data: any = {};
      try {
        data = raw ? JSON.parse(raw) : {};
      } catch {
        throw new Error("Backend trả về nội dung không phải JSON.");
      }
      if (!response.ok) {
        throw new Error(data.error || "Chatbox chưa thể trả lời.");
      }

      const answer = cleanMarkdown(data.answer || "Mình chưa có câu trả lời phù hợp.");
      setMessages((current) => [
        ...current,
        {
          id: `assistant-${Date.now()}`,
          role: "assistant",
          text: answer,
          mode: String(data.mode || "knowledge"),
        },
      ]);

      const nextContextCode = String(
        data.context_code || activeContextCode || ""
      ).trim();

      if (nextContextCode !== activeContextCode) {
        setActiveContextCode(nextContextCode);
      }

      if (nextContextCode) {
        if (Array.isArray(data.suggestions) && data.suggestions.length) {
          setSuggestions(data.suggestions.slice(0, 5).map(String));
        } else {
          setSuggestions(CONTEXT_SUGGESTIONS);
        }
      } else {
        setSuggestions(GENERAL_SUGGESTIONS);
      }

      if (data.rag_status) setRagReady(Boolean(data.rag_status.ready));
    } catch (error: any) {
      setMessages((current) => [
        ...current,
        {
          id: `error-${Date.now()}`,
          role: "assistant",
          text:
            error?.message ||
            "Không kết nối được trợ lý. Hãy kiểm tra Flask và địa chỉ IP.",
          mode: "error",
        },
      ]);
    } finally {
      setSending(false);
    }
  };

  return (
    <SafeAreaView style={styles.safeArea}>
      <KeyboardAvoidingView
        style={styles.flex}
        behavior={Platform.OS === "ios" ? "padding" : "height"}
        keyboardVerticalOffset={0}
      >
        <View style={styles.header}>
          <TouchableOpacity style={styles.headerButton} onPress={onBack}>
            <Text style={styles.headerButtonText}>‹ Nhận diện</Text>
          </TouchableOpacity>
          <View style={styles.headerCenter}>
            <Text style={[styles.headerTitle, isCompactPhone && styles.headerTitleCompact]} numberOfLines={1}>Trợ lý luật giao thông</Text>
            <Text style={styles.headerSubtitle} numberOfLines={1}>
              {activeContextCode ? `Ngữ cảnh: ${activeContextCode}` : `Tài khoản: ${username}`}
            </Text>
          </View>
          <TouchableOpacity style={styles.logoutButton} onPress={onLogout}>
            <Text style={styles.logoutText}>Thoát</Text>
          </TouchableOpacity>
        </View>

        <View style={styles.statusBar}>
          <View
            style={[
              styles.statusDot,
              ragReady ? styles.statusReady : styles.statusFallback,
            ]}
          />
          <Text style={styles.statusText}>
            {ragReady === null
              ? "Đang kiểm tra trợ lý..."
              : ragReady
              ? "RAG/Qwen sẵn sàng"
              : "Đang dùng tri thức dự phòng"}
          </Text>
          <Text style={styles.vehicleText}>• {vehicleType}</Text>
        </View>

        <ScrollView
          ref={scrollRef}
          style={styles.messageList}
          contentContainerStyle={styles.messageContent}
          keyboardShouldPersistTaps="handled"
          keyboardDismissMode={Platform.OS === "ios" ? "interactive" : "on-drag"}
          onContentSizeChange={() => scrollRef.current?.scrollToEnd({ animated: true })}
        >
          {messages.map((message) => (
            <View
              key={message.id}
              style={[
                styles.messageRow,
                message.role === "user" && styles.messageRowUser,
              ]}
            >
              <View
                style={[
                  styles.bubble,
                  message.role === "user" ? styles.userBubble : styles.botBubble,
                  message.mode === "error" && styles.errorBubble,
                ]}
              >
                <Text
                  style={[
                    styles.messageText,
                    message.role === "user" && styles.userMessageText,
                  ]}
                >
                  {message.text}
                </Text>
                {message.role === "assistant" && message.mode && (
                  <Text style={styles.modeText}>{message.mode}</Text>
                )}
              </View>
            </View>
          ))}

          {sending && (
            <View style={styles.typingBox}>
              <ActivityIndicator size="small" color="#2563eb" />
              <Text style={styles.typingText}>Trợ lý đang tra cứu...</Text>
            </View>
          )}
        </ScrollView>

        {!keyboardVisible && (
        <View style={styles.suggestionWrap}>
          <ScrollView
            horizontal
            showsHorizontalScrollIndicator={false}
            keyboardShouldPersistTaps="handled"
          >
            {suggestions.map((suggestion, index) => (
              <TouchableOpacity
                key={`${suggestion}-${index}`}
                style={styles.suggestionChip}
                onPress={() => sendQuestion(suggestion)}
                disabled={sending}
              >
                <Text style={styles.suggestionText}>{suggestion}</Text>
              </TouchableOpacity>
            ))}
          </ScrollView>
        </View>
        )}

        <View style={[styles.composer, keyboardVisible && styles.composerKeyboard]}>
          <TextInput
            value={input}
            onChangeText={setInput}
            style={styles.input}
            placeholder="Hỏi về biển báo, mức phạt hoặc căn cứ pháp lý..."
            placeholderTextColor="#94a3b8"
            multiline
            editable={!sending}
            textAlignVertical="top"
            maxLength={600}
            onFocus={() =>
              setTimeout(() => scrollRef.current?.scrollToEnd({ animated: true }), 100)
            }
          />
          <TouchableOpacity
            style={[
              styles.sendButton,
              (!input.trim() || sending) && styles.sendButtonDisabled,
            ]}
            onPress={() => sendQuestion()}
            disabled={!input.trim() || sending}
          >
            <Text style={styles.sendText}>Gửi</Text>
          </TouchableOpacity>
        </View>

        {!keyboardVisible && (
          <Text style={styles.legalNote}>
            Thông tin xử phạt mang tính tham khảo; cần đối chiếu văn bản hiện hành khi áp dụng thực tế.
          </Text>
        )}
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safeArea: { flex: 1, backgroundColor: "#eef4fb" },
  flex: { flex: 1 },
  header: {
    flexDirection: "row",
    alignItems: "center",
    paddingHorizontal: 12,
    paddingVertical: 12,
    backgroundColor: "#fff",
    borderBottomWidth: 1,
    borderBottomColor: "#e2e8f0",
  },
  headerButton: { paddingVertical: 8, paddingRight: 8 },
  headerButtonText: { color: "#2563eb", fontWeight: "800", fontSize: 13 },
  headerCenter: { flex: 1, alignItems: "center" },
  headerTitle: { color: "#111827", fontWeight: "900", fontSize: 16 },
  headerTitleCompact: { fontSize: 14 },
  headerSubtitle: { color: "#64748b", fontSize: 11, marginTop: 2, maxWidth: 180 },
  logoutButton: {
    backgroundColor: "#fff1f2",
    borderRadius: 999,
    paddingHorizontal: 10,
    paddingVertical: 7,
  },
  logoutText: { color: "#dc2626", fontWeight: "800", fontSize: 12 },
  statusBar: {
    flexDirection: "row",
    alignItems: "center",
    paddingHorizontal: 16,
    paddingVertical: 8,
    backgroundColor: "#f8fafc",
  },
  statusDot: { width: 8, height: 8, borderRadius: 4, marginRight: 7 },
  statusReady: { backgroundColor: "#16a34a" },
  statusFallback: { backgroundColor: "#f59e0b" },
  statusText: { color: "#475569", fontSize: 11, fontWeight: "700" },
  vehicleText: { color: "#64748b", fontSize: 11, marginLeft: 4 },
  messageList: { flex: 1 },
  messageContent: { padding: 14, paddingBottom: 14 },
  messageRow: { flexDirection: "row", marginBottom: 11 },
  messageRowUser: { justifyContent: "flex-end" },
  bubble: { maxWidth: "86%", borderRadius: 17, paddingHorizontal: 14, paddingVertical: 11 },
  botBubble: { backgroundColor: "#fff", borderTopLeftRadius: 5 },
  userBubble: { backgroundColor: "#2563eb", borderTopRightRadius: 5 },
  errorBubble: { backgroundColor: "#fff1f2", borderWidth: 1, borderColor: "#fecdd3" },
  messageText: { color: "#1f2937", lineHeight: 21, fontSize: 14 },
  userMessageText: { color: "#fff" },
  modeText: {
    color: "#94a3b8",
    fontSize: 10,
    marginTop: 8,
    textTransform: "uppercase",
    fontWeight: "800",
  },
  typingBox: { flexDirection: "row", alignItems: "center", gap: 8, paddingVertical: 8 },
  typingText: { color: "#64748b", fontSize: 12 },
  suggestionWrap: {
    borderTopWidth: 1,
    borderTopColor: "#e2e8f0",
    backgroundColor: "#fff",
    paddingVertical: 8,
    paddingLeft: 10,
  },
  suggestionChip: {
    borderWidth: 1,
    borderColor: "#bfdbfe",
    backgroundColor: "#eff6ff",
    borderRadius: 999,
    paddingHorizontal: 12,
    paddingVertical: 8,
    marginRight: 8,
  },
  suggestionText: { color: "#1d4ed8", fontWeight: "700", fontSize: 12 },
  composer: {
    flexDirection: "row",
    alignItems: "flex-end",
    gap: 8,
    backgroundColor: "#fff",
    paddingHorizontal: 10,
    paddingTop: 9,
    paddingBottom: 8,
  },
  composerKeyboard: {
    borderTopWidth: 1,
    borderTopColor: "#e2e8f0",
    paddingBottom: 10,
  },
  input: {
    flex: 1,
    minHeight: 44,
    maxHeight: 110,
    borderWidth: 1,
    borderColor: "#cbd5e1",
    borderRadius: 14,
    paddingHorizontal: 12,
    paddingVertical: 10,
    color: "#111827",
    backgroundColor: "#f8fafc",
  },
  sendButton: {
    backgroundColor: "#1d4ed8",
    borderRadius: 13,
    minHeight: 44,
    paddingHorizontal: 17,
    alignItems: "center",
    justifyContent: "center",
  },
  sendButtonDisabled: { opacity: 0.45 },
  sendText: { color: "#fff", fontWeight: "900" },
  legalNote: {
    color: "#94a3b8",
    backgroundColor: "#fff",
    fontSize: 10,
    textAlign: "center",
    paddingHorizontal: 12,
    paddingTop: 6,
    paddingBottom: 9,
  },
});