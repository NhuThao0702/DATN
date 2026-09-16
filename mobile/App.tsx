import React, { useEffect, useRef, useState } from "react";
import {
  ActivityIndicator,
  Alert,
  Image,
  SafeAreaView,
  ScrollView,
  StyleSheet,
  Text,
  TouchableOpacity,
  useWindowDimensions,
  View,
} from "react-native";
import * as ImagePicker from "expo-image-picker";
import * as ImageManipulator from "expo-image-manipulator";
import * as Speech from "expo-speech";
import { CameraView, type CameraType, useCameraPermissions } from "expo-camera";
import AuthScreen, { type AuthUser } from "./src/screens/AuthScreen";
import LegalChatScreen from "./src/screens/LegalChatScreen";

const API_URL = "http://192.168.1.4:5000";

type VehicleType = "all" | "xe_may" | "oto" | "xe_tai" | "xe_khach";

type Detection = {
  cls_id: number;
  code: string;
  label: string;
  group: string;
  name: string;
  warning: string;
  conf: number;
  box: number[];
  ocr_text?: string;
};

type LiveTrack = Detection & {
  age: number;
  missing: number;
};

type TimingInfo = {
  prepareMs: number;
  requestMs: number;
  inferenceMs: number;
  backendTotalMs: number;
  payloadKb: number;
};

type CameraMode = "photo" | "journey";

type JourneyEvent = {
  id: string;
  time: string;
  codes: string;
  text: string;
};

const LIVE_INTERVAL_MS = 120;
const LIVE_CONF = 0.25;
const LIVE_IMGSZ = 512;
const LIVE_MAX_SIDE = 720;
const LIVE_JPEG_QUALITY = 0.50;
const LIVE_IMMEDIATE_CONF = 0.68;
const LIVE_CONFIRM_WINDOW_MS = 3500;
const LIVE_AUDIO_COOLDOWN_MS = 7000;

// Chỉ dùng cho CAMERA HÀNH TRÌNH. Ảnh chọn từ máy và ảnh chụp tĩnh giữ nguyên.
// Camera xe máy rung nên box phải bám frame mới nhanh hơn ảnh/video tĩnh.
const LIVE_DISPLAY_MIN_CONF = 0.32;
const LIVE_TRACK_ALPHA = 0.58;
const LIVE_TRACK_IOU = 0.18;
const LIVE_TRACK_CENTER_GATE = 0.80;
const LIVE_TRACK_TTL = 1;
const LIVE_MIN_BOX_SIDE = 14;
const LIVE_MIN_AGE_TO_SHOW = 2;
const LIVE_MAX_TRACKS = 12;
const LIVE_CAPTURE_TIMEOUT_MS = 2600;

const VEHICLES: Array<{ value: VehicleType; label: string; icon: string }> = [
  { value: "all", label: "Tất cả", icon: "🚦" },
  { value: "xe_may", label: "Xe máy", icon: "🏍️" },
  { value: "oto", label: "Ô tô", icon: "🚗" },
  { value: "xe_tai", label: "Xe tải", icon: "🚚" },
  { value: "xe_khach", label: "Xe khách", icon: "🚌" },
];

export default function App() {
  const { width: screenWidth, height: screenHeight } = useWindowDimensions();
  const isCompactPhone = screenWidth < 360;
  const pagePadding = screenWidth < 360 ? 12 : screenWidth < 430 ? 16 : 20;
  const cameraHeight = Math.round(
    Math.min(screenHeight * 0.58, Math.max(300, screenWidth * 1.02))
  );
  const resultImageHeight = Math.round(
    Math.min(screenHeight * 0.50, Math.max(250, screenWidth * 0.88))
  );
  const [authToken, setAuthToken] = useState("");
  const [authUser, setAuthUser] = useState<AuthUser | null>(null);
  const [imageUri, setImageUri] = useState<string | null>(null);
  const [annotatedImage, setAnnotatedImage] = useState<string | null>(null);
  const [detections, setDetections] = useState<Detection[]>([]);
  const [audioText, setAudioText] = useState("");
  const [loading, setLoading] = useState(false);
  const [loadingText, setLoadingText] = useState("YOLO đang nhận diện...");
  const [timing, setTiming] = useState<TimingInfo | null>(null);
  const [vehicleType, setVehicleType] = useState<VehicleType>("all");
  const [activeScreen, setActiveScreen] = useState<"detect" | "chat">(
  "detect"
);
  const [audioEnabled, setAudioEnabled] = useState(true);
  const [cameraOpen, setCameraOpen] = useState(false);
  const [cameraMode, setCameraMode] = useState<CameraMode>("photo");
  const [cameraReady, setCameraReady] = useState(false);
  const [livePictureSize, setLivePictureSize] = useState<string | undefined>(undefined);
  const [journeyRunning, setJourneyRunning] = useState(false);
  const [journeyStatus, setJourneyStatus] = useState("Sẵn sàng");
  const [journeyHistory, setJourneyHistory] = useState<JourneyEvent[]>([]);
  const [journeyFrameCount, setJourneyFrameCount] = useState(0);
  const [journeyFrameSize, setJourneyFrameSize] = useState({ width: 0, height: 0 });
  const [cameraPreviewSize, setCameraPreviewSize] = useState({ width: 0, height: 0 });
  const [facing, setFacing] = useState<CameraType>("back");
  const [permission, requestPermission] = useCameraPermissions();
  const cameraRef = useRef<CameraView | null>(null);

  const journeyRunningRef = useRef(false);
  const journeyBusyRef = useRef(false);
  const journeyTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const journeyHitsRef = useRef<Record<string, { hits: number; lastSeen: number }>>({});
  const spokenAtByCodeRef = useRef<Record<string, number>>({});
  const journeyTracksRef = useRef<LiveTrack[]>([]);
  const journeySpeechBusyRef = useRef(false);
  const latestJourneySpeechDetectionsRef = useRef<Detection[]>([]);

  const handleAuthenticated = (token: string, user: AuthUser) => {
    setAuthToken(token);
    setAuthUser(user);
    setActiveScreen("detect");
  };

  const handleLogout = async () => {
    try {
      if (authToken) {
        await fetch(`${API_URL}/api/mobile/logout`, {
          method: "POST",
          headers: { Authorization: `Bearer ${authToken}` },
        });
      }
    } catch (error) {
      console.log("Logout request error:", error);
    } finally {
      journeyRunningRef.current = false;
      if (journeyTimerRef.current) clearTimeout(journeyTimerRef.current);
      await Speech.stop();
      setCameraOpen(false);
      setJourneyRunning(false);
      setAuthToken("");
      setAuthUser(null);
      setActiveScreen("detect");
      resetResult();
    }
  };

  useEffect(() => {
    return () => {
      journeyRunningRef.current = false;
      if (journeyTimerRef.current) {
        clearTimeout(journeyTimerRef.current);
      }
      Speech.stop();
    };
  }, []);

  const resetResult = () => {
    setAnnotatedImage(null);
    setDetections([]);
    setAudioText("");
    setTiming(null);
  };

  const normalizeAnnotatedImage = (value?: string | null) => {
    if (!value) return null;
    if (
      value.startsWith("data:") ||
      value.startsWith("file:") ||
      value.startsWith("http")
    ) {
      return value;
    }
    return `data:image/jpeg;base64,${value}`;
  };

  const speakWarning = async (text: string) => {
    if (!audioEnabled || !text.trim()) return;
    try {
      await Speech.stop();
      Speech.speak(text, {
        language: "vi-VN",
        rate: 1.08,
        pitch: 1.0,
      });
    } catch (error) {
      console.log("TTS error:", error);
    }
  };

  const getImageSize = (uri: string) =>
    new Promise<{ width: number; height: number }>((resolve, reject) => {
      Image.getSize(
        uri,
        (width, height) => resolve({ width, height }),
        reject
      );
    });

  const prepareImage = async (
    uri: string,
    maxSide = 960,
    compress = 0.62
  ) => {
    const started = Date.now();
    const { width, height } = await getImageSize(uri);
    const actions: any[] = [];

    if (Math.max(width, height) > maxSide) {
      if (width >= height) {
        actions.push({ resize: { width: maxSide } });
      } else {
        actions.push({ resize: { height: maxSide } });
      }
    }

    const result = await ImageManipulator.manipulateAsync(
      uri,
      actions,
      {
        compress,
        format: ImageManipulator.SaveFormat.JPEG,
        base64: true,
      }
    );

    if (!result.base64) {
      throw new Error("Không tạo được dữ liệu ảnh tối ưu.");
    }

    return {
      uri: result.uri,
      base64: result.base64,
      prepareMs: Date.now() - started,
      payloadKb: Math.round((result.base64.length * 3) / 4 / 1024),
    };
  };

  const prepareLivePictureRef = async (pictureRef: any) => {
    const started = Date.now();

    const width = Number(pictureRef?.width || 0);
    const height = Number(pictureRef?.height || 0);

    const context = ImageManipulator.ImageManipulator.manipulate(pictureRef);

    if (Math.max(width, height) > LIVE_MAX_SIDE && width > 0 && height > 0) {
      if (width >= height) {
        context.resize({ width: LIVE_MAX_SIDE, height: null });
      } else {
        context.resize({ width: null, height: LIVE_MAX_SIDE });
      }
    }

    const rendered = await context.renderAsync();
    const result = await rendered.saveAsync({
      compress: LIVE_JPEG_QUALITY,
      format: ImageManipulator.SaveFormat.JPEG,
      base64: true,
    });

    if (!result.base64) {
      throw new Error("Không tạo được Base64 từ frame live.");
    }

    return {
      base64: result.base64,
      prepareMs: Date.now() - started,
      payloadKb: Math.round((result.base64.length * 3) / 4 / 1024),
    };
  };

  const chooseLivePictureSize = (sizes: string[]) => {
    const parsed = sizes
      .map((value) => {
        const match = value.match(/^(\d+)x(\d+)$/i);
        if (!match) return null;
        const width = Number(match[1]);
        const height = Number(match[2]);
        return {
          value,
          width,
          height,
          area: width * height,
          maxSide: Math.max(width, height),
        };
      })
      .filter(Boolean) as Array<{
        value: string;
        width: number;
        height: number;
        area: number;
        maxSide: number;
      }>;

    if (parsed.length === 0) return undefined;

    const reasonable = parsed
      .filter((item) => item.maxSide >= 960 && item.maxSide <= 1400)
      .sort((a, b) => a.area - b.area);

    if (reasonable.length > 0) return reasonable[0].value;

    const targetArea = 1280 * 720;
    return parsed.sort(
      (a, b) => Math.abs(a.area - targetArea) - Math.abs(b.area - targetArea)
    )[0]?.value;
  };

  const handleCameraReady = async () => {
    setCameraReady(true);

    if (cameraMode !== "journey" || !cameraRef.current) return;

    try {
      const sizes = await cameraRef.current.getAvailablePictureSizesAsync();
      const selected = chooseLivePictureSize(sizes || []);
      if (selected) {
        setLivePictureSize(selected);
        setJourneyStatus(`Camera sẵn sàng • ${selected}`);
      } else {
        setJourneyStatus("Camera sẵn sàng");
      }
    } catch (error) {
      console.log("Picture size error:", error);
      setJourneyStatus("Camera sẵn sàng");
    }
  };

  const livePriority = (item: Detection) => {
    if (item.code === "den_do") return 0;
    if (item.code.startsWith("P.")) return 1;
    if (item.code.startsWith("W.")) return 2;
    if (item.code.startsWith("R.")) return 3;
    return 4;
  };

  const liveBoxArea = (box: number[]) => {
    if (!box || box.length < 4) return 0;
    return Math.max(0, box[2] - box[0]) * Math.max(0, box[3] - box[1]);
  };

  const liveBoxIoU = (a: number[], b: number[]) => {
    if (!a || !b || a.length < 4 || b.length < 4) return 0;
    const x1 = Math.max(a[0], b[0]);
    const y1 = Math.max(a[1], b[1]);
    const x2 = Math.min(a[2], b[2]);
    const y2 = Math.min(a[3], b[3]);
    const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
    const union = liveBoxArea(a) + liveBoxArea(b) - inter;
    return union <= 0 ? 0 : inter / union;
  };

  const liveBoxCenterDistance = (a: number[], b: number[]) => {
    const acx = (a[0] + a[2]) / 2;
    const acy = (a[1] + a[3]) / 2;
    const bcx = (b[0] + b[2]) / 2;
    const bcy = (b[1] + b[3]) / 2;
    return Math.hypot(acx - bcx, acy - bcy);
  };

  const liveBoxDiagonal = (box: number[]) =>
    Math.max(1, Math.hypot(box[2] - box[0], box[3] - box[1]));

  const smoothLiveBox = (oldBox: number[], newBox: number[], alpha: number) =>
    oldBox.map((value, index) => value * (1 - alpha) + newBox[index] * alpha);

  const updateJourneyStableDetections = (rawDetections: Detection[]) => {
    const cleanDetections = (rawDetections || [])
      .filter((item) => item && Array.isArray(item.box) && item.box.length === 4)
      .filter((item) => Number(item.conf || 0) >= LIVE_DISPLAY_MIN_CONF)
      .filter((item) => {
        const [x1, y1, x2, y2] = item.box.map(Number);
        const boxWidth = x2 - x1;
        const boxHeight = y2 - y1;
        return boxWidth >= LIVE_MIN_BOX_SIDE && boxHeight >= LIVE_MIN_BOX_SIDE;
      })
      .sort((a, b) => Number(b.conf || 0) - Number(a.conf || 0));

    const oldTracks = journeyTracksRef.current;
    const usedTrackIndexes = new Set<number>();
    const nextTracks: LiveTrack[] = [];

    cleanDetections.forEach((item) => {
      let bestIndex = -1;
      let bestScore = -Infinity;
      let bestIoU = 0;

      oldTracks.forEach((track, index) => {
        if (usedTrackIndexes.has(index)) return;
        if (String(track.code || "") !== String(item.code || "")) return;

        const iou = liveBoxIoU(track.box, item.box);
        const distance = liveBoxCenterDistance(track.box, item.box);
        const normalizer = Math.max(liveBoxDiagonal(track.box), liveBoxDiagonal(item.box));
        const normalizedDistance = distance / normalizer;

        // Không ghép nhầm hai biển cùng class nhưng nằm xa nhau.
        if (iou < LIVE_TRACK_IOU && normalizedDistance > LIVE_TRACK_CENTER_GATE) return;

        const score = iou - normalizedDistance * 0.20;
        if (score > bestScore) {
          bestScore = score;
          bestIndex = index;
          bestIoU = iou;
        }
      });

      if (bestIndex >= 0) {
        const oldTrack = oldTracks[bestIndex];
        usedTrackIndexes.add(bestIndex);

        // Box ổn định -> alpha nhỏ để giảm rung; chuyển động nhanh -> bám box mới nhanh hơn.
        const alpha = bestIoU >= 0.55
          ? LIVE_TRACK_ALPHA
          : 0.82;

        nextTracks.push({
          ...item,
          box: smoothLiveBox(oldTrack.box, item.box.map(Number), alpha),
          conf: Number(item.conf || 0) * 0.70 + Number(oldTrack.conf || 0) * 0.30,
          age: (oldTrack.age || 0) + 1,
          missing: 0,
        });
      } else {
        nextTracks.push({
          ...item,
          box: item.box.map(Number),
          age: 1,
          missing: 0,
        });
      }
    });

    // Track mất tạm thời chỉ giữ trong bộ nhớ để có thể ghép lại ở 1-2 nhịp kế tiếp.
    // Không vẽ track đang missing, nhờ đó box không "treo" lại khi biển đã đi qua.
    oldTracks.forEach((track, index) => {
      if (usedTrackIndexes.has(index)) return;
      const missing = (track.missing || 0) + 1;
      if (missing <= LIVE_TRACK_TTL) {
        nextTracks.push({
          ...track,
          missing,
          conf: Number(track.conf || 0) * 0.80,
        });
      }
    });

    journeyTracksRef.current = nextTracks
      .sort((a, b) => Number(b.conf || 0) - Number(a.conf || 0))
      .slice(0, LIVE_MAX_TRACKS);

    return journeyTracksRef.current
      .filter((track) =>
        track.age >= LIVE_MIN_AGE_TO_SHOW || Number(track.conf || 0) >= LIVE_IMMEDIATE_CONF
      )
      .filter((track) => track.missing === 0)
      .map((track) => ({
        ...track,
        box: track.box.map((value) => Math.round(value)),
      }));
  };

  const getJourneyBoxStyle = (box: number[]) => {
    const sourceWidth = journeyFrameSize.width;
    const sourceHeight = journeyFrameSize.height;
    const displayWidth = cameraPreviewSize.width;
    const displayHeight = cameraPreviewSize.height;

    if (
      !box || box.length < 4 ||
      sourceWidth <= 0 || sourceHeight <= 0 ||
      displayWidth <= 0 || displayHeight <= 0
    ) {
      return null;
    }

    // CameraView hiển thị preview theo kiểu cover. Tính cả phần crop hai cạnh để box
    // bám đúng vị trí trên điện thoại dọc thay vì scale thẳng toàn khung.
    const scale = Math.max(displayWidth / sourceWidth, displayHeight / sourceHeight);
    const renderedWidth = sourceWidth * scale;
    const renderedHeight = sourceHeight * scale;
    const offsetX = (displayWidth - renderedWidth) / 2;
    const offsetY = (displayHeight - renderedHeight) / 2;

    const x1 = box[0] * scale + offsetX;
    const y1 = box[1] * scale + offsetY;
    const x2 = box[2] * scale + offsetX;
    const y2 = box[3] * scale + offsetY;

    const left = Math.max(0, Math.min(displayWidth, x1));
    const top = Math.max(0, Math.min(displayHeight, y1));
    const right = Math.max(0, Math.min(displayWidth, x2));
    const bottom = Math.max(0, Math.min(displayHeight, y2));

    if (right - left < 2 || bottom - top < 2) return null;

    return {
      left,
      top,
      width: right - left,
      height: bottom - top,
    };
  };

  const liveApproachScore = (item: Detection) => {
    const box = item.box || [];
    const area = liveBoxArea(box);
    const y2 = Number(box[3] || 0);
    // Biển thấp hơn trong khung + box lớn hơn thường là biển đang tới gần trước.
    return y2 * 2 + Math.sqrt(Math.max(0, area)) + Number(item.conf || 0) * 100;
  };

  const buildSingleLiveSpeech = (item: Detection) => {
    const warning = String(item.warning || "").trim();
    if (warning) {
      return warning
        .replace(/\s+ở phía trước\.?$/i, " phía trước.")
        .replace(/phía trước\s+phía trước/gi, "phía trước");
    }
    const name = String(item.name || item.label || item.code || "Biển báo").trim();
    return `${name} phía trước.`;
  };

  const maybeSpeakJourneyCurrent = () => {
    if (!audioEnabled || journeySpeechBusyRef.current || !journeyRunningRef.current) return;

    const now = Date.now();
    const candidate = latestJourneySpeechDetectionsRef.current
      .filter((item) => {
        const code = String(item.code || item.label || "unknown");
        const lastSpoken = spokenAtByCodeRef.current[code] || 0;
        return now - lastSpoken >= LIVE_AUDIO_COOLDOWN_MS;
      })
      .sort((a, b) => liveApproachScore(b) - liveApproachScore(a))[0];

    if (!candidate) return;

    const code = String(candidate.code || candidate.label || "unknown");
    const text = buildSingleLiveSpeech(candidate);
    if (!text.trim()) return;

    spokenAtByCodeRef.current[code] = now;
    journeySpeechBusyRef.current = true;

    const event: JourneyEvent = {
      id: `${now}-${code}`,
      time: new Date().toLocaleTimeString("vi-VN", {
        hour: "2-digit",
        minute: "2-digit",
        second: "2-digit",
      }),
      codes: code,
      text,
    };
    setJourneyHistory((items) => [event, ...items].slice(0, 16));

    try {
      Speech.speak(text, {
        language: "vi-VN",
        rate: 1.22,
        pitch: 1.0,
        onDone: () => {
          journeySpeechBusyRef.current = false;
          setTimeout(maybeSpeakJourneyCurrent, 80);
        },
        onStopped: () => {
          journeySpeechBusyRef.current = false;
        },
        onError: () => {
          journeySpeechBusyRef.current = false;
        },
      });
    } catch (error) {
      journeySpeechBusyRef.current = false;
      console.log("Live TTS error:", error);
    }
  };

  const detectImageUri = async (uri: string) => {
    try {
      setLoading(true);
      setLoadingText("Đang tối ưu ảnh...");

      const prepared = await prepareImage(uri);
      setImageUri(prepared.uri);

      setLoadingText("YOLO đang nhận diện...");
      await detectImage(prepared.base64, prepared.prepareMs, prepared.payloadKb);
    } catch (error: any) {
      console.log(error);
      Alert.alert(
        "Lỗi xử lý ảnh",
        error?.message || "Không thể tối ưu ảnh trước khi nhận diện."
      );
      setLoading(false);
    }
  };

  const detectImage = async (
    base64: string,
    prepareMs = 0,
    payloadKb = 0
  ) => {
    try {
      await Speech.stop();
      const requestStarted = Date.now();

      const response = await fetch(`${API_URL}/api/mobile/detect`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${authToken}`,
        },
        body: JSON.stringify({
          image: `data:image/jpeg;base64,${base64}`,
          conf: 0.35,
          imgsz: 512,
          vehicle_type: vehicleType,
        }),
      });

      const text = await response.text();
      let data: any;

      try {
        data = JSON.parse(text);
      } catch {
        throw new Error("Backend không trả JSON hợp lệ.");
      }

      if (response.status === 401) {
        await handleLogout();
        throw new Error("Phiên đăng nhập đã hết hạn. Vui lòng đăng nhập lại.");
      }

      if (!response.ok) {
        throw new Error(data.error || "Backend xử lý thất bại.");
      }

      const resultDetections: Detection[] = data.detections || [];
      const resultAudio = data.audio_text || "";

      setDetections(resultDetections);
      setAudioText(resultAudio);
      setAnnotatedImage(normalizeAnnotatedImage(data.annotated_image));

      setTiming({
        prepareMs,
        requestMs: Date.now() - requestStarted,
        inferenceMs: Number(data.inference_ms || 0),
        backendTotalMs: Number(data.total_ms || 0),
        payloadKb,
      });

      if (resultAudio) {
        await speakWarning(resultAudio);
      }
    } catch (error: any) {
      console.log(error);
      Alert.alert(
        "Không kết nối được Backend",
        error?.message || "Kiểm tra Flask, IP laptop và Wi-Fi."
      );
    } finally {
      setLoading(false);
    }
  };

  const pickImage = async () => {
    try {
      setCameraOpen(false);
      const result = await ImagePicker.launchImageLibraryAsync({
        mediaTypes: ["images"],
        quality: 1,
        base64: false,
      });

      if (result.canceled) return;
      const asset = result.assets[0];
      resetResult();
      await detectImageUri(asset.uri);
    } catch (error) {
      console.log(error);
      Alert.alert("Lỗi", "Không thể chọn ảnh.");
    }
  };

  const openCamera = async (mode: CameraMode = "photo") => {
    try {
      if (!permission?.granted) {
        const result = await requestPermission();
        if (!result.granted) {
          Alert.alert(
            "Chưa có quyền camera",
            "Hãy cho phép ứng dụng sử dụng camera để chụp ảnh nhận diện."
          );
          return;
        }
      }

      await Speech.stop();
      setCameraMode(mode);
      setCameraReady(false);
      setCameraOpen(true);
      if (mode === "journey") {
        setJourneyStatus("Camera đã mở - bấm Bắt đầu hành trình");
      }
    } catch (error) {
      console.log(error);
      Alert.alert("Lỗi", "Không thể mở camera.");
    }
  };

  const takePhoto = async () => {
    try {
      if (!cameraRef.current) return;

      const photo = await cameraRef.current.takePictureAsync({
        quality: 0.9,
        base64: false,
        skipProcessing: false,
      });

      if (!photo) return;

      setCameraOpen(false);
      resetResult();
      await detectImageUri(photo.uri);
    } catch (error) {
      console.log(error);
      Alert.alert("Lỗi camera", "Không thể chụp ảnh. Hãy thử lại.");
    }
  };

  const processJourneyFrame = async () => {
    if (
      !journeyRunningRef.current ||
      journeyBusyRef.current ||
      !cameraRef.current ||
      !cameraReady
    ) {
      return;
    }

    journeyBusyRef.current = true;

    try {
      setJourneyStatus("Đang quét biển báo...");

      let prepared: { base64: string; prepareMs: number; payloadKb: number };

      // Trên một số Android, PictureRef có thể treo ở lần chụp đầu tiên mà không throw.
      // Hành trình vì vậy dùng takePictureAsync(base64=true) - cùng cơ chế camera đã
      // chứng minh hoạt động ở nút Chụp, nhưng bỏ bước ghi/đọc/manipulate file.
      const captureStarted = Date.now();
      const capturePromise = cameraRef.current.takePictureAsync({
        quality: LIVE_JPEG_QUALITY,
        base64: true,
        skipProcessing: true,
        shutterSound: false,
      });
      const timeoutPromise = new Promise<never>((_, reject) => {
        setTimeout(
          () => reject(new Error("Camera chưa trả frame live sau 2.6 giây.")),
          LIVE_CAPTURE_TIMEOUT_MS
        );
      });

      const photo: any = await Promise.race([capturePromise, timeoutPromise]);
      if (!photo || !journeyRunningRef.current) return;

      if (photo.base64) {
        prepared = {
          base64: String(photo.base64),
          prepareMs: Date.now() - captureStarted,
          payloadKb: Math.round((String(photo.base64).length * 3) / 4 / 1024),
        };
      } else if (photo.uri) {
        // Fallback hiếm: thiết bị không trả base64 dù đã yêu cầu.
        prepared = await prepareImage(photo.uri, LIVE_MAX_SIDE, LIVE_JPEG_QUALITY);
      } else {
        throw new Error("Camera không trả dữ liệu frame live.");
      }

      const requestStarted = Date.now();
      const response = await fetch(`${API_URL}/api/mobile/live-detect`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${authToken}`,
        },
        body: JSON.stringify({
          image: `data:image/jpeg;base64,${prepared.base64}`,
          conf: LIVE_CONF,
          imgsz: LIVE_IMGSZ,
          vehicle_type: vehicleType,
        }),
      });

      const text = await response.text();
      let data: any;

      try {
        data = JSON.parse(text);
      } catch {
        throw new Error("Backend live không trả JSON hợp lệ.");
      }

      if (response.status === 401) {
        journeyRunningRef.current = false;
        setJourneyRunning(false);
        await handleLogout();
        throw new Error("Phiên đăng nhập đã hết hạn.");
      }

      if (!response.ok) {
        throw new Error(data.error || "Live detection thất bại.");
      }

      const resultDetections: Detection[] = data.detections || [];
      const resultAudio = String(data.audio_text || "");
      const now = Date.now();
      const frameWidth = Number(data.frame_width || 0);
      const frameHeight = Number(data.frame_height || 0);

      if (frameWidth > 0 && frameHeight > 0) {
        setJourneyFrameSize({ width: frameWidth, height: frameHeight });
      }

      // Chỉ phần Hành trình dùng tracking/smoothing.
      // Ảnh upload/chụp tĩnh vẫn đi qua detectImage() như cũ.
      const visibleDetections = updateJourneyStableDetections(resultDetections);

      setJourneyFrameCount((value) => value + 1);
      setDetections(visibleDetections);
      setAudioText(resultAudio);
      setTiming({
        prepareMs: prepared.prepareMs,
        requestMs: Date.now() - requestStarted,
        inferenceMs: Number(data.inference_ms || 0),
        backendTotalMs: Number(data.total_ms || 0),
        payloadKb: prepared.payloadKb,
      });

      // Live ưu tiên bắt biển thoáng qua:
      // - conf >= 0.60: cảnh báo ngay ở lần thấy đầu tiên.
      // - conf 0.32..0.60: cần thấy 2 lần trong cửa sổ 4 giây.
      // Không bắt buộc hai frame phải liền nhau.
      resultDetections.forEach((item) => {
        const old = journeyHitsRef.current[item.code];
        const withinWindow =
          old && now - old.lastSeen <= LIVE_CONFIRM_WINDOW_MS;

        journeyHitsRef.current[item.code] = {
          hits: withinWindow ? Math.min(old.hits + 1, 4) : 1,
          lastSeen: now,
        };
      });

      Object.keys(journeyHitsRef.current).forEach((code) => {
        if (now - journeyHitsRef.current[code].lastSeen > 6000) {
          delete journeyHitsRef.current[code];
        }
      });

      const stableDetections = resultDetections.filter((item) => {
        const hits = journeyHitsRef.current[item.code]?.hits || 0;
        return item.conf >= LIVE_IMMEDIATE_CONF || hits >= 2;
      });

      // TTS không xếp hàng câu cũ. Luôn giữ danh sách biển ĐANG CÒN trong frame mới
      // nhất, rồi chọn biển gần xe hơn (box thấp/lớn hơn) để đọc trước. Nếu lúc đọc
      // xong biển tiếp theo đã biến mất thì nó sẽ tự bị loại, không đọc muộn sau khi qua.
      latestJourneySpeechDetectionsRef.current = stableDetections;
      maybeSpeakJourneyCurrent();

      setJourneyStatus(
        visibleDetections.length > 0
          ? `Ổn định ${visibleDetections.length} biển • đang quét nhanh...`
          : resultDetections.length > 0
          ? "Đang xác nhận box..."
          : "Đang chạy • quét nhanh"
      );
    } catch (error: any) {
      console.log("Journey error:", error);
      setJourneyStatus(error?.message || "Lỗi quét hành trình");
    } finally {
      journeyBusyRef.current = false;

      if (journeyRunningRef.current) {
        journeyTimerRef.current = setTimeout(
          processJourneyFrame,
          LIVE_INTERVAL_MS
        );
      }
    }
  };

  const startJourney = async () => {
    if (!cameraReady) {
      Alert.alert("Camera chưa sẵn sàng", "Đợi camera mở xong rồi thử lại.");
      return;
    }

    await Speech.stop();
    journeyHitsRef.current = {};
    spokenAtByCodeRef.current = {};
    journeyTracksRef.current = [];
    latestJourneySpeechDetectionsRef.current = [];
    journeySpeechBusyRef.current = false;
    setJourneyFrameSize({ width: 0, height: 0 });
    setJourneyFrameCount(0);
    setJourneyHistory([]);
    setDetections([]);
    setAudioText("");
    setTiming(null);
    setJourneyRunning(true);
    journeyRunningRef.current = true;
    setJourneyStatus("Bắt đầu quét hành trình...");

    setTimeout(processJourneyFrame, 120);
  };

  const stopJourney = async () => {
    journeyRunningRef.current = false;
    setJourneyRunning(false);
    journeyBusyRef.current = false;
    journeyTracksRef.current = [];
    latestJourneySpeechDetectionsRef.current = [];
    journeySpeechBusyRef.current = false;
    setDetections([]);
    setJourneyFrameSize({ width: 0, height: 0 });

    if (journeyTimerRef.current) {
      clearTimeout(journeyTimerRef.current);
      journeyTimerRef.current = null;
    }

    await Speech.stop();
    setJourneyStatus("Đã dừng hành trình");
  };

  const closeCamera = async () => {
    if (journeyRunningRef.current) {
      await stopJourney();
    }
    setCameraOpen(false);
    setCameraReady(false);
  };

  const toggleFacing = () => {
    setLivePictureSize(undefined);
    setFacing((current) => (current === "back" ? "front" : "back"));
  };

  const toggleAudio = async () => {
    const next = !audioEnabled;
    setAudioEnabled(next);
    if (!next) await Speech.stop();
  };

  if (!authToken || !authUser) {
    return (
      <AuthScreen
        apiUrl={API_URL}
        onAuthenticated={handleAuthenticated}
      />
    );
  }

  if (activeScreen === "chat") {
    return (
      <LegalChatScreen
        apiUrl={API_URL}
        authToken={authToken}
        username={authUser.username}
        vehicleType={vehicleType}
        contextCode={detections[0]?.code || ""}
        onBack={() => setActiveScreen("detect")}
        onUnauthorized={handleLogout}
        onLogout={handleLogout}
      />
    );
  }

  return (
    <SafeAreaView style={styles.safeArea}>
      <ScrollView
        style={styles.screen}
        contentContainerStyle={[styles.container, { paddingHorizontal: pagePadding }]}
        keyboardShouldPersistTaps="handled"
        showsVerticalScrollIndicator={false}
      >
      <View style={styles.accountBar}>
        <Text style={styles.accountText}>👤 {authUser.username}</Text>
        <TouchableOpacity style={styles.logoutButton} onPress={handleLogout}>
          <Text style={styles.logoutText}>Đăng xuất</Text>
        </TouchableOpacity>
      </View>
      <Text style={[styles.title, isCompactPhone && styles.titleCompact]}>Trợ lý biển báo giao thông</Text>
      <Text style={styles.subtitle}>Nhận diện biển báo bằng YOLOv8</Text>
      <TouchableOpacity
        style={styles.chatEntryButton}
        onPress={() => setActiveScreen("chat")}
>
      <Text style={styles.chatEntryText}> Mở trợ lý luật giao thông</Text>
      <Text style={styles.chatEntryHint}>
        {detections[0]?.code
        ? `Hỏi thêm về biển ${detections[0].code}`
        : "Tra cứu biển báo, mức phạt và quy định"}
      </Text>
    </TouchableOpacity>

      <View style={styles.card}>
        <Text style={styles.sectionTitle}>1. Chọn loại phương tiện</Text>
        <Text style={styles.helpText}>
          Cảnh báo được điều chỉnh theo phương tiện đã chọn.
        </Text>

        <View style={styles.vehicleWrap}>
          {VEHICLES.map((item) => {
            const active = item.value === vehicleType;
            return (
              <TouchableOpacity
                key={item.value}
                style={[styles.vehicleChip, active && styles.vehicleChipActive]}
                onPress={() => setVehicleType(item.value)}
                disabled={loading}
              >
                <Text
                  style={[styles.vehicleText, active && styles.vehicleTextActive]}
                >
                  {item.icon} {item.label}
                </Text>
              </TouchableOpacity>
            );
          })}
        </View>
      </View>

      <View style={styles.card}>
        <View style={styles.rowBetween}>
          <View style={{ flex: 1 }}>
            <Text style={styles.sectionTitle}>2. Nhận diện</Text>
            <Text style={styles.helpText}>
              Chọn ảnh có sẵn hoặc chụp trực tiếp bằng camera.
            </Text>
          </View>

          <TouchableOpacity
            style={[styles.soundButton, audioEnabled && styles.soundButtonActive]}
            onPress={toggleAudio}
          >
            <Text style={styles.soundText}>
              {audioEnabled ? "🔊 Bật" : "🔇 Tắt"}
            </Text>
          </TouchableOpacity>
        </View>

        <View style={[styles.actionRow, isCompactPhone && styles.actionRowCompact]}>
          <TouchableOpacity
            style={[styles.primaryButton, styles.actionButton, isCompactPhone && styles.actionButtonCompact]}
            onPress={pickImage}
            disabled={loading || journeyRunning}
          >
            <Text style={styles.buttonText}>🖼️ Ảnh</Text>
          </TouchableOpacity>

          <TouchableOpacity
            style={[styles.cameraButton, styles.actionButton, isCompactPhone && styles.actionButtonCompact]}
            onPress={() => openCamera("photo")}
            disabled={loading || journeyRunning}
          >
            <Text style={styles.buttonText}>📷 Chụp</Text>
          </TouchableOpacity>

          <TouchableOpacity
            style={[styles.journeyButton, styles.actionButton, isCompactPhone && styles.actionButtonCompact]}
            onPress={() => openCamera("journey")}
            disabled={loading}
          >
            <Text style={styles.buttonText}>🚘 Hành trình</Text>
          </TouchableOpacity>
        </View>
      </View>

      {cameraOpen && (
        <View style={styles.cameraCard}>
          <View
            style={styles.cameraWrap}
            onLayout={(event) => {
              const { width, height } = event.nativeEvent.layout;
              setCameraPreviewSize({ width, height: Math.min(height, cameraHeight) });
            }}
          >
            <CameraView
              ref={cameraRef}
              style={[styles.camera, { height: cameraHeight }]}
              facing={facing}
              pictureSize={cameraMode === "journey" ? livePictureSize : undefined}
              onCameraReady={handleCameraReady}
            />

            {cameraMode === "journey" && detections.length > 0 && (
              <View style={styles.liveBoxLayer} pointerEvents="none">
                {detections.map((item, index) => {
                  const boxStyle = getJourneyBoxStyle(item.box);
                  if (!boxStyle) return null;

                  return (
                    <View
                      key={`live-box-${item.code}-${index}`}
                      style={[styles.liveDetectionBox, boxStyle]}
                    >
                      <Text style={styles.liveDetectionLabel} numberOfLines={1}>
                        {item.code} {(Number(item.conf || 0) * 100).toFixed(0)}%
                      </Text>
                    </View>
                  );
                })}
              </View>
            )}

            {cameraMode === "journey" && (
              <View style={styles.liveBadge}>
                <View
                  style={[
                    styles.liveDot,
                    journeyRunning ? styles.liveDotOn : styles.liveDotOff,
                  ]}
                />
                <Text style={styles.liveBadgeText}>
                  {journeyRunning ? "LIVE" : "HÀNH TRÌNH"}
                </Text>
              </View>
            )}

            {cameraMode === "journey" && detections.length > 0 && (
              <View style={styles.liveResultOverlay}>
                {detections.slice(0, 3).map((item, index) => (
                  <Text
                    key={`${item.code}-${index}`}
                    style={styles.liveResultText}
                  >
                    {item.code} • {(item.conf * 100).toFixed(0)}% • {item.name || item.label}
                  </Text>
                ))}
              </View>
            )}
          </View>

          {cameraMode === "photo" ? (
            <View style={styles.cameraControls}>
              <TouchableOpacity style={styles.smallButton} onPress={toggleFacing}>
                <Text style={styles.smallButtonText}>🔄 Đổi camera</Text>
              </TouchableOpacity>

              <TouchableOpacity
                style={styles.captureButton}
                onPress={takePhoto}
                disabled={loading || !cameraReady}
              >
                <Text style={styles.captureButtonText}>📸 Chụp</Text>
              </TouchableOpacity>

              <TouchableOpacity style={styles.smallButton} onPress={closeCamera}>
                <Text style={styles.smallButtonText}>✕ Đóng</Text>
              </TouchableOpacity>
            </View>
          ) : (
            <>
              <View style={styles.journeyInfoBar}>
                <Text style={styles.journeyInfoText}>{journeyStatus}</Text>
                <Text style={styles.journeyFrameText}>
                  Đã quét: {journeyFrameCount} frame
                </Text>
                <Text style={styles.journeyFastHint}>
                  FAST: conf ≥ {LIVE_IMMEDIATE_CONF.toFixed(2)} đọc ngay • thấp hơn cần 2 lần
                </Text>
              </View>

              <View style={styles.cameraControls}>
                {!journeyRunning ? (
                  <TouchableOpacity
                    style={styles.startJourneyButton}
                    onPress={startJourney}
                    disabled={!cameraReady}
                  >
                    <Text style={styles.captureButtonText}>▶ Bắt đầu hành trình</Text>
                  </TouchableOpacity>
                ) : (
                  <TouchableOpacity
                    style={styles.stopJourneyButton}
                    onPress={stopJourney}
                  >
                    <Text style={styles.captureButtonText}>■ Dừng hành trình</Text>
                  </TouchableOpacity>
                )}

                <TouchableOpacity
                  style={styles.smallButton}
                  onPress={closeCamera}
                >
                  <Text style={styles.smallButtonText}>✕ Đóng</Text>
                </TouchableOpacity>
              </View>
            </>
          )}
        </View>
      )}

      {loading && (
        <View style={styles.loadingBox}>
          <ActivityIndicator size="large" />
          <Text style={styles.loadingText}>{loadingText}</Text>
        </View>
      )}

      {!loading && (annotatedImage || imageUri) && (
        <View style={styles.imageCard}>
          <Image
            source={{ uri: annotatedImage || imageUri! }}
            style={[styles.image, { height: resultImageHeight }]}
            resizeMode="contain"
          />
          {annotatedImage && (
            <Text style={styles.success}>✓ Đã nhận diện xong</Text>
          )}
        </View>
      )}

      {!loading && imageUri && detections.length === 0 && (
        <View style={styles.emptyCard}>
          <Text style={styles.emptyText}>Không phát hiện biển báo</Text>
        </View>
      )}

      {detections.length > 0 && (
        <Text style={styles.resultTitle}>
          Kết quả nhận diện ({detections.length})
        </Text>
      )}

      {detections.map((item, index) => (
        <View key={`${item.code}-${index}`} style={styles.resultCard}>
          <View style={styles.codeRow}>
            <Text style={styles.code}>{item.code}</Text>
            <Text style={styles.conf}>{(item.conf * 100).toFixed(1)}%</Text>
          </View>

          <Text style={styles.name}>{item.name || item.label}</Text>
          <Text style={styles.group}>{item.group}</Text>
          <Text style={styles.warning}>⚠️ {item.warning}</Text>

          {item.ocr_text ? (
            <Text style={styles.ocrText}>OCR: {item.ocr_text}</Text>
          ) : null}
        </View>
      ))}

      {journeyHistory.length > 0 && (
        <View style={styles.historyCard}>
          <Text style={styles.historyTitle}>🛣️ Nhật ký cảnh báo hành trình</Text>
          {journeyHistory.map((event) => (
            <View key={event.id} style={styles.historyItem}>
              <View style={styles.rowBetween}>
                <Text style={styles.historyCode}>{event.codes}</Text>
                <Text style={styles.historyTime}>{event.time}</Text>
              </View>
              <Text style={styles.historyText}>{event.text}</Text>
            </View>
          ))}
        </View>
      )}

      {timing ? (
        <View style={styles.timingCard}>
          <Text style={styles.timingTitle}>⚡ Thời gian xử lý</Text>
          <Text style={styles.timingText}>Chuẩn bị frame trên điện thoại: {timing.prepareMs} ms</Text>
          <Text style={styles.timingText}>YOLO inference: {timing.inferenceMs.toFixed(0)} ms</Text>
          <Text style={styles.timingText}>Backend tổng: {timing.backendTotalMs.toFixed(0)} ms</Text>
          <Text style={styles.timingText}>Mobile ↔ Backend tổng: {timing.requestMs} ms</Text>
          <Text style={styles.timingText}>Dung lượng ảnh gửi: ~{timing.payloadKb} KB</Text>
        </View>
      ) : null}

      {audioText ? (
        <View style={styles.audioCard}>
          <View style={styles.rowBetween}>
            <Text style={styles.audioTitle}>🔊 Nội dung cảnh báo</Text>
            <TouchableOpacity onPress={() => speakWarning(audioText)}>
              <Text style={styles.replayText}>Đọc lại</Text>
            </TouchableOpacity>
          </View>
          <Text style={styles.audioText}>{audioText}</Text>
        </View>
      ) : null}

      <Text style={styles.footerNote}>Backend: {API_URL}</Text>
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safeArea: { flex: 1, backgroundColor: "#f3f7fc" },
  screen: { flex: 1, backgroundColor: "#f3f7fc" },
  container: {
    flexGrow: 1,
    backgroundColor: "#f3f7fc",
    width: "100%",
    maxWidth: 720,
    alignSelf: "center",
    paddingTop: 18,
    paddingBottom: 36,
  },
  title: {
    color: "#111827",
    fontSize: 27,
    fontWeight: "700",
    textAlign: "center",
  },
  titleCompact: { fontSize: 23 },
  accountBar: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    marginBottom: 16,
    paddingHorizontal: 4,
  },
  accountText: {
    color: "#374151",
    fontWeight: "700",
    fontSize: 13,
  },
  logoutButton: {
    borderWidth: 1,
    borderColor: "#fecaca",
    backgroundColor: "#fff1f2",
    borderRadius: 999,
    paddingHorizontal: 12,
    paddingVertical: 7,
  },
  logoutText: {
    color: "#dc2626",
    fontWeight: "800",
    fontSize: 12,
  },
  subtitle: {
    marginTop: 7,
    marginBottom: 22,
    color: "#6b7280",
    fontSize: 15,
    textAlign: "center",
  },
  card: {
    backgroundColor: "#fff",
    borderRadius: 18,
    padding: 16,
    marginBottom: 14,
  },
  sectionTitle: {
    color: "#111827",
    fontSize: 17,
    fontWeight: "800",
  },
  helpText: {
    color: "#6b7280",
    marginTop: 5,
    lineHeight: 19,
    fontSize: 13,
  },
  vehicleWrap: {
    flexDirection: "row",
    flexWrap: "wrap",
    gap: 8,
    marginTop: 13,
  },
  vehicleChip: {
    borderWidth: 1,
    borderColor: "#d1d5db",
    backgroundColor: "#f9fafb",
    paddingHorizontal: 12,
    paddingVertical: 10,
    borderRadius: 999,
  },
  vehicleChipActive: {
    borderColor: "#2563eb",
    backgroundColor: "#eff6ff",
  },
  vehicleText: {
    color: "#4b5563",
    fontWeight: "600",
  },
  vehicleTextActive: {
    color: "#1d4ed8",
    fontWeight: "800",
  },
  rowBetween: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    gap: 10,
  },
  soundButton: {
    borderWidth: 1,
    borderColor: "#d1d5db",
    borderRadius: 999,
    paddingHorizontal: 11,
    paddingVertical: 8,
    backgroundColor: "#f9fafb",
  },
  soundButtonActive: {
    borderColor: "#16a34a",
    backgroundColor: "#f0fdf4",
  },
  soundText: {
    color: "#15803d",
    fontWeight: "700",
    fontSize: 13,
  },
  actionRow: {
    flexDirection: "row",
    gap: 10,
    marginTop: 15,
  },
  actionButton: { flex: 1, minWidth: 0 },
  actionRowCompact: { flexWrap: "wrap" },
  actionButtonCompact: { flexBasis: "47%", flexGrow: 1 },
  primaryButton: {
    backgroundColor: "#2563eb",
    paddingVertical: 15,
    borderRadius: 14,
    alignItems: "center",
  },
  cameraButton: {
    backgroundColor: "#0f766e",
    paddingVertical: 15,
    borderRadius: 14,
    alignItems: "center",
  },
  journeyButton: {
    backgroundColor: "#7c3aed",
    paddingVertical: 15,
    borderRadius: 14,
    alignItems: "center",
  },
  buttonText: {
    color: "#fff",
    fontSize: 13,
    fontWeight: "700",
  },
  cameraCard: {
    backgroundColor: "#111827",
    padding: 10,
    borderRadius: 18,
    marginBottom: 15,
    overflow: "hidden",
  },
  cameraWrap: {
    position: "relative",
  },
  camera: {
    width: "100%",
    height: 430,
    borderRadius: 14,
    overflow: "hidden",
  },
  liveBoxLayer: {
    position: "absolute",
    left: 0,
    right: 0,
    top: 0,
    bottom: 0,
  },
  liveDetectionBox: {
    position: "absolute",
    borderWidth: 2,
    borderColor: "#22c55e",
    borderRadius: 6,
  },
  liveDetectionLabel: {
    position: "absolute",
    left: -2,
    top: -24,
    maxWidth: 150,
    backgroundColor: "rgba(0,0,0,0.78)",
    color: "#ffffff",
    fontSize: 11,
    fontWeight: "800",
    paddingHorizontal: 5,
    paddingVertical: 3,
    borderRadius: 4,
  },
  liveBadge: {
    position: "absolute",
    top: 12,
    left: 12,
    flexDirection: "row",
    alignItems: "center",
    backgroundColor: "rgba(17,24,39,0.82)",
    paddingHorizontal: 10,
    paddingVertical: 7,
    borderRadius: 999,
  },
  liveDot: {
    width: 9,
    height: 9,
    borderRadius: 5,
    marginRight: 7,
  },
  liveDotOn: { backgroundColor: "#ef4444" },
  liveDotOff: { backgroundColor: "#f59e0b" },
  liveBadgeText: {
    color: "#fff",
    fontWeight: "800",
    fontSize: 12,
  },
  liveResultOverlay: {
    position: "absolute",
    left: 10,
    right: 10,
    bottom: 10,
    backgroundColor: "rgba(0,0,0,0.72)",
    borderRadius: 12,
    padding: 10,
  },
  liveResultText: {
    color: "#fff",
    fontSize: 12,
    fontWeight: "700",
    marginVertical: 2,
  },
  journeyFastHint: {
    color: "#9ca3af",
    fontSize: 11,
    marginTop: 4,
  },
  cameraControls: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    gap: 8,
    marginTop: 10,
  },
  smallButton: {
    flex: 1,
    backgroundColor: "#374151",
    borderRadius: 12,
    paddingVertical: 12,
    alignItems: "center",
  },
  smallButtonText: {
    color: "#fff",
    fontWeight: "700",
    fontSize: 12,
  },
  captureButton: {
    flex: 1.15,
    backgroundColor: "#dc2626",
    borderRadius: 12,
    paddingVertical: 13,
    alignItems: "center",
  },
  startJourneyButton: {
    flex: 2,
    backgroundColor: "#16a34a",
    borderRadius: 12,
    paddingVertical: 13,
    alignItems: "center",
  },
  stopJourneyButton: {
    flex: 2,
    backgroundColor: "#dc2626",
    borderRadius: 12,
    paddingVertical: 13,
    alignItems: "center",
  },
  journeyInfoBar: {
    marginTop: 10,
    backgroundColor: "#1f2937",
    borderRadius: 12,
    paddingHorizontal: 12,
    paddingVertical: 10,
  },
  journeyInfoText: {
    color: "#f9fafb",
    fontWeight: "700",
  },
  journeyFrameText: {
    color: "#9ca3af",
    marginTop: 3,
    fontSize: 12,
  },
  captureButtonText: {
    color: "#fff",
    fontWeight: "800",
  },
  loadingBox: {
    marginTop: 20,
    alignItems: "center",
  },
  loadingText: {
    marginTop: 10,
    color: "#374151",
  },
  imageCard: {
    marginTop: 16,
    padding: 12,
    backgroundColor: "#fff",
    borderRadius: 18,
  },
  image: {
    width: "100%",
    height: 360,
    backgroundColor: "#e5e7eb",
    borderRadius: 14,
  },
  success: {
    color: "#15803d",
    textAlign: "center",
    fontWeight: "700",
    fontSize: 16,
    marginTop: 12,
  },
  emptyCard: {
    marginTop: 15,
    padding: 16,
    backgroundColor: "#fff",
    borderRadius: 14,
  },
  emptyText: {
    textAlign: "center",
    color: "#6b7280",
  },
  resultTitle: {
    fontSize: 20,
    fontWeight: "700",
    color: "#111827",
    marginTop: 22,
    marginBottom: 5,
  },
  resultCard: {
    backgroundColor: "#fff",
    padding: 16,
    borderRadius: 16,
    marginTop: 12,
  },
  codeRow: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
  },
  code: {
    fontSize: 21,
    fontWeight: "800",
    color: "#2563eb",
  },
  conf: {
    color: "#15803d",
    fontWeight: "700",
  },
  name: {
    marginTop: 8,
    fontSize: 18,
    fontWeight: "700",
    color: "#111827",
  },
  group: {
    color: "#6b7280",
    marginTop: 5,
  },
  warning: {
    marginTop: 12,
    color: "#b45309",
    lineHeight: 21,
  },
  ocrText: {
    marginTop: 10,
    color: "#4b5563",
    fontStyle: "italic",
  },
  historyCard: {
    backgroundColor: "#fff",
    padding: 16,
    borderRadius: 16,
    marginTop: 18,
  },
  historyTitle: {
    color: "#111827",
    fontWeight: "800",
    fontSize: 16,
    marginBottom: 6,
  },
  historyItem: {
    borderTopWidth: 1,
    borderTopColor: "#e5e7eb",
    paddingTop: 10,
    marginTop: 10,
  },
  historyCode: {
    color: "#2563eb",
    fontWeight: "800",
  },
  historyTime: {
    color: "#9ca3af",
    fontSize: 12,
  },
  historyText: {
    color: "#4b5563",
    marginTop: 5,
    lineHeight: 19,
  },
  timingCard: {
    backgroundColor: "#f0fdf4",
    padding: 16,
    borderRadius: 16,
    marginTop: 18,
    borderWidth: 1,
    borderColor: "#bbf7d0",
  },
  timingTitle: {
    color: "#15803d",
    fontWeight: "800",
    marginBottom: 7,
  },
  timingText: {
    color: "#374151",
    lineHeight: 20,
    fontSize: 13,
  },
  audioCard: {
    backgroundColor: "#eff6ff",
    padding: 16,
    borderRadius: 16,
    marginTop: 18,
  },
  audioTitle: {
    color: "#1d4ed8",
    fontWeight: "700",
  },
  audioText: {
    marginTop: 8,
    lineHeight: 21,
    color: "#374151",
  },
  replayText: {
    color: "#2563eb",
    fontWeight: "800",
  },
  chatEntryButton: {
  backgroundColor: "#1d4ed8",
  borderRadius: 16,
  paddingHorizontal: 16,
  paddingVertical: 14,
  marginBottom: 16,
},
chatEntryText: {
  color: "#ffffff",
  fontSize: 16,
  fontWeight: "800",
  textAlign: "center",
},
chatEntryHint: {
  color: "#dbeafe",
  fontSize: 13,
  marginTop: 5,
  textAlign: "center",
},
  footerNote: {
    marginTop: 24,
    textAlign: "center",
    color: "#9ca3af",
    fontSize: 12,
  },
});