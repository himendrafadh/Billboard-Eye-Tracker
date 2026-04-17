import cv2
import mediapipe as mp
import numpy as np
from datetime import datetime

# ─────────────────────────────────────────────
# KONFIGURASI
# ─────────────────────────────────────────────

# Index landmark MediaPipe FaceMesh (dari 468 titik)
# Iris
LEFT_IRIS   = [474, 475, 476, 477]
RIGHT_IRIS  = [469, 470, 471, 472]

# Sudut mata (untuk hitung eye aspect ratio)
# format: [kiri, kanan, atas-luar, bawah-luar, atas-dalam, bawah-dalam]
LEFT_EYE    = [362, 263, 387, 380, 373, 385]
RIGHT_EYE   = [33,  133, 160, 144, 158, 153]

# Threshold
LOOKING_THRESHOLD = 0.25   # makin kecil = makin ketat (harus tepat ke kamera)
EAR_THRESHOLD     = 0.20   # eye aspect ratio minimum — mata harus terbuka


class EyeTracker:
    def __init__(self):
        print("[INFO] Loading MediaPipe FaceMesh...")
        self.mp_face   = mp.solutions.face_mesh
        self.mp_draw   = mp.solutions.drawing_utils

        # refine_landmarks=True wajib untuk aktifkan iris landmarks (469-477)
        self.face_mesh = self.mp_face.FaceMesh(
            max_num_faces        = 10,    # deteksi sampai 10 wajah sekaligus
            refine_landmarks     = True,
            min_detection_confidence = 0.5,
            min_tracking_confidence  = 0.5,
        )

    # ── helper: ambil koordinat pixel dari landmark ──────────────────────
    def _landmark_point(self, landmarks, idx, w, h):
        lm = landmarks[idx]
        return int(lm.x * w), int(lm.y * h)

    # ── eye aspect ratio: ukuran bukaan mata ─────────────────────────────
    def _eye_aspect_ratio(self, landmarks, eye_indices, w, h):
        pts = [self._landmark_point(landmarks, i, w, h) for i in eye_indices]
        # jarak vertikal
        v1 = np.linalg.norm(np.array(pts[2]) - np.array(pts[3]))
        v2 = np.linalg.norm(np.array(pts[4]) - np.array(pts[5]))
        # jarak horizontal
        hz = np.linalg.norm(np.array(pts[0]) - np.array(pts[1]))
        if hz == 0:
            return 0
        return (v1 + v2) / (2.0 * hz)

    # ── hitung posisi iris relatif terhadap tengah mata ──────────────────
    def _iris_offset(self, landmarks, iris_indices, eye_indices, w, h):
        """
        Return (offset_x, offset_y) dalam range [-1, 1].
        0,0 = iris tepat di tengah mata = lagi lihat lurus ke depan.
        """
        iris_pts = [self._landmark_point(landmarks, i, w, h)
                    for i in iris_indices]
        eye_pts  = [self._landmark_point(landmarks, i, w, h)
                    for i in eye_indices]

        iris_center = np.mean(iris_pts, axis=0)
        eye_left    = np.array(eye_pts[0])
        eye_right   = np.array(eye_pts[1])
        eye_center  = (eye_left + eye_right) / 2.0
        eye_width   = np.linalg.norm(eye_right - eye_left)

        if eye_width == 0:
            return 0, 0

        offset_x = (iris_center[0] - eye_center[0]) / eye_width
        offset_y = (iris_center[1] - eye_center[1]) / eye_width
        return float(offset_x), float(offset_y)

    # ── method utama ─────────────────────────────────────────────────────
    def process_frame(self, frame):
        """
        Terima 1 frame, deteksi semua wajah, cek apakah tiap wajah
        lagi liat ke kamera.

        Return:
          - frame dengan anotasi
          - list of dict per wajah: {
              'looking': bool,
              'face_box': (x1,y1,x2,y2),
              'left_offset': (ox, oy),
              'right_offset': (ox, oy),
            }
        """
        h, w = frame.shape[:2]
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        faces_data = []

        if not results.multi_face_landmarks:
            return frame, faces_data

        for face_landmarks in results.multi_face_landmarks:
            lms = face_landmarks.landmark

            # ── eye aspect ratio (pastikan mata terbuka) ─────────────────
            ear_l = self._eye_aspect_ratio(lms, LEFT_EYE,  w, h)
            ear_r = self._eye_aspect_ratio(lms, RIGHT_EYE, w, h)
            eyes_open = (ear_l > EAR_THRESHOLD) and (ear_r > EAR_THRESHOLD)

            # ── iris offset ──────────────────────────────────────────────
            lox, loy = self._iris_offset(lms, LEFT_IRIS,  LEFT_EYE,  w, h)
            rox, roy = self._iris_offset(lms, RIGHT_IRIS, RIGHT_EYE, w, h)

            # rata-rata offset kiri & kanan
            avg_ox = (abs(lox) + abs(rox)) / 2.0
            avg_oy = (abs(loy) + abs(roy)) / 2.0

            looking = (
                eyes_open
                and avg_ox < LOOKING_THRESHOLD
                and avg_oy < LOOKING_THRESHOLD
            )

            # ── bounding box wajah dari semua landmark ───────────────────
            xs = [int(lm.x * w) for lm in lms]
            ys = [int(lm.y * h) for lm in lms]
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)

            faces_data.append({
                "looking"      : looking,
                "face_box"     : (x1, y1, x2, y2),
                "left_offset"  : (lox, loy),
                "right_offset" : (rox, roy),
                "ear_left"     : ear_l,
                "ear_right"    : ear_r,
            })

            # ── gambar anotasi ───────────────────────────────────────────
            color  = (0, 255, 0) if looking else (0, 0, 255)
            label  = "LIHAT" if looking else "tidak lihat"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

            # dot iris kiri
            for idx in LEFT_IRIS:
                px, py = self._landmark_point(lms, idx, w, h)
                cv2.circle(frame, (px, py), 2, (255, 200, 0), -1)

            # dot iris kanan
            for idx in RIGHT_IRIS:
                px, py = self._landmark_point(lms, idx, w, h)
                cv2.circle(frame, (px, py), 2, (255, 200, 0), -1)

            # debug: tampilkan offset value
            cv2.putText(
                frame,
                f"ox:{avg_ox:.2f} oy:{avg_oy:.2f}  EAR:{(ear_l+ear_r)/2:.2f}",
                (x1, y2 + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1,
            )

        # ── overlay summary ──────────────────────────────────────────────
        watching = sum(1 for f in faces_data if f["looking"])
        cv2.putText(frame, f"Liat billboard: {watching}/{len(faces_data)}",
                    (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 220, 255), 2)

        return frame, faces_data


# ─────────────────────────────────────────────
# TEST LANGSUNG
# ─────────────────────────────────────────────
if __name__ == "__main__":
    tracker = EyeTracker()

    SOURCE = 0   # 0 = webcam, atau ganti path video

    cap = cv2.VideoCapture(SOURCE)
    if not cap.isOpened():
        print("[ERROR] Tidak bisa membuka sumber video.")
        exit(1)

    print("[INFO] Tekan 'Q' untuk keluar.")
    print("[INFO] Kotak HIJAU = lagi lihat kamera | Kotak MERAH = tidak lihat")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame, faces = tracker.process_frame(frame)

        # log ke terminal kalau ada yang lihat
        for i, face in enumerate(faces):
            if face["looking"]:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                      f"Wajah #{i+1} terdeteksi LIHAT ke kamera")

        cv2.imshow("Eye Tracker — tekan Q untuk keluar", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()