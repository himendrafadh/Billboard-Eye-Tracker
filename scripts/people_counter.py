import cv2
from ultralytics import YOLO
from datetime import datetime

# ─────────────────────────────────────────────
# KONFIGURASI
# ─────────────────────────────────────────────
MODEL_PATH   = "yolov8n.pt"   # auto-download kalau belum ada
CONFIDENCE   = 0.4            # minimum confidence deteksi
TARGET_CLASS = 0              # class 0 = "person" di COCO dataset


class PeopleCounter:
    def __init__(self):
        print("[INFO] Loading model YOLOv8...")
        self.model       = YOLO(MODEL_PATH)
        self.tracked_ids = set()   # semua ID yang pernah masuk frame
        self.count       = 0       # total orang yang pernah terdeteksi

    # ── method utama, dipanggil tiap frame ──────────────────────────────
    def process_frame(self, frame):
        """
        Terima 1 frame, jalankan tracking, return:
          - frame yang sudah digambar box-nya
          - jumlah orang di frame ini (aktif)
          - total orang unik sejak mulai
        """
        results = self.model.track(
            source      = frame,
            persist     = True,        # wajib True agar ID konsisten antar frame
            tracker     = "bytetrack.yaml",
            classes     = [TARGET_CLASS],
            conf        = CONFIDENCE,
            verbose     = False,       # matikan log per-frame biar ga berisik
        )

        active_count = 0

        if results and results[0].boxes is not None:
            boxes = results[0].boxes

            for box in boxes:
                # skip kalau belum ada track ID (frame pertama kadang belum assign)
                if box.id is None:
                    continue

                track_id   = int(box.id.item())
                confidence = float(box.conf.item())
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                # kalau ID ini belum pernah kita lihat → tambah counter
                if track_id not in self.tracked_ids:
                    self.tracked_ids.add(track_id)
                    self.count += 1
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                          f"Orang baru terdeteksi! ID={track_id} | "
                          f"Total unik: {self.count}")

                active_count += 1

                # ── gambar bounding box ──────────────────────────────────
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"ID:{track_id} ({confidence:.2f})",
                    (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2
                )

        return frame, active_count, self.count

    def reset(self):
        """Reset counter (dipanggil tiap 10 menit oleh aggregator nanti)."""
        self.tracked_ids.clear()
        self.count = 0


# ─────────────────────────────────────────────
# TEST LANGSUNG — jalankan file ini untuk coba
# ─────────────────────────────────────────────
if __name__ == "__main__":
    counter = PeopleCounter()

    # ganti 0 → path video kalau mau test pakai file, contoh: "test.mp4"
    SOURCE = 0

    cap = cv2.VideoCapture(SOURCE)
    if not cap.isOpened():
        print("[ERROR] Tidak bisa membuka sumber video. "
              "Pastikan webcam terhubung atau path video benar.")
        exit(1)

    print("[INFO] Tekan 'Q' untuk keluar.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] Video selesai / frame tidak terbaca.")
            break

        frame, active, total = counter.process_frame(frame)

        cv2.imshow("People Counter — tekan Q untuk keluar", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n[SELESAI] Total orang unik terdeteksi: {counter.count}")