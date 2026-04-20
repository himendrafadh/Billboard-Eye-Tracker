"""Billboard Eye Tracker — real-time analytics pipeline.

Supports two modes:
  - Standalone : python main.py [source]     → outputs JSON lines to stdout
  - Via Electron: spawned by Electron main process, same JSON protocol
"""

import base64
import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import pandas as pd

from scripts.people_counter import PeopleCounter
from scripts.eye_tracker import EyeTracker

# ─────────────────────────────────────────────
# KONFIGURASI
# ─────────────────────────────────────────────
COOLDOWN_MINUTES  = 5     # cooldown per orang (menit)
INTERVAL_MINUTES  = 10    # akumulasi data sebelum flush ke CSV (menit)
INTERVAL_SECONDS  = INTERVAL_MINUTES * 60
OUTPUT_DIR        = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

# Resize width untuk encode frame (lebih kecil = lebih cepat transfer)
FRAME_WIDTH = 640


# ─────────────────────────────────────────────
# JSON IPC HELPERS
# ─────────────────────────────────────────────
def send(msg: dict):
    """Kirim satu JSON line ke stdout dan flush segera."""
    print(json.dumps(msg, ensure_ascii=False), flush=True)


# ─────────────────────────────────────────────
# COOLDOWN TRACKER
# ─────────────────────────────────────────────
class CooldownTracker:
    """
    Simpan timestamp terakhir tiap track_id "dihitung lihat".
    Kalau belum lewat COOLDOWN_MINUTES → skip, tidak dihitung lagi.
    """
    def __init__(self, cooldown_minutes=COOLDOWN_MINUTES):
        self.cooldown    = timedelta(minutes=cooldown_minutes)
        self.last_seen   = {}   # {track_id: datetime}
        self.total_watch = 0    # total orang unik yang dihitung lihat

    def check_and_register(self, track_id):
        """
        Return True  → orang ini boleh dihitung (baru / sudah lewat cooldown)
        Return False → masih dalam cooldown, skip
        """
        now = datetime.now()

        if track_id not in self.last_seen:
            self.last_seen[track_id] = now
            self.total_watch += 1
            return True

        elapsed = now - self.last_seen[track_id]
        if elapsed >= self.cooldown:
            self.last_seen[track_id] = now
            self.total_watch += 1
            return True

        return False

    def reset_interval(self):
        """Reset counter interval (dipanggil tiap 10 menit)."""
        self.total_watch = 0


# ─────────────────────────────────────────────
# CSV LOGGER
# ─────────────────────────────────────────────
class CSVLogger:
    """Append-mode CSV logger for interval-based billboard stats."""

    def __init__(self):
        timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path  = OUTPUT_DIR / f"billboard_{timestamp}.csv"
        self.rows  = []

        # tulis header
        pd.DataFrame(columns=[
            "timestamp", "people_passing", "people_watching"
        ]).to_csv(self.path, index=False)

    def log(self, people_passing, people_watching):
        """Append one row of interval data to the CSV file."""
        row = {
            "timestamp"       : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "people_passing"  : people_passing,
            "people_watching" : people_watching,
        }
        self.rows.append(row)

        # append ke file tanpa rewrite seluruhnya
        pd.DataFrame([row]).to_csv(self.path, mode="a", header=False, index=False)

        return row


# ─────────────────────────────────────────────
# PIPELINE UTAMA
# ─────────────────────────────────────────────
class BillboardPipeline:
    """Main pipeline: people counting, eye tracking, and CSV logging."""

    def __init__(self, source=0):
        self.source   = source
        self.counter  = PeopleCounter()
        self.tracker  = EyeTracker()
        self.cooldown = CooldownTracker()
        self.logger   = CSVLogger()

        # state interval
        self.interval_start    = datetime.now()
        self.interval_passing  = 0   # orang lewat dalam interval ini
        self.interval_watching = 0   # orang lihat dalam interval ini

    # ── flush data interval ke CSV ───────────────────────────────────────
    def _flush_interval(self):
        row = self.logger.log(self.interval_passing, self.interval_watching)

        # kirim csv_row ke Electron
        send({
            "type": "csv_row",
            "timestamp": row["timestamp"],
            "people_passing": row["people_passing"],
            "people_watching": row["people_watching"],
        })

        # reset counter interval (bukan total)
        self.counter.reset()
        self.cooldown.reset_interval()
        self.interval_passing  = 0
        self.interval_watching = 0
        self.interval_start    = datetime.now()

    # ── encode frame ke base64 JPEG ──────────────────────────────────────
    @staticmethod
    def _encode_frame(frame):
        """Resize frame dan encode ke base64 JPEG string."""
        h, w = frame.shape[:2]
        if w > FRAME_WIDTH:
            ratio = FRAME_WIDTH / w
            frame = cv2.resize(frame, (FRAME_WIDTH, int(h * ratio)))

        _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        return base64.b64encode(buf).decode('ascii')

    # ── loop utama ───────────────────────────────────────────────────────
    def run(self):
        """Open video source and run the analytics loop."""
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            send({"type": "error", "message": "Tidak bisa membuka sumber video."})
            return

        send({"type": "ready"})

        # frame pacing: target ~30 fps agar tidak spam terlalu banyak
        frame_interval = 1.0 / 30.0

        while True:
            t_start = time.time()

            ret, frame = cap.read()
            if not ret:
                break

            # ── 1. people counter ────────────────────────────────────────
            frame, active_people, _ = self.counter.process_frame(frame)

            # sync interval_passing dengan counter
            self.interval_passing = self.counter.count

            # ── 2. eye tracker ───────────────────────────────────────────
            frame, faces = self.tracker.process_frame(frame)

            # ── 3. cooldown check per wajah yang lihat ───────────────────
            watching_now = 0
            for i, face in enumerate(faces):
                if face["looking"]:
                    watching_now += 1
                    # pakai index wajah sebagai proxy ID
                    # (nanti di fase lanjutan bisa di-match ke track_id)
                    face_id = f"face_{i}"
                    if self.cooldown.check_and_register(face_id):
                        self.interval_watching += 1


            # ── 5. hitung sisa waktu flush ───────────────────────────────
            elapsed = datetime.now() - self.interval_start
            remaining = timedelta(minutes=INTERVAL_MINUTES) - elapsed
            flush_in_seconds = max(0, int(remaining.total_seconds()))

            # ── 6. kirim frame + stats ke Electron ───────────────────────
            b64 = self._encode_frame(frame)
            send({"type": "frame", "data": b64})

            send({
                "type": "stats",
                "active_people": active_people,
                "people_passing": self.interval_passing,
                "watching_now": watching_now,
                "people_watching": self.interval_watching,
                "flush_in_seconds": flush_in_seconds,
            })

            # ── 7. cek apakah sudah waktunya flush interval ──────────────
            if elapsed >= timedelta(minutes=INTERVAL_MINUTES):
                self._flush_interval()

            # ── 8. frame pacing ──────────────────────────────────────────
            dt = time.time() - t_start
            if dt < frame_interval:
                time.sleep(frame_interval - dt)

        # flush data terakhir sebelum tutup
        self._flush_interval()

        cap.release()
        send({"type": "done", "message": "Pipeline selesai."})


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
def main():
    """Entry point — baca source dari CLI argument."""
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        try:
            source = int(arg)   # webcam index (0, 1, ...)
        except ValueError:
            source = arg        # file path
    else:
        source = 0              # default: webcam

    pipeline = BillboardPipeline(source=source)
    pipeline.run()


if __name__ == "__main__":
    main()