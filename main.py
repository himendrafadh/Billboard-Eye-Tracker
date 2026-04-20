"""Billboard Eye Tracker — real-time analytics pipeline."""

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
OUTPUT_DIR        = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)


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

        print(f"[INFO] CSV output: {self.path}")

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

        print(f"[CSV] {row['timestamp']} | "
              f"Lewat: {people_passing} | "
              f"Lihat: {people_watching}")


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
        self.logger.log(self.interval_passing, self.interval_watching)

        # reset counter interval (bukan total)
        self.counter.reset()
        self.cooldown.reset_interval()
        self.interval_passing  = 0
        self.interval_watching = 0
        self.interval_start    = datetime.now()

    # ── gambar overlay info di frame ─────────────────────────────────────
    def _draw_overlay(self, frame, active_people, watching_now):
        elapsed    = datetime.now() - self.interval_start
        remaining  = timedelta(minutes=INTERVAL_MINUTES) - elapsed
        rem_sec    = int(remaining.total_seconds())
        rem_str    = f"{rem_sec // 60:02d}:{rem_sec % 60:02d}"

        lines = [
            f"Orang di frame : {active_people}",
            f"Total lewat    : {self.interval_passing}",
            f"Lihat sekarang : {watching_now}",
            f"Total lihat    : {self.interval_watching}",
            f"Flush dalam    : {rem_str}",
        ]

        # background semi-transparan
        overlay = frame.copy()
        cv2.rectangle(overlay, (8, 8), (300, 165), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

        for i, line in enumerate(lines):
            cv2.putText(frame, line, (14, 32 + i * 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2)

        return frame

    # ── loop utama ───────────────────────────────────────────────────────
    def run(self):
        """Open video source and run the analytics loop until 'Q' is pressed."""
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            print("[ERROR] Tidak bisa membuka sumber video.")
            return

        print("[INFO] Pipeline berjalan. Tekan 'Q' untuk keluar.")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("[INFO] Video selesai / frame tidak terbaca.")
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
                        print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                              f"Wajah {face_id} dihitung lihat! "
                              f"Total interval: {self.interval_watching}")

            # ── 4. overlay UI ────────────────────────────────────────────
            frame = self._draw_overlay(frame, active_people, watching_now)

            # ── 5. cek apakah sudah waktunya flush interval ──────────────
            elapsed = datetime.now() - self.interval_start
            if elapsed >= timedelta(minutes=INTERVAL_MINUTES):
                self._flush_interval()

            cv2.imshow("Billboard Analytics — tekan Q untuk keluar", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        # flush data terakhir sebelum tutup
        print("[INFO] Menyimpan data terakhir...")
        self._flush_interval()

        cap.release()
        cv2.destroyAllWindows()
        print(f"[SELESAI] CSV tersimpan di: {self.logger.path}")


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    # source=0 → webcam
    # source="video.mp4" → file video
    pipeline = BillboardPipeline(source=0)
    pipeline.run()