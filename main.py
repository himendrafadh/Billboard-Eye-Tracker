"""Billboard Eye Tracker — Production-grade Multiprocessing Pipeline."""

import multiprocessing as mp
from multiprocessing import shared_memory
import numpy as np
import cv2
import time
import sys
import signal
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd

# Konvensi modul Anda: Pastikan import spesifik tidak berada di global runtime, 
# kecuali dibutuhkan di level signature (hanya saat eksekusi worker).
# Kita biarkan di-import di global, tapi ingat CUDA inisialisasi *ada di worker*.

# ==========================================
# 1. KONFIGURASI GLOBAL
# ==========================================
FRAME_W, FRAME_H, FRAME_C = 640, 480, 3
FRAME_SHAPE = (FRAME_H, FRAME_W, FRAME_C)
FRAME_DTYPE = np.uint8
FRAME_SIZE = int(np.prod(FRAME_SHAPE)) * np.dtype(FRAME_DTYPE).itemsize

# Ring Buffer Ganda sudah cukup jika kita gunakan absolute latest dropping method
SHM_SLOTS = 2 
SHM_NAMES_IN = [f"shm_cam_in_{i}" for i in range(SHM_SLOTS)]   # Untreated raw frame
SHM_NAMES_OUT = [f"shm_cam_out_{i}" for i in range(SHM_SLOTS)] # Processed frame w/ bbox

COOLDOWN_MINUTES  = 5     
INTERVAL_MINUTES  = 10    
OUTPUT_DIR        = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)


# ==========================================
# 2. DEFINISI LOGICAL CLASS
# ==========================================
class CooldownTracker:
    def __init__(self, cooldown_minutes=COOLDOWN_MINUTES):
        self.cooldown    = timedelta(minutes=cooldown_minutes)
        self.last_seen   = {}   
        self.total_watch = 0    

    def check_and_register(self, track_id):
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
        self.total_watch = 0

class CSVLogger:
    def __init__(self):
        timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path  = OUTPUT_DIR / f"billboard_{timestamp}.csv"
        self.rows  = []
        pd.DataFrame(columns=[
            "timestamp", "people_passing", "people_watching"
        ]).to_csv(self.path, index=False)
        print(f"[INFO] CSV output: {self.path}")

    def log(self, people_passing, people_watching):
        row = {
            "timestamp"       : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "people_passing"  : people_passing,
            "people_watching" : people_watching,
        }
        self.rows.append(row)
        pd.DataFrame([row]).to_csv(self.path, mode="a", header=False, index=False)
        print(f"[CSV] {row['timestamp']} | Lewat: {people_passing} | Lihat: {people_watching}")


# ==========================================
# 3. PRODUCER: CAMERA I/O PROCESS
# ==========================================
def camera_producer(exit_event, latest_in_idx, frame_ready_event, source=0):
    # Hijack signal, Worker harus cuek pada OS kill. Mati patuh pada 'exit_event' bapaknya.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)

    print("[Producer] Menyalakan kamera...")
    backend = cv2.CAP_DSHOW if sys.platform == 'win32' else cv2.CAP_ANY
    if isinstance(source, str):
        cap = cv2.VideoCapture(source)
    else:
        cap = cv2.VideoCapture(source, backend)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) # Potong buffer antrean laten HW
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

    shm_blocks = [shared_memory.SharedMemory(name=name) for name in SHM_NAMES_IN]
    slot_idx = 0

    try:
        while not exit_event.is_set():
            ret, frame = cap.read()
            if not ret:
                # Bila file video habis, shutdown pipeline. Bila webcam, break saja.
                if isinstance(source, str): break 
                continue
            
            # Pengamanan jika resolusi kamera tidak patuh
            if frame.shape != FRAME_SHAPE:
                frame = cv2.resize(frame, (FRAME_W, FRAME_H))

            # Tulis ke memori (copyto) -> zero cross-process boundaries (O(N) bytes cost, super low overhead)
            shm = shm_blocks[slot_idx]
            dst_array = np.ndarray(FRAME_SHAPE, dtype=FRAME_DTYPE, buffer=shm.buf)
            np.copyto(dst_array, frame)

            # Sinkronisasi lock-free: update pointer lalu bunyikan lonceng bangun 
            with latest_in_idx.get_lock():
                latest_in_idx.value = slot_idx
            
            frame_ready_event.set()
            slot_idx = (slot_idx + 1) % SHM_SLOTS

    finally:
        print("[Producer] Membersihkan resource I/O...")
        cap.release()
        for shm in shm_blocks: shm.close() 

# ==========================================
# 4. CONSUMER: AI WORKER PROCESS
# ==========================================
def draw_overlay(frame, active, passing, watching_now, total_watch, start_time):
    elapsed    = datetime.now() - start_time
    remaining  = timedelta(minutes=INTERVAL_MINUTES) - elapsed
    rem_sec    = int(remaining.total_seconds())
    rem_str    = f"{rem_sec // 60:02d}:{rem_sec % 60:02d}"

    lines = [
        f"Orang di frame : {active}",
        f"Total lewat    : {passing}",
        f"Lihat sekarang : {watching_now}",
        f"Total lihat    : {total_watch}",
        f"Flush dalam    : {rem_str}",
    ]
    overlay = frame.copy()
    cv2.rectangle(overlay, (8, 8), (180, 115), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (14, 25 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 220, 255), 1)
    return frame

def ai_worker_process(exit_event, latest_in_idx, latest_out_idx, frame_ready_event, result_queue):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)

    print("[AI Worker] Melakukan inisialisasi model CUDA/YOLO..")
    # Memuat object CUDA/Network DENGAN AMAN DI SINI, tidak di-fork dari proses luar.
    from scripts.people_counter import PeopleCounter
    from scripts.eye_tracker import EyeTracker
    
    counter  = PeopleCounter()
    tracker  = EyeTracker()
    cooldown = CooldownTracker()
    logger   = CSVLogger()

    interval_start    = datetime.now()
    interval_passing  = 0
    interval_watching = 0

    known_shms_in = {name: shared_memory.SharedMemory(name=name) for name in SHM_NAMES_IN}
    known_shms_out = {name: shared_memory.SharedMemory(name=name) for name in SHM_NAMES_OUT}
    out_slot_idx = 0
    
    try:
        while not exit_event.is_set():
            # Block tunggu trigger (0% CPU Burner)
            if not frame_ready_event.wait(timeout=0.1):
                continue
            
            start_infer = time.time()
            frame_ready_event.clear()
            
            with latest_in_idx.get_lock():
                in_target_slot = latest_in_idx.value
                
            shm_in = known_shms_in[SHM_NAMES_IN[in_target_slot]]
            frame_view = np.ndarray(FRAME_SHAPE, dtype=FRAME_DTYPE, buffer=shm_in.buf)
            
            # Wajib meng-copy view read-only ini agar bisa digambar kotak warna-warni oleh library
            frame_work = frame_view.copy()

            # --- 1. PEOPLE COUNTER ---
            frame_work, active_people, _ = counter.process_frame(frame_work)
            interval_passing = counter.count
            
            # --- 2. EYE TRACKER ---
            frame_work, faces = tracker.process_frame(frame_work)

            # --- 3. LOGGING COOLDOWN ---
            watching_now = 0
            for i, face in enumerate(faces):
                if face["looking"]:
                    watching_now += 1
                    face_id = f"face_{i}"
                    if cooldown.check_and_register(face_id):
                        interval_watching += 1
                        
            # --- 4. OVERLAY ---
            frame_work = draw_overlay(frame_work, active_people, interval_passing, watching_now, interval_watching, interval_start)

            # Cek flush Interval
            elapsed_min = datetime.now() - interval_start
            if elapsed_min >= timedelta(minutes=INTERVAL_MINUTES):
                logger.log(interval_passing, interval_watching)
                counter.reset()
                cooldown.reset_interval()
                interval_passing = 0
                interval_watching = 0
                interval_start = datetime.now()

            # Tulis hasil digambar ke SHM OUT
            shm_out = known_shms_out[SHM_NAMES_OUT[out_slot_idx]]
            dst_out = np.ndarray(FRAME_SHAPE, dtype=FRAME_DTYPE, buffer=shm_out.buf)
            np.copyto(dst_out, frame_work)

            with latest_out_idx.get_lock():
                latest_out_idx.value = out_slot_idx

            # Kirim flag ke Main Thread bahwa frame siap dirender, dan selipkan timing inferensi
            result_queue.put({"timestamp": start_infer})
            out_slot_idx = (out_slot_idx + 1) % SHM_SLOTS
            
    finally:
        print("[AI Worker] Menyimpan data terakhir sebelum dimatikan...")
        logger.log(interval_passing, interval_watching)
        for shm in known_shms_in.values(): shm.close()
        for shm in known_shms_out.values(): shm.close()
        print("[AI Worker] Dimatikan.")


# ==========================================
# 5. MAIN THREAD & GRACEFUL SHUTDOWN
# ==========================================
def shutdown_handler(signum, frame):
    """Callback di level sistem OS/Terminal untuk graceful death"""
    global exit_event_global
    print(f"\n[Term Signal {signum}] Menginisiasi Global Shutdown...")
    if 'exit_event_global' in globals():
        exit_event_global.set()

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    
    # Orkestrasi Sinyal Cepat
    exit_event_global = mp.Event() 
    latest_in_idx     = mp.Value('i', 0)     
    latest_out_idx    = mp.Value('i', 0)
    frame_ready_event = mp.Event()         
    result_queue      = mp.Queue()
    
    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    print("[Main] Mengalokasikan RAM Shared Memory...")
    shm_blocks_all = []
    
    # Helper pre-alokasi
    def allocate_shm(names):
        blocks = []
        for name in names:
            try:
                shm = shared_memory.SharedMemory(create=True, name=name, size=FRAME_SIZE)
            except getattr(shared_memory, 'SharedMemoryError', FileExistsError):
                # Pada Windows, memory dari terminal terputus tetap ditahan oleh lock internal.
                # Alih-alih menghapus dan mereplika seketika, cukup 'attach' ke eksistensinya.
                shm = shared_memory.SharedMemory(name=name)
            blocks.append(shm)
        return blocks

    shm_blocks_in  = allocate_shm(SHM_NAMES_IN)
    shm_blocks_out = allocate_shm(SHM_NAMES_OUT)
    shm_blocks_all.extend(shm_blocks_in + shm_blocks_out)

    # Spawn Worker
    # Argument sys.argv bisa dimasukkan ke camera misalnya 'video.mp4'. Disini source=0.
    producer_process = mp.Process(target=camera_producer, args=(exit_event_global, latest_in_idx, frame_ready_event, 1))
    ai_process       = mp.Process(target=ai_worker_process, args=(exit_event_global, latest_in_idx, latest_out_idx, frame_ready_event, result_queue))
    
    producer_process.start()
    ai_process.start()

    print("[Main] Semua Sub-Sistem Dinyalakan. Menunggu frame pertama...")
    
    try:
        while not exit_event_global.is_set():
            if not result_queue.empty():
                result = result_queue.get()
                
                # Zero-Copy fetching dari AI
                with latest_out_idx.get_lock():
                    out_target_slot = latest_out_idx.value
                
                shm_out = shm_blocks_out[out_target_slot]
                frame_view = np.ndarray(FRAME_SHAPE, dtype=FRAME_DTYPE, buffer=shm_out.buf)
                
                # Kalkulasi Hardware-to-Display E2E Latency 
                # (Asumsi Frame dikirim hingga di render Display Monitor)
                latency = (time.time() - result["timestamp"]) * 1000
                display_img = frame_view.copy() 
                
                cv2.putText(display_img, f"Net E2E Processing: {latency:.1f} ms", (10, 465), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (50, 255, 50), 1)
                
                cv2.imshow("Billboard Eye Tracker (Multiprocessing)", display_img)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                exit_event_global.set()
                
    except Exception as e:
        print(f"[Main] Eksepsi tidak terduga: {e}")
        exit_event_global.set()
    
    finally:
        print("\n[Main] Memulai House-Keeping. Jangan tutup paksa...!!")
        cv2.destroyAllWindows()
        
        producer_process.join(timeout=2)
        ai_process.join(timeout=2)
        
        if producer_process.is_alive(): producer_process.terminate()
        if ai_process.is_alive(): ai_process.terminate()
        
        print("[Main] Menghapus Kernel Memory Block...")
        for shm in shm_blocks_all:
            shm.close()
            try:
                shm.unlink()
            except Exception: pass 
            
        print("[Main] Tuntas. Sampai Jumpa.")
        sys.exit(0)