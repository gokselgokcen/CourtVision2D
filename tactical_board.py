import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque

# --- AYARLAR ---
VIDEO_PATH = 'basket2.mp4'
PLAYER_MODEL_PATH = 'yolov8m.pt'
COURT_MODEL_PATH = 'best.pt'
TACTICAL_BOARD_IMAGE = 'court2D.jpg' # Siyah arka plan resminiz

# Yumuşatma penceresi (Titremeyi azaltır)
SMOOTHING_WINDOW = 5

# --- SİZİN VERDİĞİNİZ KOORDİNATLAR ---
ref_court_points = {
    10: [605, 24],   # Sağ Üst Köşe
    3:  [605, 333],  # Sağ Alt Köşe
    18: [489, 130],  # Sağ Serbest Atış (Üst)
    22: [489, 228],  # Sağ Serbest Atış (Alt)
    23: [452, 178],  # Sağ Serbest Atış (Tepe)
    12 : [605,129],
    17 : [605,229]

    # Orta saha tespit edilirse buraya eklersiniz
}

class HomographySmoother:
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.buffer = deque(maxlen=window_size)
    
    def update(self, H):
        if H is None:
            return self.get_smoothed() # Eğer hesaplanamazsa eskisiyle devam et
        self.buffer.append(H)
        return self.get_smoothed()
    
    def get_smoothed(self):
        if not self.buffer:
            return None
        return np.mean(self.buffer, axis=0)

def main():
    player_model = YOLO(PLAYER_MODEL_PATH)
    court_model = YOLO(COURT_MODEL_PATH)
    
    cap = cv2.VideoCapture(VIDEO_PATH)
    
    # Taktik tahtası yükle (yoksa siyah ekran)
    board_img = cv2.imread(TACTICAL_BOARD_IMAGE)
    if board_img is None:
        board_img = np.zeros((600, 1000, 3), dtype=np.uint8)

    smoother = HomographySmoother(window_size=SMOOTHING_WINDOW)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        current_board = board_img.copy()

        # --- 1. KEYPOINT TESPİTİ ---
        # conf=0.25 -> Eşiği düşürdük, artık daha silik noktaları da görecek.
        court_results = court_model(frame, verbose=False, conf=0.25)[0]
        
        src_points = []
        dst_points = []
        
        if court_results.keypoints is not None and len(court_results.keypoints) > 0:
            kp_xy = court_results.keypoints.xy.cpu().numpy()[0]
            kp_conf = court_results.keypoints.conf.cpu().numpy()[0]
            
            for idx, (x, y) in enumerate(kp_xy):
                # ID listemizde var mı?
                if idx in ref_court_points:
                    # Koordinat mantıklı mı? (0,0 değilse)
                    if x > 1 and y > 1:
                        src_points.append([x, y])
                        dst_points.append(ref_court_points[idx])
                        
                        # Ekrana hangi noktaları kullandığını yaz
                        cv2.circle(frame, (int(x), int(y)), 6, (0, 255, 0), 2)
                        cv2.putText(frame, str(idx), (int(x), int(y)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        # --- 2. MATRİS HESAPLAMA (HYBRID YÖNTEM) ---
        H_curr = None
        point_count = len(src_points)

        src_arr = np.float32(src_points).reshape(-1, 1, 2)
        dst_arr = np.float32(dst_points).reshape(-1, 1, 2)

        if point_count >= 4:
            # EN İYİSİ: 4+ Nokta -> Perspektif (Homography)
            H_curr, _ = cv2.findHomography(src_arr, dst_arr, cv2.RANSAC, 5.0)
            cv2.putText(frame, "Mod: Homography (4+ Pts)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        elif point_count >= 2:
            # YEDEK PLAN: 2-3 Nokta -> Affine (Döndürme + Kaydırma + Zoom)
            # estimateAffinePartial2D sadece 2 nokta ile bile çalışır.
            # Çıktısı 2x3 matristir, bunu 3x3'e çevirmemiz gerekir.
            affine_matrix, _ = cv2.estimateAffinePartial2D(src_arr, dst_arr)
            
            if affine_matrix is not None:
                # 2x3 matrisi [0,0,1] ekleyerek 3x3 yap
                row = np.array([[0, 0, 1]])
                H_curr = np.vstack((affine_matrix, row))
                cv2.putText(frame, "Mod: Affine (2-3 Pts)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        else:
            cv2.putText(frame, "Yetersiz Nokta (<2)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Matrisi yumuşat
        H_smooth = smoother.update(H_curr)

        # --- 3. OYUNCULARI AKTAR ---
        if H_smooth is not None:
            player_results = player_model(frame, verbose=False, classes=[0])[0] # Sadece insan
            
            for box in player_results.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                foot_x, foot_y = (x1 + x2) / 2, y2
                
                # Oyuncuyu videoda işaretle
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                
                # Transform
                pt_cam = np.array([[[foot_x, foot_y]]], dtype=np.float32)
                pt_board = cv2.perspectiveTransform(pt_cam, H_smooth)
                
                bx, by = int(pt_board[0][0][0]), int(pt_board[0][0][1])
                
                # Tahtaya çiz
                if 0 <= bx < 1000 and 0 <= by < 600:
                     cv2.circle(current_board, (bx, by), 8, (0, 0, 255), -1)

        # Göster
        frame_s = cv2.resize(frame, (800, 450))
        cv2.imshow('Video', frame_s)
        cv2.imshow('Taktik Tahtasi', current_board)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()