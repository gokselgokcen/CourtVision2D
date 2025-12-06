import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque
from sklearn.cluster import KMeans

# ==========================================
# 1. AYARLAR VE SABİTLER (KONFİGÜRASYON)
# ==========================================
CONFIG = {
    'VIDEO_PATH': 'basket2.mp4',
    'PLAYER_MODEL': 'yolov8m.pt',
    'COURT_MODEL': 'best.pt',
    'BOARD_IMG': 'court2D.jpg',
    'SMOOTH_WINDOW': 5,
    'CONF_THRESHOLD': 0.25,  # Nokta tespit hassasiyeti
    'SKIP_FRAMES': 0         # Hızlandırmak için kare atlama (gerekirse artırın)
}

# Saha Referans Noktaları (Burası Sizin Doldurduğunuz Yer)
REF_COURT_POINTS = {
    10: [605, 24],   # Sağ Üst Köşe
    3:  [605, 333],  # Sağ Alt Köşe
    18: [489, 130],  # Sağ Serbest Atış (Üst)
    21: [489, 228],  # Sağ Serbest Atış (Alt)
    23: [452, 178],  # Sağ Serbest Atış (Tepe)
    12 : [605,129],
    17 : [605,229]
}

# ==========================================
# 2. MODÜL: GÖRÜNTÜ VE MODEL YÖNETİMİ
# ==========================================
class GameDetector:
    def __init__(self):
        print("Modeller Yükleniyor...")
        self.player_model = YOLO(CONFIG['PLAYER_MODEL'])
        self.court_model = YOLO(CONFIG['COURT_MODEL'])
    
    def detect_court(self, frame):
        """Sadece saha noktalarını bulur"""
        return self.court_model(frame, verbose=False, conf=CONFIG['CONF_THRESHOLD'])[0]

    def detect_players(self, frame):
        """Sadece oyuncuları (insanları) bulur"""
        return self.player_model(frame, verbose=False, classes=[0])[0]

# ==========================================
# 3. MODÜL: GEOMETRİ VE HARİTALAMA
# ==========================================
class CourtMapper:
    def __init__(self):
        self.smoother_buffer = deque(maxlen=CONFIG['SMOOTH_WINDOW'])

    def get_homography(self, court_result):
        """
        Keypointleri alır, Hybrid (Homography veya Affine) matris hesaplar.
        """
        src_points = []
        dst_points = []
        
        if court_result.keypoints is None:
            return self._smooth(None)

        kp_xy = court_result.keypoints.xy.cpu().numpy()[0]
        
        for idx, (x, y) in enumerate(kp_xy):
            if idx in REF_COURT_POINTS and x > 1 and y > 1:
                src_points.append([x, y])
                dst_points.append(REF_COURT_POINTS[idx])

        # Matris Hesaplama Mantığı
        H = None
        n_points = len(src_points)
        
        src_arr = np.float32(src_points).reshape(-1, 1, 2)
        dst_arr = np.float32(dst_points).reshape(-1, 1, 2)

        if n_points >= 4:
            # Tam Perspektif
            H, _ = cv2.findHomography(src_arr, dst_arr, cv2.RANSAC, 5.0)
        elif n_points >= 2:
            # Yedek Plan (Affine)
            affine, _ = cv2.estimateAffinePartial2D(src_arr, dst_arr)
            if affine is not None:
                H = np.vstack((affine, [[0, 0, 1]]))
        
        return self._smooth(H)

    def _smooth(self, H):
        """Titremeyi engeller"""
        if H is not None:
            self.smoother_buffer.append(H)
        
        if len(self.smoother_buffer) > 0:
            return np.mean(self.smoother_buffer, axis=0)
        return None

# ==========================================
# 4. MODÜL: OYUNCU İŞLEME (FİLTRELER)
# ==========================================
class PlayerProcessor:
    def __init__(self):
        self.kmeans = KMeans(n_clusters=2, n_init=10)
        
    def process_all(self, player_result, frame, H_matrix):
        """
        Tüm oyuncuları alır:
        1. Koordinat dönüşümü yapar.
        2. Saha dışındakileri (seyirci) eler.
        3. Hakemleri eler.
        4. Takım renklerine ayırır.
        """
        valid_players = []
        colors_for_clustering = []

        if H_matrix is None:
            return []

        for box in player_result.boxes:
            # Kutu Koordinatları
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            foot_x, foot_y = (x1 + x2) / 2, y2
            
            # --- ADIM 1: Dönüşüm (Transformation) ---
            pt_cam = np.array([[[foot_x, foot_y]]], dtype=np.float32)
            pt_board = cv2.perspectiveTransform(pt_cam, H_matrix)
            bx, by = int(pt_board[0][0][0]), int(pt_board[0][0][1])

            # --- ADIM 2: Seyirci Filtresi (Saha Sınırı) ---
            # Taktik tahtası sınırları + biraz pay
            if not (-50 < bx < 1050 and -50 < by < 650):
                continue

            # --- ADIM 3: Renk Analizi ---
            crop = frame[int(y1):int(y2), int(x1):int(x2)]
            dom_color = self._get_dominant_color(crop)

            # --- ADIM 4: Hakem Filtresi ---
            if self._is_referee(dom_color):
                # Hakemse listeye ekleme (veya özel işaretle)
                continue 

            # Listeye ekle
            valid_players.append({
                'box': [int(x1), int(y1), int(x2), int(y2)],
                'board': (bx, by),
                'color': dom_color,
                'team': 0 # Sonra atanacak
            })
            colors_for_clustering.append(dom_color)

        # --- ADIM 5: Takım Kümeleme (Clustering) ---
        if len(colors_for_clustering) >= 2:
            labels = self.kmeans.fit_predict(colors_for_clustering)
            for i, player in enumerate(valid_players):
                player['team'] = labels[i]
        
        return valid_players

    def _get_dominant_color(self, img):
        if img.size == 0: return np.array([0,0,0])
        h, w = img.shape[:2]
        # Sadece gövde kısmını al (kafa ve bacakları at)
        center = img[int(h*0.2):int(h*0.6), int(w*0.2):int(w*0.8)]
        if center.size == 0: return np.array([0,0,0])
        return np.mean(center, axis=(0, 1))

    def _is_referee(self, bgr):
        # Gri ton kontrolü
        hsv = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2HSV)[0][0]
        return hsv[1] < 40 or hsv[2] < 30 # Low Saturation or Very Dark

# ==========================================
# 5. MODÜL: GÖRSELLEŞTİRME (ÇİZİM)
# ==========================================
class Visualizer:
    def __init__(self):
        self.board_base = cv2.imread(CONFIG['BOARD_IMG'])
        if self.board_base is None:
            self.board_base = np.zeros((600, 1000, 3), dtype=np.uint8)

    def draw(self, frame, players):
        board_view = self.board_base.copy()

        for p in players:
            # Renkler (Takım 0: Kırmızı, Takım 1: Mavi)
            color = (0, 0, 255) if p['team'] == 0 else (255, 0, 0)
            
            # 1. Videoda Kutu Çiz
            x1, y1, x2, y2 = p['box']
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # 2. Tahtada Daire Çiz
            bx, by = p['board']
            # Sınır kontrolü (Görsellik için)
            if 0 <= bx < 1000 and 0 <= by < 600:
                cv2.circle(board_view, (bx, by), 10, color, -1)
                cv2.circle(board_view, (bx, by), 12, (255, 255, 255), 1)

        return frame, board_view

# ==========================================
# ANA ÇALIŞTIRICI (MAIN)
# ==========================================
def main():
    # Sınıfları Başlat
    detector = GameDetector()
    mapper = CourtMapper()
    processor = PlayerProcessor()
    visualizer = Visualizer()

    cap = cv2.VideoCapture(CONFIG['VIDEO_PATH'])

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # --- PIPELINE (AKIŞ) ---
        
        # 1. Tespit
        court_res = detector.detect_court(frame)
        player_res = detector.detect_players(frame)

        # 2. Haritalama
        H_matrix = mapper.get_homography(court_res)

        # 3. Oyuncu İşleme (Filtreleme, Renk, Konum)
        clean_players = processor.process_all(player_res, frame, H_matrix)

        # 4. Çizim
        final_frame, final_board = visualizer.draw(frame, clean_players)

        # --- GÖSTERİM ---
        frame_resized = cv2.resize(final_frame, (800, 450))
        cv2.imshow('Analiz', frame_resized)
        cv2.imshow('Taktik Tahta', final_board)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()