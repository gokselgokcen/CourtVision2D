import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque
import config

class CourtManager:
    def __init__(self):
        print(f"Loading Court Model: {config.COURT_MODEL_PATH}")
        self.model = YOLO(config.COURT_MODEL_PATH)
        
        # Matris yumuşatma (Smoothing)
        self.matrix_buffer = deque(maxlen=5) 
        self.smooth_matrix = None
        
        self.court_keypoints = config.COURT_KEYPOINTS
        
        # Son bilinen yönü hafızada tut (kamera zoom yaparsa diye)
        self.current_side = "UNKNOWN" 


    def predict(self, frame):
        results = self.model(frame, conf=config.CONFIDENCE_THRESHOLD, verbose=False)
        return results[0]

    def get_side_from_geometry(self, keypoints_pixel):
        """
        Geometrik Yön Tayini (Pusula):
        Serbest Atış Çizgisi (FT) ile Üçlük Tepesi (Key) konumunu kıyaslar.
        """
        
        # 1. Referans noktaların koordinatlarını bul
        ft_x = None
        key_x = None

        for i, (x, y) in enumerate(keypoints_pixel):
            if x < 1: continue # Bulunamayan nokta
            
            # Bu nokta Serbest Atış çizgisinden biri mi? (18 veya 21)
            if i in config.REF_FT_IDS:
                ft_x = x
            
            # Bu nokta Üçlük Tepesinden biri mi? (23 veya 30)
            if i in config.REF_TOP_KEY_IDS:
                key_x = x

        # 2. Karşılaştırma Yap
        if ft_x is not None and key_x is not None:
            # SOL SAHA MANTIĞI:
            # Pota solda, Serbest Atış ortada, Üçlük Tepesi sağda kalır.
            # Yani: FT_X < KEY_X
            if ft_x < key_x:
                return "LEFT"
            
            # SAĞ SAHA MANTIĞI:
            # Üçlük Tepesi solda, Serbest Atış ortada, Pota sağda kalır.
            # Yani: KEY_X < FT_X (veya FT_X > KEY_X)
            else:
                return "RIGHT"
        
        return None # Karar veremedik (Yeterli nokta yok)

    def update_homography(self, result):
        if result.keypoints is None:
            return False

        keypoints_pixel = result.keypoints.xy[0].cpu().numpy()
        
        # --- ADIM 1: YÖNÜ TAYİN ET ---
        detected_side = self.get_side_from_geometry(keypoints_pixel)
        
        if detected_side:
            self.current_side = detected_side
        
        # Eğer hala yönü bilmiyorsak işlem yapma (veya varsayılanı kullan)
        if self.current_side == "UNKNOWN":
            return False

        # --- ADIM 2: FİLTRELEME (TEMİZLİK) ---
        # Hangi ID listesini kabul edeceğiz?
        valid_ids = []
        if self.current_side == "LEFT":
            valid_ids = config.LEFT_SIDE_IDS
        else:
            valid_ids = config.RIGHT_SIDE_IDS
            
        src_points = []
        dst_points = []

        for i, point in enumerate(keypoints_pixel):
            x, y = point
            
            # 1. Nokta bulundu mu?
            # 2. Config'de var mı?
            # 3. VE EN ÖNEMLİSİ: Bu nokta bizim tarafın (valid_ids) listesinde mi?
            if x > 1 and y > 1 and i in self.court_keypoints:
                if i in valid_ids:
                    src_points.append([x, y])
                    dst_points.append(self.court_keypoints[i])
        
        # --- ADIM 3: MATRİS HESABI ---
        if len(src_points) >= 4:
            src_arr = np.array(src_points, dtype=np.float32)
            dst_arr = np.array(dst_points, dtype=np.float32)
            
            matrix, _ = cv2.findHomography(src_arr, dst_arr, cv2.RANSAC, 5.0)
            
            if matrix is not None:
                self.matrix_buffer.append(matrix)
                self.smooth_matrix = np.mean(self.matrix_buffer, axis=0)
                return True
        
        # Buffer varsa eskisiyle devam et
        if len(self.matrix_buffer) > 0:
            return True
            
        return False

    def transform_point(self, point):
        if self.smooth_matrix is None:
            return None
        src_point = np.array([[[point[0], point[1]]]], dtype=np.float32)
        try:
            dst_point = cv2.perspectiveTransform(src_point, self.smooth_matrix)
            return (int(dst_point[0][0][0]), int(dst_point[0][0][1]))
        except:
            return None

    def draw_keypoints(self, frame, result):
        annotated_frame = frame.copy()
        
        # Ekrana hangi tarafta olduğumuzu yazalım (Debug için harika)
        color = (0, 255, 0) if self.current_side == "LEFT" else (0, 0, 255)
        text = f"SIDE: {self.current_side}"
        cv2.putText(annotated_frame, text, (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)

        if result.keypoints is not None:
            xy = result.keypoints.xy[0].cpu().numpy()
            
            # Geçerli ID listesini al
            valid_ids = config.LEFT_SIDE_IDS if self.current_side == "LEFT" else config.RIGHT_SIDE_IDS

            for i, (x, y) in enumerate(xy):
                if x > 1 and y > 1:
                    # Sadece geçerli tarafın noktalarını çiz
                    if i in valid_ids:
                        cv2.circle(annotated_frame, (int(x), int(y)), 5, (0, 255, 0), -1)
                        cv2.putText(annotated_frame, str(i), (int(x), int(y)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    else:
                        # Geçersiz noktaları (Hayaletleri) küçük kırmızı nokta yap (Görmek istersen)
                        cv2.circle(annotated_frame, (int(x), int(y)), 3, (0, 0, 255), -1)
        
        return annotated_frame