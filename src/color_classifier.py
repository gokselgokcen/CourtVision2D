import cv2
import numpy as np
from sklearn.cluster import KMeans

class TeamColorClassifier:
    def __init__(self):
        self.kmeans = None
        self.team_colors = [] 
        self.trained = False

    def extract_dominant_color(self, frame, box):
        # ... (Bu kısım aynen kalabilir, üst gövdeyi kesiyor) ...
        x1, y1, x2, y2 = map(int, box)
        h, w, _ = frame.shape
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        player_roi = frame[y1:y2, x1:x2]
        if player_roi.size == 0: return None

        ph, pw, _ = player_roi.shape
        # Gövdeyi al (Ayaklar ve kafa hariç)
        torso_roi = player_roi[int(ph*0.20):int(ph*0.60), int(pw*0.25):int(pw*0.75)]
        
        if torso_roi.size == 0: return None
            
        avg_color_per_row = np.average(torso_roi, axis=0)
        avg_color = np.average(avg_color_per_row, axis=0) # [B, G, R]
        
        return avg_color

    def is_referee(self, color_bgr):
        """
        Rengin gri/siyah/beyaz olup olmadığını kontrol eder.
        Dönüş: True (Hakem) / False (Oyuncu)
        """
        # BGR'den HSV'ye çevir
        # openCV 3D array ister, o yüzden reshape yapıyoruz
        color_uint8 = np.uint8([[color_bgr]]) 
        hsv = cv2.cvtColor(color_uint8, cv2.COLOR_BGR2HSV)[0][0]
        
        saturation = hsv[1] # 0-255 arası
        value = hsv[2]      # Parlaklık
        
        # --- HAKEM FİLTRESİ AYARLARI ---
        # Gri, Siyah veya Beyaz formalar düşük saturation'a sahiptir.
        # Takım formaları canlı renktir (Yüksek saturation).
        # Threshold: 40-50 arası iyidir (Deneyerek bulabiliriz)
        if saturation < 40: 
            return True
            
        # Ekstra: Çok koyu (Siyah) ise yine hakemdir
        if value < 40:
            return True
            
        return False

    def fit(self, color_samples):
        """
        Modeli eğitirken hakem renklerini VERİ SETİNDEN ÇIKAR!
        Yoksa model Gri rengi bir takım sanar.
        """
        # Sadece renkli olanları (oyuncuları) filtrele
        filtered_samples = [c for c in color_samples if not self.is_referee(c)]
        
        if len(filtered_samples) < 20: # Yeterli oyuncu yoksa eğitme
            print("Yetersiz oyuncu verisi (Çoğu hakem veya gürültü olabilir).")
            return False
            
        print(f"Eğitim başlıyor. Toplam Örnek: {len(color_samples)}, Oyuncu: {len(filtered_samples)}")
        
        self.kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        self.kmeans.fit(filtered_samples)
        
        self.team_colors = self.kmeans.cluster_centers_
        self.trained = True
        return True

    def predict(self, color):
        """
        Return: 
         0: Team A
         1: Team B
         99: Referee (Hakem)
         -1: Error
        """
        if color is None: return -1
        
        # 1. Önce Hakem mi diye bak
        if self.is_referee(color):
            return 99 # HAKEM ID'si
            
        if not self.trained:
            return -1
            
        # 2. Değilse hangi takıma yakın?
        color_reshaped = np.array([color]) 
        label = self.kmeans.predict(color_reshaped)[0]
        return label