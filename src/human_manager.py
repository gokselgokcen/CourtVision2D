import cv2
import numpy as np
from ultralytics import YOLO
import config
from src.color_classifier import TeamColorClassifier

class HumanManager:
    def __init__(self):
        print(f"Loading Human Model: {config.HUMAN_MODEL_PATH}")
        self.model = YOLO(config.HUMAN_MODEL_PATH)
        
        self.classifier = TeamColorClassifier()
        self.color_samples = [] # Kalibrasyon için havuz
        self.calibration_done = False
        
        # Kaç frame boyunca veri toplayacağız?
        self.calibration_frames = 60 
        self.frame_count = 0

    def detect_and_process(self, frame, court_manager):
        """
        1. İnsanları bul
        2. Seyircileri ele (Saha dışı kontrolü)
        3. Renk öğren veya sınıflandır
        """
        self.frame_count += 1
        results = self.model(frame, classes=[0], conf=0.3, verbose=False) # class 0 = person
        
        detected_players = [] # (Position_2D, Team_ID, Box)

        if results[0].boxes is None:
            return frame, []

        boxes = results[0].boxes.xyxy.cpu().numpy()
        
        for box in boxes:
            x1, y1, x2, y2 = box
            
            # --- 1. AYAK NOKTASI (POSİTİON) ---
            # Kutunun alt orta noktası
            foot_x = int((x1 + x2) / 2)
            foot_y = int(y2)
            
            # --- 2. SEYİRCİ FİLTRESİ (GÜNCELLENDİ) ---
            mapped_point = court_manager.transform_point((foot_x, foot_y))
            
            is_inside_court = False
            if mapped_point is not None:
                mx, my = mapped_point
                
                # --- ÇÖZÜM: BUFFER ZONE (TOLERANS PAYI) ---
                # Sınırları 0-1200 değil, -50 ile 1250 arası yapıyoruz.
                # Böylece çizgi üstündeki veya dönüşüm hatasıyla az dışarı düşenler de alınır.
                offset_x = 100  # Genişlik toleransı
                offset_y = 100  # Yükseklik toleransı (Pota altı dışarı taşabilir)
                
                # court_width ve height configden gelse iyi olur ama şimdilik manuel:
                court_w, court_h = 1200, 800 # Senin resim boyutların
                
                if (-offset_x <= mx <= court_w + offset_x) and (-offset_y <= my <= court_h + offset_y):
                    is_inside_court = True
            
            if not is_inside_court:
                continue # Bu kişi hala çok uzakta (Tribünde)

            # --- 3. RENK ANALİZİ ---
            player_color = self.classifier.extract_dominant_color(frame, box)
            
            team_id = -1 # Bilinmiyor
            
            if not self.calibration_done:
                # EĞİTİM AŞAMASI: Sadece veri topla
                if player_color is not None:
                    self.color_samples.append(player_color)
            else:
                # TAHMİN AŞAMASI
                team_id = self.classifier.predict(player_color)

            detected_players.append({
                'box': box,
                'mapped_point': mapped_point,
                'team_id': team_id,
                'color_vec': player_color
            })

        # --- KALİBRASYON KONTROLÜ ---
        if not self.calibration_done and self.frame_count > self.calibration_frames:
            success = self.classifier.fit(self.color_samples)
            if success:
                self.calibration_done = True
                self.color_samples = [] # Hafızayı temizle

        return self.draw_annotations(frame, detected_players)

    def draw_annotations(self, frame, players):
        annotated_frame = frame.copy()
        
        # Kalibrasyon durumu bilgisi
        status_text = "MODE: DETECTING" if self.calibration_done else f"MODE: LEARNING ({self.frame_count}/{self.calibration_frames})"
        cv2.putText(annotated_frame, status_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        for p in players:
            x1, y1, x2, y2 = map(int, p['box'])
            team_id = p['team_id']
            
            # Renk belirle (Görselleştirme için)
            if team_id == 0:
                color = (255, 0, 0) # Mavi Takım (Temsili)
            elif team_id == 1:
                color = (0, 0, 255) # Kırmızı Takım (Temsili)
            else:
                color = (128, 128, 128) # Öğreniliyor...

            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            # Ayak noktası
            cx = int((x1+x2)/2)
            cy = int(y2)
            cv2.circle(annotated_frame, (cx, cy), 5, (0, 255, 0), -1)

        return annotated_frame, players