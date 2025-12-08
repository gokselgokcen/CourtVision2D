import cv2
import config
from src.court_manager import CourtManager
from src.human_manager import HumanManager # Yeni eklenen modülümüz

def main():
    # 1. Video ve Kaynaklar
    video_path = "assets/videos/basketLeft.mp4" 
    cap = cv2.VideoCapture(video_path)
    
    # Referans 2D Taktik Tahtası
    tactical_board = cv2.imread(config.REF_IMAGE_PATH)
    if tactical_board is None:
        print(f"HATA: 2D Saha resmi bulunamadı! Path: {config.REF_IMAGE_PATH}")
        return

    # 2. Yöneticileri Başlat
    print("Saha Modeli Yükleniyor...")
    court_manager = CourtManager()
    
    print("İnsan Tespiti ve Renk Modülü Yükleniyor...")
    human_manager = HumanManager()

    print("Sistem Hazır. Başlatılıyor...")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Video bitti.")
            break

        # İşlem hızı ve ekran sığması için resize (Opsiyonel)
        # Not: Koordinatlar resize'a göre değişeceği için modeller bunu handle eder
        frame = cv2.resize(frame, (1280, 720))

        # --- AŞAMA 1: SAHA ZEMİNİ VE MATRİS (COURT) ---
        # Önce sahayı bulup "Yerçekimini" ve "Koordinat Sistemini" kuruyoruz.
        court_result = court_manager.predict(frame)
        matrix_ready = court_manager.update_homography(court_result)
        
        # --- AŞAMA 2: OYUNCULAR VE TAKIMLAR (HUMANS) ---
        # HumanManager'a frame'i ve court_manager'ı veriyoruz.
        # O da bize işlenmiş görüntüyü ve oyuncu verilerini dönüyor.
        annotated_frame, players_data = human_manager.detect_and_process(frame, court_manager)

        # --- AŞAMA 3: 2D BOARD GÖRSELLEŞTİRME ---
        # Her karede temiz bir tahta kopya al
        board_display = tactical_board.copy()
        
        # Eğer matris hazırsa ve oyuncu verisi varsa çizmeye başla
        if matrix_ready and players_data:
            for player in players_data:
                # transform_point sonucu None değilse (yani saha içindeyse)
                mapped_point = player['mapped_point']
                team_id = player['team_id']
                
                if mapped_point is not None:
                    # Rengi belirle (Takım 0: Mavi, Takım 1: Kırmızı, Bilinmeyen: Gri)
                    if team_id == 0:
                        color = (255, 0, 0)   # Mavi (OpenCV'de BGR -> Blue)
                    elif team_id == 1:
                        color = (0, 0, 255)   # Kırmızı (Red)
                    elif team_id == 99:
                        color = (50,50,50) # gri (Hakem)
                    else:
                        color = (128, 128, 128) # Gri (Öğrenme aşaması)

                    # 2D Tahtaya Nokta Çiz
                    cv2.circle(board_display, mapped_point, 10, color, -1)
                    # Etrafına siyah kontür atalım ki belli olsun
                    cv2.circle(board_display, mapped_point, 10, (0,0,0), 2)

        # --- EKRANA BASMA ---
        # Saha çizgilerini görmek istiyorsan court_manager.draw_keypoints de kullanabilirsin
        # Ama human_manager zaten frame üzerine kutu çiziyor.
        
        # Küçük bir bilgi ekranı ekle
        cv2.putText(annotated_frame, f"Matrix Ready: {matrix_ready}", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        
        cv2.imshow("CourtVision AI - Main Camera", annotated_frame)
        cv2.imshow("2D Tactical Board", board_display)

        # 'q' ile çıkış
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()