import cv2
from ultralytics import YOLO
import config
import time

def main():
    # 1. Modeli ve Videoyu Yükle
    print(f"Model yükleniyor: {config.COURT_MODEL_PATH}...")
    model = YOLO(config.COURT_MODEL_PATH)
    
    video_path = "assets/videos/basketLeft2.mp4" # Test etmek istediğin videonun yolu
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Hata: Video açılamadı -> {video_path}")
        return

    print("\n--- KONTROLLER ---")
    print("SPACE (Boşluk): Durdur / Devam Et")
    print("Q: Çıkış")
    print("------------------\n")

    paused = False

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("Video bitti.")
                break
            
            # İşlem hızını artırmak için frame'i biraz küçültebiliriz (isteğe bağlı)
            # frame = cv2.resize(frame, (1280, 720)) 

            # Inference (Tahmin)
            results = model(frame, conf=0.25, verbose=False) # Conf biraz düşürdük ki zor noktaları da görelim
            result = results[0]

            # Çizim için kopyasını al
            display_frame = frame.copy()

            if result.keypoints is not None:
                # Keypointleri al (xy formatında)
                keypoints = result.keypoints.xy[0].cpu().numpy()

                for i, (x, y) in enumerate(keypoints):
                    # Eğer nokta bulunamadıysa (0,0) atla
                    if x < 1 and y < 1:
                        continue
                    
                    x, y = int(x), int(y)

                    # Noktayı çiz (Sarı Daire)
                    cv2.circle(display_frame, (x, y), 6, (0, 255, 255), -1)
                    
                    # ID Numarasını Yaz (Kırmızı, kalın ve okunaklı)
                    # Arkaplanı siyah yapalım ki okunsun
                    cv2.putText(display_frame, str(i), (x + 8, y - 8), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4) # Siyah dış hat
                    cv2.putText(display_frame, str(i), (x + 8, y - 8), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2) # Kırmızı iç yazı

        # Görüntüyü göster
        # Eğer ekranına sığmıyorsa resize yapabilirsin:
        h, w = display_frame.shape[:2]
        cv2.imshow("Court Keypoint Inspector", cv2.resize(display_frame, (1280, 720)))

        key = cv2.waitKey(30) & 0xFF

        # SPACE tuşu ile durdur/başlat
        if key == ord(' '):
            paused = not paused
            if paused:
                print("Video DURAKLATILDI. ID'leri not alıp SPACE'e bas.")
            else:
                print("Video DEVAM EDİYOR.")
        
        # Q tuşu ile çık
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()