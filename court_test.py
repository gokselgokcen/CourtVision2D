import cv2
from ultralytics import YOLO

def main():
    # 1. Eğittiğin modeli yükle
    print("Saha modeli (best.pt) yükleniyor...")
    try:
        model = YOLO('best.pt')
    except Exception as e:
        print(f"HATA: best.pt dosyası bulunamadı! {e}")
        return

    # 2. Videoyu aç
    video_path = "basket2.mp4" # Video ismini kontrol et
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Video açılamadı.")
        return

    print("Video oynatılıyor... Noktaların üzerindeki NUMARALARA dikkat et.")

    while True:
        success, frame = cap.read()
        if not success:
            break

        # 3. Tahmin Yap (Keypoint Detection)q
        results = model(frame, stream=True)

        for result in results:
            # Keypoint'leri (noktaları) al
            if result.keypoints is not None:
                # xy koordinatlarını al (ilk tespit edilen sahayı alıyoruz)
                # keypoints.xy -> (N, K, 2) shape
                # N: Nesne sayısı, K: Nokta sayısı, 2: x,y
                
                # Sadece güvenilir tespit varsa işlem yap
                if result.keypoints.conf is None:
                     continue
                
                keypoints = result.keypoints.xy.cpu().numpy()

                for kpts in keypoints:
                    for i, (x, y) in enumerate(kpts):
                        # Eğer nokta (0,0) değilse çiz (YOLO bulamadığı noktaya 0,0 verir)
                        if x > 0 and y > 0:
                            # Noktayı çiz (Sarı)
                            cv2.circle(frame, (int(x), int(y)), 6, (0, 255, 255), -1)
                            # Yanına NUMARASINI yaz (Kırmızı ve Büyük)
                            cv2.putText(frame, str(i), (int(x), int(y)-10), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Görüntüyü küçültüp göster (sığması için)
        cv2.imshow("Saha Nokta Testi", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()