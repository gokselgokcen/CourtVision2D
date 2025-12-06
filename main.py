import cv2
import numpy as np
from ultralytics import YOLO

# Foot point
def get_foot_point(box):
    """
    Tespit kutusunun alt orta noktasını (ayak noktası) hesaplar.
    box: (x1, y1, x2, y2) formatında sınırlayıcı kutu koordinatları.
    """
    x1, y1, x2, y2 = box
    center_x = int((x1 + x2) / 2) #center
    bottom_y = int(y2)           # bottom of box
    return (center_x, bottom_y)  


def main():
    # YOLO modelini yükle
    model = YOLO("yolov8m.pt")

    video_path = 'basket2.mp4'
    cap = cv2.VideoCapture(video_path)

    # Video kontrolü
    if not cap.isOpened():
        print(f"HATA: '{video_path}' dosyası bulunamadı!")
        print("Lütfen video isminin ve uzantısının (.mp4) doğru olduğundan emin ol.")
        return
    
    print("Video başladı! Çıkmak için klavyeden 'q' tuşuna bas.")

    # Pencere oluşturma ve boyutlandırma
    cv2.namedWindow('Basketbol Takip', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Basketbol Takip', 1280, 800)

    # Ana video döngüsü
    while True:
        succeess, frame = cap.read()
        if not succeess:
            print("Video bitti veya okunamadı.")
            break

        #human detection
        results = model(frame, stream=False, conf=0.25 ) 

        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Sınıf ID'si 0 ise (YOLO için 'person' sınıfı)
                if int(box.cls[0]) == 0:
                    # Kutu koordinatlarını al (x1, y1, x2, y2)
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

                    # Ayak noktasını hesapla
                    foot_point = get_foot_point((x1, y1, x2, y2))

                    # Kutu çizimi
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 100, 0), 2)
                    cv2.putText(frame, 'PLAYER', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 0), 2)

                    # Ayak noktası çizimi
                    cv2.circle(frame, foot_point, 5, (0, 0, 255), -1)

        # Görüntüyü göster
        cv2.imshow("Court", frame)

        # 'q' tuşuna basılırsa döngüden çık
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    # Kaynakları serbest bırak
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()