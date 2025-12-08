import os

# --- PATHS ---
BASE_DIR = os.getcwd()
ASSETS_DIR = os.path.join(BASE_DIR, 'assets')
MODELS_DIR = os.path.join(ASSETS_DIR, 'models')
IMAGES_DIR = os.path.join(ASSETS_DIR, 'images')

COURT_MODEL_PATH = os.path.join(MODELS_DIR, 'best.pt')
REF_IMAGE_PATH = os.path.join(IMAGES_DIR, 'court2D.jpg')
CONFIDENCE_THRESHOLD = 0.3

# --- ID LİSTELERİ ---
# Senin veri setine göre gruplar:

# 1. SOL TAKIM (Sol Pota Görüntüsündeki ID'ler)
LEFT_SIDE_IDS = [30, 21, 19, 16, 13] # 20 ve 15'i tahmini ekledim, seninkiler neyse o.

# 2. SAĞ TAKIM (Sağ Pota Görüntüsündeki ID'ler)
RIGHT_SIDE_IDS = [23, 18, 22, 12, 17]

# --- REFERANS NOKTALAR (Pusula İçin) ---
# Yönü anlamak için bu ID'lerin X değerlerini kıyaslayacağız.
# Üçlük Tepesi ID'leri (Hem sağ hem sol versiyonu)
REF_TOP_KEY_IDS = [23, 30] 

# Serbest Atış Çizgisi ID'leri (Hem sağ hem sol versiyonu)
REF_FT_IDS = [18, 21]

# --- KOORDİNAT HARİTASI (Merged) ---
# Tek bir sözlük, hepsi burada.
COURT_KEYPOINTS = {
    # --- SAĞ TARAF ---
    23: (452, 178), # Sağ Serbest Atış Tepe
    18: (489, 130), # Sağ Serbest Atış Üst
    22: (489, 228), # Sağ Serbest Atış Alt
    12: (605, 129), # Sağ Çizgi Üst
    17: (605, 229), # Sağ Çizgi Alt
    
    # --- SOL TARAF ---
    30: (180, 178), # Sol Serbest Atış Tepe
    21: (143, 130), # Sol Serbest Atış Üst
    19: (143, 228), # Sol Serbest Atış Alt
    16: (26, 129),  # Sol Çizgi Üst
    13: (26, 229),  # Sol Çizgi Alt
}

HUMAN_MODEL_PATH = os.path.join(MODELS_DIR, 'yolov8m.pt')