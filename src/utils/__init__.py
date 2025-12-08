# src/utils/__init__.py

import cv2
import numpy as np

def is_point_in_polygon(point, polygon):
    """
    Verilen (x, y) noktası, belirtilen poligonun (saha sınırları) içinde mi?
    """
    # polygon yoksa veya boşsa False dön
    if polygon is None:
        return False
        
    # pointPolygonTest: Sonuç >= 0 ise nokta içeride veya kenardadır.
    # measureDist=False sadece içeride/dışarıda kontrolü yapar (daha hızlıdır).
    result = cv2.pointPolygonTest(polygon, point, False)
    return result >= 0