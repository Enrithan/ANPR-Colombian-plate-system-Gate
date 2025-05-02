import cv2 
from matplotlib import pyplot as plt
import numpy as np
import easyocr

import random


# Cargar imagen
image_path = 'Placas/1.jpg'
image = cv2.imread(image_path)

if image is None:
    print("❌ No se pudo cargar la imagen. Verifica la ruta.")
    exit()

# Convertir de BGR a RGB
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Crear el lector de EasyOCR
reader = easyocr.Reader(['es'])  # Puedes usar ['en', 'es'] si quieres español también

# Ejecutar OCR
results = reader.readtext(image_rgb)

# Mostrar resultados
for (bbox, text, prob) in results:
    print(f"📍 Texto detectado: {text} (Confianza: {prob:.2f})")

    # Dibujar la caja en la imagen
    (top_left, top_right, bottom_right, bottom_left) = bbox
    top_left = tuple(map(int, top_left))
    bottom_right = tuple(map(int, bottom_right))

    cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)
    cv2.putText(image, text, (top_left[0], top_left[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# Mostrar imagen con texto detectado
plt.figure(figsize=(10, 6))
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("📸 Resultado con EasyOCR")
plt.axis("off")
plt.show()