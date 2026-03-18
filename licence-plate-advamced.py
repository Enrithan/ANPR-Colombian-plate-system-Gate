import re
import string
import easyocr
import cv2
import matplotlib.pyplot as plt

# Cargar imagen
image_path = 'Placas/1.jpg'
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Crear lector OCR
reader = easyocr.Reader(['en'])

# Detectar texto, string allowlist para performance
results = reader.readtext(image_rgb, allowlist=string.ascii_uppercase + string.digits)

# Lista para guardar posibles placas
detected_plates = []

# Iterar y filtrar
for (bbox, text, prob) in results:
    # Convertir a mayúsculas y eliminar espacios
    cleaned_text = text.upper().replace(" ", "")
    
    
     
    
    
    # Regex simple para placa tipo "ABC123" o "ABC-123"
    if re.match(r'^[A-Z]{3}[-]?[0-9]{3}$', cleaned_text):
        label = "✅ PLAUSIBLE"
    else:
        label = "❌ NO FORMATO"
        detected_plates.append((cleaned_text, prob))
        print(f"{label} → {cleaned_text} (Confianza: {prob:.2f})")
        # Dibujar en imagen
        (tl, tr, br, bl) = bbox
        tl = tuple(map(int, tl))
        br = tuple(map(int, br))
        cv2.rectangle(image, tl, br, (0, 255, 0), 2)
        cv2.putText(image, cleaned_text, (tl[0], tl[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# Mostrar resultados
if detected_plates:
    for plate, confidence in detected_plates:
        print(f"✅ Posible placa: {plate} (Confianza: {confidence:.2f})")
else:
    print("❌ No se detectaron placas con confianza suficiente.")

# Mostrar imagen procesada
plt.figure(figsize=(10, 6))
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("📸 Resultado Filtrado de Placa")
plt.axis("off")
plt.show()
