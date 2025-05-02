import cv2
import pytesseract

# Ruta a Tesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Cargar imagen
image = cv2.imread('Placas/1.jpg')
image = cv2.resize(image, (800, 600))

# Recortar la zona estimada de la placa (ajustar si hace falta)
# Formato: [y1:y2, x1:x2]
plate_roi = image[280:370, 320:480]

# Procesar: escala de grises + threshold
gray = cv2.cvtColor(plate_roi, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)

# Mostrar la imagen de la placa recortada
cv2.imshow("Placa", thresh)

# OCR
config = r'--oem 3 --psm 7'
text = pytesseract.image_to_string(thresh, config=config)
cleaned_text = ''.join(filter(str.isalnum, text)).upper()

print(f"✅ Placa Detectada: {cleaned_text}")

cv2.waitKey(0)
cv2.destroyAllWindows()
