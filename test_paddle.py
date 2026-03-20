import os
import cv2
import sys
from vision.paddle_reader import PaddleOCRPlateReader

def test_paddle():
    print("Testing PaddleOCR initialization...")
    try:
        reader = PaddleOCRPlateReader()
        
        # Try to find an image in dataset
        img_path = None
        valid_dir = os.path.join('dataset', 'valid', 'images')
        if os.path.exists(valid_dir):
            for file in os.listdir(valid_dir):
                if file.endswith('.jpg'):
                    img_path = os.path.join(valid_dir, file)
                    break
        
        if not img_path:
            print("No image found to test.")
            return

        print(f"Testing on image: {img_path}")
        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to read image {img_path}")
            return
            
        # Mock crop: just use the whole image or crop the center
        h, w = img.shape[:2]
        crop = img[int(h*0.2):int(h*0.8), int(w*0.2):int(w*0.8)]
        
        text, score = reader.read_text(crop)
        print(f"\n--- Result ---\nText: {text}\nConfidence: {score}\n")

    except Exception as e:
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_paddle()
