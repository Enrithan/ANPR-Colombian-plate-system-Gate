import cv2
import numpy as np
from matplotlib import pyplot as plt
import imutils

class PLR:
    def __init__(self, lower= np.uint8([15,100,100]), upper= np.uint8([35,255,255]),dimension=(500,200),plate = None):
        self.lower = lower
        self.upper = upper
        self.dimension = dimension
        self.plate = plate
        
    
    # convertir imagen a los diferente tonalidades para asi tener una mejor deteccion
    def __convertHSV(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    def __convertGREY(self,img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    def __convertRGB(self,img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    def __findContours(self, f_type, img):
        if f_type == 'external':
            return cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        elif f_type == 'all':
            return cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2]
        else:
            raise ValueError("Invalid contour type. Use 'external' or 'all'.")
    

