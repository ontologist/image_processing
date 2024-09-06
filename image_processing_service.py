import cv2
import numpy as np

def apply_otsu_threshold(image):
    """
    Applies Otsu's thresholding to the input grayscale image to segment
    the foreground (black) and background (white) automatically based on pixel intensity distribution.
    """
    _, binary_otsu = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary_otsu

def fill_closed_areas(binary_mask):
    """
    Detects closed contours in the input binary mask and fills them with black.
    """
    inverted = cv2.bitwise_not(binary_mask)
    contours, _ = cv2.findContours(inverted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled_image = np.zeros_like(binary_mask)
    cv2.drawContours(filled_image, contours, -1, (255), thickness=cv2.FILLED)
    filled_image = cv2.bitwise_not(filled_image)
    return filled_image

def process_image_file(image_path):
    """
    Processes the input image by performing Otsu's thresholding and filling closed areas.
    """
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    otsu_result = apply_otsu_threshold(gray)
    filled_otsu = fill_closed_areas(otsu_result)
    return otsu_result, filled_otsu
