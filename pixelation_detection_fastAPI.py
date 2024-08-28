from fastapi import FastAPI, HTTPException, UploadFile, File
from PIL import Image
import numpy as np
import cv2
import base64
from io import BytesIO

app = FastAPI()

def enhance_image(image: np.ndarray) -> np.ndarray:
    """
    Enhances the image by increasing resolution and applying edge enhancement.

    Parameters:
        image (np.ndarray): Input image in OpenCV format.

    Returns:
        np.ndarray: Enhanced image.
    """
    # Increase resolution by resizing
    scale_factor = 2  # Adjustable scale factor
    enhanced_image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)

    # Apply edge enhancement using a kernel
    kernel = np.array([[-1, -1, -1], 
                       [-1,  9, -1],
                       [-1, -1, -1]])
    enhanced_image = cv2.filter2D(enhanced_image, -1, kernel)

    return enhanced_image

def detect_pixelation(image: np.ndarray, sensitivity=0.5, threshold=0.5) -> bool:
    """
    Detects pixelation in the image using the Laplacian method.

    Parameters:
        image (np.ndarray): Input image in OpenCV format.
        sensitivity (float): Sensitivity for pixelation detection.
        threshold (float): Pixelation detection threshold.

    Returns:
        bool: True if the image is pixelated, False otherwise.
    """
    # Enhance the image before processing
    enhanced_image = enhance_image(image)

    # Convert to grayscale
    gray = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2GRAY)

    # Apply Laplacian filter and compute variance
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    variance = np.var(laplacian)

    # Normalize the variance value
    laplacian_abs = np.abs(laplacian)
    normalized_variance = np.mean(laplacian_abs)

    # Thresholding to create a binary image for pixelation
    _, binary_image = cv2.threshold(laplacian_abs, sensitivity * np.max(laplacian_abs), 255, cv2.THRESH_BINARY)
    binary_image = binary_image.astype(np.uint8)
    
    # Calculate pixelation ratio
    pixelated_area = np.sum(binary_image > 0)
    total_area = binary_image.size
    pixelation_ratio = pixelated_area / total_area

    # Determine if the image is pixelated based on the threshold
    return pixelation_ratio > threshold

@app.post("/detect_pixelation/")
async def detect_pixelation_endpoint(file: UploadFile = File(...)):
    """
    Receives a PNG image file, detects pixelation in the image, and returns the result.

    Parameters:
        file (UploadFile): PNG image file.

    Returns:
        dict: A dictionary containing the pixelated result.
    """
    try:
        # Read image from the uploaded file
        image_data = await file.read()
        image = Image.open(BytesIO(image_data))
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Detect pixelation
        pixelated = detect_pixelation(image)

        return {"pixelated": pixelated}

    except Exception as e:
        # Handle errors by returning an HTTP exception
        raise HTTPException(status_code=400, detail=str(e))

# To run the server, use the following command:
# uvicorn filename:app --reload
