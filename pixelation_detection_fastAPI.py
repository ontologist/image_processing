from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from PIL import Image
import numpy as np
import cv2
import base64
from io import BytesIO
import matplotlib.pyplot as plt

app = FastAPI()

class ImageRequest(BaseModel):
    """
    This model represents the request body for image analysis. It expects a list of Base64 encoded images.

    Attributes:
        images_base64 (list of str): List of Base64 encoded image strings.
    """
    images_base64: list[str]

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

def detect_pixelation(base64_images, sensitivity=0.5, threshold=0.5):
    """
    Detects pixelation in the images using the Laplacian method.

    Parameters:
        base64_images (list of str): List of Base64 encoded images.
        sensitivity (float): Sensitivity for pixelation detection.
        threshold (float): Pixelation detection threshold.

    Returns:
        tuple: (results, laplacian_images)
            results (list of tuples): Each tuple contains (bool, float) indicating pixelation and its ratio.
            laplacian_images (list of np.ndarray): List of Laplacian variance images.
    """
    results = []
    laplacian_images = []

    for base64_image in base64_images:
        # Decode Base64 image and convert to OpenCV format
        image_data = base64.b64decode(base64_image)
        image = Image.open(BytesIO(image_data))
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

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
        pixelated = pixelation_ratio > threshold

        results.append((pixelated, pixelation_ratio))
        laplacian_images.append(binary_image)

    return results, laplacian_images

@app.post("/detect_pixelation/")
async def detect_pixelation_endpoint(request: ImageRequest):
    """
    Receives a list of Base64 encoded images, detects pixelation in each image, and returns the results.

    Parameters:
        request (ImageRequest): Request payload containing Base64 encoded images.

    Returns:
        dict: A dictionary containing results and visualizations.
    """
    try:
        # Detect pixelation in images
        results, laplacian_images = detect_pixelation(request.images_base64)

        # Visualization of results
        num_images = len(request.images_base64)
        cols = 2  # Original and highlighted images side by side
        rows = num_images  # Each image occupies two rows

        # Create a figure with subplots
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows), gridspec_kw={'wspace': 0.25, 'hspace': 1})
        fig.suptitle('Pixelation Detection Results', fontsize=16, ha='center')

        for i, (pixelated, pixelation_ratio) in enumerate(results):
            # Convert Base64 image back to OpenCV format for display
            image_data = base64.b64decode(request.images_base64[i])
            image = Image.open(BytesIO(image_data))
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

            # Resize laplacian image to match original image dimensions
            laplacian_resized = cv2.resize(laplacian_images[i], (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

            # Create an image highlighting pixelation
            highlight = np.zeros_like(image)
            highlight[laplacian_resized > 0] = [0, 0, 255]  # Red for pixelated areas

            # Plot original and highlighted images side by side
            axes[i, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            axes[i, 0].set_title(f'Original Image {i+1}')
            axes[i, 0].axis('off')

            axes[i, 1].imshow(cv2.cvtColor(highlight, cv2.COLOR_BGR2RGB))
            axes[i, 1].set_title(f'Pixelated: {pixelated}\nPixelation Ratio: {pixelation_ratio:.3f}')
            axes[i, 1].axis('off')
            
        # Save the visualization to a BytesIO object
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        plt.close(fig)

        # Encode visualization to Base64
        visualization_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

        return {"results": results, "visualization_base64": visualization_base64}
    
    except Exception as e:
        # Handle errors by returning an HTTP exception
        raise HTTPException(status_code=400, detail=str(e))

# To run the server, use the following command:
# uvicorn filename:app --reload
