import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt

def enhance_image(image):
    """
    Enhance the image by increasing resolution and applying edge enhancement.

    Parameters:
    - image: Input image in OpenCV format.

    Returns:
    - Enhanced image.
    """
    # Increase resolution by resizing
    scale_factor = 2  # You can adjust this value
    enhanced_image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)

    # Apply edge enhancement using a kernel
    kernel = np.array([[-1, -1, -1], 
                       [-1,  9, -1],
                       [-1, -1, -1]])
    enhanced_image = cv2.filter2D(enhanced_image, -1, kernel)

    return enhanced_image

def detect_pixelation(base64_images, sensitivity=0.5, threshold=0.5):
    """
    Detects pixelation in the entire image using the Laplacian method.

    Parameters:
    - base64_images: List of images in Base64 format.
    - sensitivity: Sensitivity threshold for pixelation detection (default is 0.5).
    - threshold: Pixelation detection threshold.

    Returns:
    - List of tuples containing (True if pixelation is detected, False otherwise, pixelation ratio) for each image.
    - List of Laplacian variance images.
    """
    results = []
    laplacian_images = []

    for base64_image in base64_images:
        # Convert Base64 image to OpenCV format
        image_data = base64.b64decode(base64_image)
        image = Image.open(BytesIO(image_data))
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Enhance the image before processing
        enhanced_image = enhance_image(image)

        # Convert image to grayscale
        gray = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2GRAY)

        # Apply Laplacian filter
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = np.var(laplacian)

        # Normalize the variance value to get a ratio of the image that is pixelated
        laplacian_abs = np.abs(laplacian)
        normalized_variance = np.mean(laplacian_abs)

        # Thresholding to create a binary image for pixelation
        _, binary_image = cv2.threshold(laplacian_abs, sensitivity * np.max(laplacian_abs), 255, cv2.THRESH_BINARY)
        binary_image = binary_image.astype(np.uint8)
        
        # Calculate pixelation ratio
        pixelated_area = np.sum(binary_image > 0)
        total_area = binary_image.size
        pixelation_ratio = pixelated_area / total_area

        # Detect pixelation based on the threshold
        pixelated = pixelation_ratio > threshold

        results.append((pixelated, pixelation_ratio))
        laplacian_images.append(binary_image)

    return results, laplacian_images

def convert_png_to_base64(image_paths):
    """
    Converts multiple PNG images to a list of Base64 strings.

    Parameters:
    - image_paths: List of paths to the PNG images.

    Returns:
    - List of Base64 strings of the images.
    """
    base64_strings = []
    for image_path in image_paths:
        with open(image_path, "rb") as image_file:
            base64_string = base64.b64encode(image_file.read()).decode("utf-8")
            base64_strings.append(base64_string)
    return base64_strings

def visualize_results(image_paths, sensitivity=0.5, threshold=0.5):
    """
    Visualizes the pixelation detection results for multiple images in a single subplot.

    Parameters:
    - image_paths: List of paths to the PNG images.
    - sensitivity: Sensitivity threshold for pixelation detection.
    - threshold: Pixelation detection threshold.
    """
    # Convert images to Base64
    base64_images = convert_png_to_base64(image_paths)

    # Detect pixelation
    results, laplacian_images = detect_pixelation(base64_images, sensitivity, threshold)

    # Determine grid size based on the number of images
    num_images = len(image_paths)
    cols = 2  # Original and highlighted images side by side
    rows = num_images  # Each image occupies two rows

    # Create a figure with subplots
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows), gridspec_kw={'wspace': 0.25, 'hspace': 1})
    fig.suptitle(f'Pixelation Detection Results\nSensitivity: {sensitivity:.2f}, Threshold: {threshold:.2f}', fontsize=16, ha='center')

    for i, (pixelated, pixelation_ratio) in enumerate(results):
        # Convert Base64 image back to OpenCV image for display
        image_data = base64.b64decode(base64_images[i])
        image = Image.open(BytesIO(image_data))
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Resize laplacian image to match the original image dimensions
        laplacian_resized = cv2.resize(laplacian_images[i], (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Create an image highlighting pixelation
        highlight = np.zeros_like(image)
        highlight[laplacian_resized > 0] = [0, 0, 255]  # Red for pixelated areas

        # Plot original and highlighted results side by side
        axes[i, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axes[i, 0].set_title(f'Original Image {i+1}')
        axes[i, 0].axis('off')

        axes[i, 1].imshow(cv2.cvtColor(highlight, cv2.COLOR_BGR2RGB))
        axes[i, 1].set_title(f'Pixelated: {pixelated}\nPixelation Ratio: {pixelation_ratio:.3f}\nThreshold: {threshold:.3f}')
        axes[i, 1].axis('off')
        
    plt.show()

# Example usage:
image_paths = ["C:/Users/Canip/Desktop/6.png", 
               "C:/Users/Canip/Desktop/5.png", 
               "C:/Users/Canip/Desktop/4.png",]  # Replace with your image paths

visualize_results(image_paths, sensitivity=0.5, threshold=0.05)
