import cv2
import numpy as np
import matplotlib.pyplot as plt


# Function to apply Otsu's thresholding technique
def apply_otsu_threshold(image):
    """
    Applies Otsu's thresholding to the input grayscale image to segment
    the foreground (black) and background (white) automatically based on pixel intensity distribution.

    Parameters:
    image (numpy.ndarray): Grayscale input image.

    Returns:
    binary_otsu (numpy.ndarray): Binary image after Otsu's thresholding (foreground in black, background in white).
    """
    _, binary_otsu = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary_otsu


# Function to apply Sobel edge detection
def apply_sobel_edge_detection(image, ksize=3, scale=1, delta=0):
    """
    Applies Sobel edge detection on the input image to detect edges along the x and y axes.

    Parameters:
    image (numpy.ndarray): Grayscale input image.
    ksize (int): Size of the Sobel kernel. Larger values detect broader edges. Default is 3.
    scale (int): Optional scale factor for the computed derivative values. Default is 1.
    delta (int): Optional delta value added to the results before storing them. Default is 0.

    Returns:
    binary_sobel (numpy.ndarray): Binary image after Sobel edge detection.
    """
    # Calculate gradients along the x and y axes
    sobel_x = cv2.Sobel(image, cv2.CV_16S, 1, 0, ksize=ksize, scale=scale, delta=delta)
    sobel_y = cv2.Sobel(image, cv2.CV_16S, 0, 1, ksize=ksize, scale=scale, delta=delta)

    # Convert back to uint8
    sobel_x_abs = cv2.convertScaleAbs(sobel_x)
    sobel_y_abs = cv2.convertScaleAbs(sobel_y)

    # Combine the gradients
    sobel_edges = cv2.addWeighted(sobel_x_abs, 0.5, sobel_y_abs, 0.5, 0)

    # Convert edges to binary
    _, binary_sobel = cv2.threshold(sobel_edges, 35, 255, cv2.THRESH_BINARY)
    return binary_sobel


# Function to preprocess image by applying Gaussian blur
def preprocess_image(image, blur_ksize=5):
    """
    Preprocesses the input image by applying Gaussian blur to reduce noise and smooth the image.

    Parameters:
    image (numpy.ndarray): Grayscale input image.
    blur_ksize (int): Kernel size for the Gaussian blur. Should be odd, e.g., (3, 3), (5, 5). Default is (5, 5).

    Returns:
    blurred (numpy.ndarray): Blurred image after Gaussian smoothing.
    """
    blurred = cv2.GaussianBlur(image, (blur_ksize, blur_ksize), 0)
    return blurred


# Function to fill closed areas in a binary image
def fill_closed_areas(binary_mask):
    """
    Detects closed contours in the input binary mask and fills them with black.

    Parameters:
    binary_mask (numpy.ndarray): Binary mask where foreground is white (255) and background is black (0).

    Returns:
    filled_image (numpy.ndarray): Binary image with closed contours filled.
    """
    # Invert the binary mask so that contours appear as white regions
    inverted = cv2.bitwise_not(binary_mask)

    # Detect contours
    contours, _ = cv2.findContours(inverted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a new black image for filling contours
    filled_image = np.zeros_like(binary_mask)

    # Draw and fill the contours
    cv2.drawContours(filled_image, contours, -1, (255), thickness=cv2.FILLED)

    # Invert the image back to maintain black foreground, white background convention
    filled_image = cv2.bitwise_not(filled_image)

    return filled_image


# Main image processing function
def process_image(image_path):
    """
    Processes the input image by performing Otsu's thresholding and Sobel edge detection, then fills
    closed areas in the binary images produced by each method.

    Parameters:
    image_path (str): Path to the input image file.

    Returns:
    Tuple of images (otsu_result, filled_otsu, sobel_result, inverted_sobel_result, filled_sobel):
    - Otsu result
    - Filled Otsu result
    - Sobel result
    - Inverted Sobel result
    - Filled Sobel result
    """
    # Load image and convert it to grayscale
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Preprocess the grayscale image using Gaussian blur
    preprocessed = preprocess_image(gray)

    # Apply Otsu's thresholding
    otsu_result = apply_otsu_threshold(preprocessed)

    # Fill closed areas in the Otsu result
    filled_otsu = fill_closed_areas(otsu_result)

    # Apply Sobel edge detection
    sobel_result = apply_sobel_edge_detection(preprocessed)

    # Invert Sobel result to have black edges on a white background
    inverted_sobel_result = cv2.bitwise_not(sobel_result)

    # Fill closed areas in the Sobel result
    filled_sobel = fill_closed_areas(inverted_sobel_result)

    return otsu_result, filled_otsu, sobel_result, inverted_sobel_result, filled_sobel


# List of image paths to process (replace with your actual image paths)
image_paths = [
    r'C:\path_to_image_1.png',
    r'C:\path_to_image_2.png',
    r'C:\path_to_image_3.png'
]

# Process each image and display the results
fig, axes = plt.subplots(nrows=len(image_paths), ncols=6, figsize=(36, 12))

for i, image_path in enumerate(image_paths):
    # Get processed results for each image
    otsu_result, filled_otsu, sobel_result, inverted_sobel_result, filled_sobel = process_image(image_path)

    # Load original image for display
    original = cv2.imread(image_path)
    axes[i, 0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    axes[i, 0].set_title(f"Original - Image {i + 1}")
    axes[i, 0].axis('off')

    # Display Otsu threshold result
    axes[i, 1].imshow(otsu_result, cmap='gray')
    axes[i, 1].set_title(f"Otsu - Image {i + 1}")
    axes[i, 1].axis('off')

    # Display filled Otsu result
    axes[i, 2].imshow(filled_otsu, cmap='gray')
    axes[i, 2].set_title(f"Filled Otsu - Image {i + 1}")
    axes[i, 2].axis('off')

    # Display Sobel edge detection result
    axes[i, 3].imshow(sobel_result, cmap='gray')
    axes[i, 3].set_title(f"Sobel - Image {i + 1}")
    axes[i, 3].axis('off')

    # Display inverted Sobel result
    axes[i, 4].imshow(inverted_sobel_result, cmap='gray')
    axes[i, 4].set_title(f"Inverted Sobel - Image {i + 1}")
    axes[i, 4].axis('off')

    # Display filled Sobel result
    axes[i, 5].imshow(filled_sobel, cmap='gray')
    axes[i, 5].set_title(f"Filled Sobel - Image {i + 1}")
    axes[i, 5].axis('off')

# Adjust subplot layout to prevent overlap
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
