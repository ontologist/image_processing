# Image Processing: Otsu Thresholding, Sobel Edge Detection, and Filling Algorithm

This project implements a pipeline for image processing using Otsu's thresholding, Sobel edge detection, and a filling algorithm for closed shapes. The goal is to separate the foreground from the background, detect edges, and fill any closed shapes found in the image with black. The project uses OpenCV and Matplotlib libraries for image processing and visualization.

## Overview

The code provided processes images using the following steps:

1. **Preprocessing:** Apply Gaussian blur to reduce noise.
2. **Otsu's Thresholding:** Automatically segments the image into foreground and background.
3. **Sobel Edge Detection:** Detects the edges in the image along the x and y axes.
4. **Inversion:** Inverts the Sobel result to prepare for further processing.
5. **Filling Algorithm:** Fills any closed areas in the thresholded and Sobel-processed images.
6. **Visualization:** Displays the original image, Otsu thresholding result, Sobel edge detection result, and the filled images using subplots for comparison.

## Libraries Required

Before running the code, ensure you have the following Python libraries installed:

- `opencv-python`
- `numpy`
- `matplotlib`

You can install these libraries using the following command:

```bash
pip install opencv-python numpy matplotlib
