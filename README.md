#FastAPI Pixelation Detection and Image Resizing
#Overview
This FastAPI service provides endpoints for:

Pixelation Detection: Detects pixelation in images and provides visualizations.
Image Resizing: Resizes images to a specified dimension if they exceed a certain size.
Endpoints
1. Pixelation Detection
Endpoint: /detect_pixelation

Method: POST

Request Body:

JSON object with a key images_base64 which is a list of Base64 encoded image strings.
Description: This endpoint detects pixelation in the provided Base64 encoded images. It returns both the detection results and a visual representation highlighting the pixelated areas.

Response Example:

results: A list with a tuple indicating whether pixelation was detected and the pixelation ratio.
visualization_base64: A Base64 encoded string of the visualization image.
Usage:

Convert your images to Base64 format.
Send a POST request to the endpoint with the Base64 encoded images.
Receive results and visualizations in the response.
2. Image Resizing
Endpoint: /resize_image

Method: POST

Request Body:

JSON object with a key image_base64 which is a Base64 encoded image string.
Description: This endpoint resizes the provided Base64 encoded image to ensure that its largest dimension is no more than 1000px. If the image is larger, it will be resized proportionally. The resized image is returned in Base64 format.

Response Example:

resized_image_base64: A Base64 encoded string of the resized image.
Usage:

Convert your image to Base64 format.
Send a POST request to the endpoint with the Base64 encoded image.
Receive the resized image in Base64 format.
Quick Example for Pixelation Detection
Request: Use Postman or cURL to send a POST request to /detect_pixelation with your Base64 encoded images.
Response: You'll get a JSON object with pixelation detection results and a Base64 encoded visualization image.
Quick Example for Image Resizing
Request: Use Postman or cURL to send a POST request to /resize_image with your Base64 encoded image.
Response: You'll get a JSON object with the resized image in Base64 format.
