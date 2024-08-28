Explanation and Usage

ImageRequest Class:

Defines the structure of the request payload which includes a Base64 encoded image. This helps in validating and documenting the expected input format.

resize_image Function:

Resizes the image proportionally so that the larger dimension is set to a specified maximum size (max_size). The aspect ratio is maintained to ensure that the image is not distorted. Uses the Image.LANCZOS filter for high-quality resizing.

resize_image_endpoint Function:

Handles POST requests to the /resize_image/ endpoint.
Receives a Base64 encoded image, decodes it, and opens it as a PIL.Image object.
Checks if resizing is needed based on the image's dimensions.
Resizes the image if necessary and converts it back to Base64 format.
Returns the Base64 encoded resized image in the response.
Handles exceptions by returning an appropriate HTTP error.
