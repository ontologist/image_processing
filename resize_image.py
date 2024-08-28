from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image
from io import BytesIO
import base64

app = FastAPI()

class ImageRequest(BaseModel):
    """
    This model defines the structure of the request payload, which includes
    a Base64 encoded image.

    Attributes:
        image_base64 (str): The Base64 encoded image data as a string.
    """
    image_base64: str

def resize_image(image: Image.Image, max_size: int) -> Image.Image:
    """
    Resizes the image proportionally such that the larger dimension is set to max_size.

    This function maintains the aspect ratio of the image to prevent distortion during resizing.

    Args:
        image (Image.Image): The PIL.Image object to be resized.
        max_size (int): The maximum pixel value for the larger dimension of the image.

    Returns:
        Image.Image: The resized PIL.Image object.
    """
    width, height = image.size

    # Calculate new dimensions to ensure the larger dimension is max_size
    if width > height:
        new_width = max_size
        new_height = int((new_width / width) * height)
    else:
        new_height = max_size
        new_width = int((new_height / height) * width)

    return image.resize((new_width, new_height), Image.LANCZOS)

@app.post("/resize_image/")
async def resize_image_endpoint(request: ImageRequest):
    """
    Receives an image in Base64 format, checks if the image's larger dimension is less than 1000 pixels,
    and if so, resizes it proportionally to make the larger dimension exactly 1000 pixels. The resized
    image is then returned in Base64 format.

    Args:
        request (ImageRequest): The request payload containing the Base64 encoded image.

    Returns:
        dict: A dictionary containing the Base64 encoded resized image.
    
    Raises:
        HTTPException: If there is an error processing the image or decoding/encoding fails.
    """
    try:
        # Decode the Base64 encoded image
        image_data = base64.b64decode(request.image_base64)
        image = Image.open(BytesIO(image_data))

        # Get the dimensions of the image
        width, height = image.size

        # Resize the image if the larger dimension is less than 1000 pixels
        if max(width, height) < 1000:
            # Resize the image to make the larger dimension 1000 pixels
            resized_image = resize_image(image, 1000)
        else:
            # No resizing needed, use the original image
            resized_image = image

        # Convert the resized image to Base64
        buffer = BytesIO()
        resized_image.save(buffer, format=resized_image.format if resized_image.format else 'PNG')  # Default to PNG if format is unknown
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

        return {"image_base64": image_base64}
    
    except Exception as e:
        # Return an appropriate HTTP error in case of an exception
        raise HTTPException(status_code=400, detail=str(e))

# To run the server, use the following command:
# uvicorn filename:app --reload
