from fastapi import APIRouter, HTTPException, UploadFile, File
from image_processing_service import process_image_file  # Import the service function

router = APIRouter()


@router.post("/process/")
async def process_image_endpoint(file: UploadFile = File(...)):
    """
    Endpoint to process an uploaded image using Otsu's thresholding and filling closed areas.

    Parameters:
    - file: The image file to be processed.

    Returns:
    - dict: Processed image results including Otsu's thresholding and filled result.
    """
    try:
        # Read the uploaded image
        contents = await file.read()
        temp_image_path = "temp_image.png"
        with open(temp_image_path, "wb") as f:
            f.write(contents)

        # Process the image
        otsu_result, filled_otsu = process_image_file(temp_image_path)

        return {
            "otsu_result": otsu_result.tolist(),  # Convert numpy array to list for JSON serialization
            "filled_otsu": filled_otsu.tolist()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
