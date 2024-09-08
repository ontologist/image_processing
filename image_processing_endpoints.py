from fastapi import APIRouter, HTTPException, UploadFile, File
from image_processing_service import process_image_file  # Import the service function
import io

router = APIRouter()

@router.post("/process/")
async def process_image_endpoint(file: UploadFile = File(...)):
    """
    Endpoint to process an uploaded image using Otsu's thresholding and filling closed areas.
    """
    try:
        # Read the uploaded image
        contents = await file.read()
        temp_image_path = io.BytesIO(contents)

        # Process the image
        otsu_result, filled_otsu = process_image_file(temp_image_path)

        return {
            "otsu_result": otsu_result.tolist(),  # Convert numpy array to list for JSON serialization
            "filled_otsu": filled_otsu.tolist()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
