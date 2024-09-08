from fastapi import FastAPI
from image_processing_endpoints import router as image_processing_router

app = FastAPI(
    title="Image Processing API",
    version="1.0.0",
    root_path="/api",
    openapi_version="3.0.0"
)

app.include_router(image_processing_router)
