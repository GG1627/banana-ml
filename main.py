from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import io
from typing import Dict

# helpers
from helpers.prediction import predict

def get_status(days_left: float) -> str:
    if days_left > 7:
        return "Very fresh! 🟢"
    elif days_left > 3:
        return "Fresh 🟡"
    elif days_left > 0:
        return "Getting ripe 🟠"
    else:
        return "Might be rotten already 🔴"

app = FastAPI(
    title="Banana Ripeness Predictor", 
    description="Predict the ripeness of a banana based on an image", 
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """API information"""
    return {
        "message": "Banana Ripeness Predictor API",
        "version": "1.0.0",
        "endpoints": {
            "POST /predict": "Upload image and get prediction",
            "GET /health": "Health check"
        }
    }

@app.post("/predict", response_model=Dict[str, str])
async def predict_banana_ripeness(file: UploadFile = File(...)):

    try:
        # validate file type
        if not file.filename.lower().endswith((".jpg", ".jpeg", ".png")):
            raise HTTPException(status_code=400, detail="Unsupported file type. Please upload a JPG or PNG image.")

        # read file bytes
        contents = await file.read()

        # use secure temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            temp_file.write(contents)
            temp_path = temp_file.name

        try:
            # prediction function
            days_left = predict(temp_file)

            # get stauts
            status = get_status(days_left)

            return {
                "predictions": "2.f",
                "status": status,
                "message": "Banana ripeness prediction successful"
            }

        finally:
            import os
            if os.path.exists(temp_file):
                os.remove(temp_file)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    """Detailed health check"""
    try:
        # test prediction with a known image
        import os
        test_image = "example_images/img0.jpg"
        if os.path.exists(test_image):
            predict(test_image)
            return {"status": "healthy", "model_loaded": True}
        else:
            return {"status": "healthy", "model_loaded": True, "note": "No test image available"}
    except Exception as e:
        return {"status": "unhealthy", "model_loaded": False, "error": str(e)}
