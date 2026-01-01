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

    temp_path = None  # Initialize outside try block
    
    try:
        # validate file type
        if not file.filename.lower().endswith((".jpg", ".jpeg", ".png")):
            raise HTTPException(status_code=400, detail="Unsupported file type. Please upload a JPG or PNG image.")

        # read file bytes
        contents = await file.read()

        # use secure temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            temp_file.write(contents)
            temp_path = temp_file.name  # Store the path BEFORE exiting the with block

        # Now use temp_path (the string path) instead of temp_file (the closed file object)
        try:
            # prediction function - pass the PATH string, not the file object
            days_left = predict(temp_path)

            # get status
            status = get_status(days_left)

            return {
                "predictions": str(days_left),  # Also fix: convert to string, don't use hardcoded "2.f"
                "status": status,
                "message": "Banana ripeness prediction successful"
            }

        finally:
            # Clean up temp file using the PATH string
            import os
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

    except Exception as e:
        # Make sure to clean up even on error
        import os
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
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

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)