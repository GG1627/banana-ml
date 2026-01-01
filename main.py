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

    temp_path = None
    
    try:
        # More lenient file type validation
        if file.filename:
            file_ext = file.filename.lower().split('.')[-1] if '.' in file.filename else ''
            if file_ext not in ["jpg", "jpeg", "png"]:
                raise HTTPException(status_code=400, detail="Unsupported file type. Please upload a JPG or PNG image.")
        
        # Check content type as fallback
        if file.content_type and not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image.")

        # read file bytes
        contents = await file.read()
        
        if len(contents) == 0:
            raise HTTPException(status_code=400, detail="Empty file received.")

        # use secure temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            temp_file.write(contents)
            temp_path = temp_file.name

        try:
            # Log for debugging
            import os
            print(f"Processing image: {temp_path}, size: {os.path.getsize(temp_path)} bytes")
            
            # prediction function - pass the PATH string
            days_left = predict(temp_path)
            
            print(f"Prediction successful: {days_left} days")

            # get status
            status = get_status(days_left)

            return {
                "predictions": str(days_left),
                "status": status,
                "message": "Banana ripeness prediction successful"
            }

        except FileNotFoundError as e:
            import traceback
            print(f"File not found error: {str(e)}")
            print(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"Model or image file not found: {str(e)}")
        except Exception as predict_error:
            # Log the prediction error with full traceback
            import traceback
            print(f"Prediction error: {str(predict_error)}")
            print(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(predict_error)}")

        finally:
            # Clean up temp file
            import os
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception as cleanup_error:
                    print(f"Warning: Could not delete temp file: {cleanup_error}")

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        # Log the full error with traceback
        import traceback
        print(f"Upload error: {str(e)}")
        print(traceback.format_exc())
        
        # Clean up on error
        import os
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass
            
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

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