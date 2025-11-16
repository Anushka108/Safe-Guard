from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse, RedirectResponse
import numpy as np
import cv2

from backend.pose_extractor import PoseExtractor
from backend.risk_model import RiskModel

app = FastAPI(title="Pose Risk Video Pipeline", version="1.0")
app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")


@app.get("/")
async def serve_frontend_root():
    """Serve the frontend index page at the root URL so visiting
    http://127.0.0.1:8000 shows the app instead of a 404.
    """
    index_path = "frontend/index.html"
    try:
        return FileResponse(index_path)
    except Exception:
        # If file not found, redirect to mounted frontend path
        return RedirectResponse(url="/frontend/")

pose_model = PoseExtractor()
risk_model = RiskModel()

# Try to load StoryTeller, but continue if it fails
story_model = None
try:
    from backend.storyteller import StoryTeller
    story_model = StoryTeller()
except Exception as e:
    print(f"Warning: Could not load StoryTeller: {e}")

@app.post("/analyze")
async def analyze_video(file: UploadFile = File(...)):

    try:
        # Read uploaded video bytes
        contents = await file.read()
        np_arr = np.frombuffer(contents, np.uint8)

        # Decode as video file
        temp_file = "temp_video.mp4"
        with open(temp_file, "wb") as f:
            f.write(contents)

        cap = cv2.VideoCapture(temp_file)

        angle_seq = []

        # Extract 30 frames
        while len(angle_seq) < 30:
            ret, frame = cap.read()
            if not ret:
                break

            angles = pose_model.extract_angles(frame)
            if angles is not None:
                angle_seq.append(angles)

        cap.release()

        if len(angle_seq) < 30:
            return JSONResponse(
                {"error": "Video does not contain enough detectable human frames."},
                status_code=400
            )

        angle_seq = np.array(angle_seq)
        hip, knee, shoulder = angle_seq[-1]

        # Predict risk using LSTM
        risk = float(risk_model.predict_risk(angle_seq))

        # Generate LLaMA explanation
        story = ""
        if story_model:
            story = story_model.explain(hip, knee, shoulder, risk)
        else:
            story = "Story generation not available. Risk analysis complete."

        return {
            "hip_angle": float(hip),
            "knee_angle": float(knee),
            "shoulder_angle": float(shoulder),
            "risk": risk,
            "story": story
        }

    except Exception as e:
        import traceback
        return JSONResponse(
            {"error": str(e), "traceback": traceback.format_exc()},
            status_code=500
        )


@app.post("/analyze_angles")
async def analyze_angles(payload: dict):
    """Accept a JSON payload with an `angles` array for quick testing.

    Expected shape: list of 30 items each being [hip, knee, shoulder].
    This is a helper endpoint to test risk+story generation without
    uploading a video file.
    """
    try:
        angles = payload.get("angles") if isinstance(payload, dict) else None
        if not angles or len(angles) < 30:
            return JSONResponse({"error": "Provide 30 frames of angles under key 'angles'."}, status_code=400)

        angle_seq = np.array(angles)
        if angle_seq.shape[-1] != 3:
            return JSONResponse({"error": "Each angle entry must have 3 values (hip,knee,shoulder)."}, status_code=400)

        hip, knee, shoulder = angle_seq[-1]
        risk = float(risk_model.predict_risk(angle_seq))

        story = ""
        if story_model:
            story = story_model.explain(float(hip), float(knee), float(shoulder), risk)
        else:
            story = "Story generation not available. Risk analysis complete."

        return {
            "hip_angle": float(hip),
            "knee_angle": float(knee),
            "shoulder_angle": float(shoulder),
            "risk": risk,
            "story": story,
        }
    except Exception as e:
        import traceback
        return JSONResponse({"error": str(e), "traceback": traceback.format_exc()}, status_code=500)
