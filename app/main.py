"""Main FastAPI application for pose estimation."""

import json
import os
import shutil
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from app.pose_analyzer import PoseAnalyzer
from app.pose_estimator import PoseEstimator

app = FastAPI(title="Pose Estimator", version="0.1.0")

# Setup directories
UPLOAD_DIR = Path("uploads")
REFERENCE_DIR = Path("reference_poses")
UPLOAD_DIR.mkdir(exist_ok=True)
REFERENCE_DIR.mkdir(exist_ok=True)

# Setup templates
templates = Jinja2Templates(directory="app/templates")

# Initialize pose estimator and analyzer
pose_estimator = PoseEstimator()
pose_analyzer = PoseAnalyzer()


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Render the main page."""
    # List available reference poses
    reference_poses = []
    if REFERENCE_DIR.exists():
        for ref_file in REFERENCE_DIR.glob("*.json"):
            reference_poses.append(ref_file.stem)

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "reference_poses": reference_poses,
        },
    )


@app.post("/analyze")
async def analyze_pose(
    request: Request,
    image: UploadFile = File(...),
    reference_pose: str | None = Form(None),
):
    """Analyze pose from uploaded image."""
    try:
        # Save uploaded file temporarily with secure filename
        file_extension = Path(image.filename).suffix if image.filename else ".jpg"
        secure_filename = f"{uuid.uuid4()}{file_extension}"
        file_path = UPLOAD_DIR / secure_filename
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)

        # Detect pose landmarks
        landmarks = pose_estimator.detect_pose(str(file_path))

        if landmarks is None:
            return templates.TemplateResponse(
                "result.html",
                {
                    "request": request,
                    "error": "No pose detected in the image.",
                },
            )

        # Calculate joint angles
        angles = pose_analyzer.calculate_joint_angles(landmarks)

        # Compare with reference pose if provided
        similarity_score = None
        angle_differences = None
        ref_angles = None

        if reference_pose:
            ref_path = REFERENCE_DIR / f"{reference_pose}.json"
            if ref_path.exists():
                with open(ref_path) as f:
                    ref_data = json.load(f)
                    ref_landmarks = ref_data["landmarks"]

                # Calculate Procrustes similarity
                similarity_score = pose_analyzer.calculate_similarity(landmarks, ref_landmarks)

                # Calculate angle differences and reference angles
                ref_angles = pose_analyzer.calculate_joint_angles(ref_landmarks)
                angle_differences = {}
                for joint in angles:
                    if joint in ref_angles:
                        angle_differences[joint] = abs(angles[joint] - ref_angles[joint])

        # Clean up uploaded file
        os.remove(file_path)

        return templates.TemplateResponse(
            "result.html",
            {
                "request": request,
                "angles": angles,
                "similarity_score": similarity_score,
                "angle_differences": angle_differences,
                "reference_pose": reference_pose,
                "ref_angles": ref_angles,
            },
        )

    except Exception as e:
        return templates.TemplateResponse(
            "result.html",
            {
                "request": request,
                "error": f"Error analyzing pose: {str(e)}",
            },
        )


@app.post("/register-reference")
async def register_reference_pose(
    request: Request,
    name: str = Form(...),
    image: UploadFile = File(...),
):
    """Register a new reference pose."""
    try:
        # Save uploaded file temporarily with secure filename
        file_extension = Path(image.filename).suffix if image.filename else ".jpg"
        secure_filename = f"{uuid.uuid4()}{file_extension}"
        file_path = UPLOAD_DIR / secure_filename
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)

        # Detect pose landmarks
        landmarks = pose_estimator.detect_pose(str(file_path))

        if landmarks is None:
            return templates.TemplateResponse(
                "register_result.html",
                {
                    "request": request,
                    "error": "No pose detected in the image.",
                },
            )

        # Save reference pose
        ref_path = REFERENCE_DIR / f"{name}.json"
        with open(ref_path, "w") as f:
            json.dump({"name": name, "landmarks": landmarks}, f, indent=2)

        # Clean up uploaded file
        os.remove(file_path)

        return templates.TemplateResponse(
            "register_result.html",
            {
                "request": request,
                "success": True,
                "name": name,
            },
        )

    except Exception as e:
        return templates.TemplateResponse(
            "register_result.html",
            {
                "request": request,
                "error": f"Error registering reference pose: {str(e)}",
            },
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
