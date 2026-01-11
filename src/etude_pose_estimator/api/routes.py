"""API route handlers.

This module implements all API endpoints for pose detection, comparison,
and reference management.
"""

import base64
import json
import shutil
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from starlette.responses import Response

from etude_pose_estimator.config import settings
from etude_pose_estimator.core.advice import GeminiAdviceGenerator
from etude_pose_estimator.core.angle import JointAngleCalculator
from etude_pose_estimator.core.compare import ProcrustesComparator
from etude_pose_estimator.core.pose_2d import YOLO11xPoseDetector
from etude_pose_estimator.core.pose_3d import MotionBERTLifter
from etude_pose_estimator.core.reference import ReferencePoseManager
from etude_pose_estimator.core.visualization import PoseVisualizer

# Create routers
router = APIRouter()
page_router = APIRouter()

# Setup templates
templates_dir = Path(__file__).parent.parent / "templates"
templates = Jinja2Templates(directory=str(templates_dir))

# Initialize components
# Note: Some components may fail to initialize if dependencies are not set up
detector: YOLO11xPoseDetector | None = None
lifter: MotionBERTLifter | None = None
comparator: ProcrustesComparator | None = None
angle_calculator: JointAngleCalculator | None = None
advice_generator: GeminiAdviceGenerator | None = None
reference_manager: ReferencePoseManager | None = None
visualizer: PoseVisualizer | None = None


def initialize_components() -> dict[str, str]:
    """Initialize all components and return status messages.

    Returns:
        Dictionary with component names and initialization status
    """
    global detector, lifter, comparator, angle_calculator, advice_generator, reference_manager, visualizer

    status = {}

    # Initialize YOLO11x detector
    try:
        if settings.yolo_model_path.exists():
            detector = YOLO11xPoseDetector(settings.yolo_model_path)
            status["detector"] = "OK"
        else:
            status["detector"] = f"Model not found: {settings.yolo_model_path}"
    except Exception as e:
        status["detector"] = f"Error: {e}"

    # Initialize MotionBERT lifter (may fail if not set up)
    try:
        if settings.motionbert_model_path.exists():
            lifter = MotionBERTLifter(settings.motionbert_model_path)
            status["lifter"] = "OK"
        else:
            # Use placeholder implementation when model not available
            lifter = MotionBERTLifter(settings.motionbert_model_path)
            status["lifter"] = f"Model not found: {settings.motionbert_model_path} (using placeholder)"
    except Exception as e:
        # Use placeholder implementation on error
        lifter = MotionBERTLifter(settings.motionbert_model_path)
        status["lifter"] = f"Not set up: {e} (using placeholder)"

    # Initialize comparator (always succeeds)
    try:
        comparator = ProcrustesComparator()
        status["comparator"] = "OK"
    except Exception as e:
        status["comparator"] = f"Error: {e}"

    # Initialize angle calculator (always succeeds)
    try:
        angle_calculator = JointAngleCalculator()
        status["angle_calculator"] = "OK"
    except Exception as e:
        status["angle_calculator"] = f"Error: {e}"

    # Initialize advice generator
    try:
        if settings.gemini_api_key:
            advice_generator = GeminiAdviceGenerator(settings.gemini_api_key)
            status["advice_generator"] = "OK"
        else:
            advice_generator = None
            status["advice_generator"] = "API key not set (placeholder advice will be used)"
    except Exception as e:
        advice_generator = None
        status["advice_generator"] = f"Error: {e} (placeholder advice will be used)"

    # Initialize reference manager (always succeeds)
    try:
        reference_manager = ReferencePoseManager(settings.reference_dir)
        status["reference_manager"] = "OK"
    except Exception as e:
        status["reference_manager"] = f"Error: {e}"

    # Initialize pose visualizer (always succeeds)
    try:
        visualizer = PoseVisualizer()
        status["visualizer"] = "OK"
    except Exception as e:
        status["visualizer"] = f"Error: {e}"

    return status


# Initialize components on module load
COMPONENT_STATUS = initialize_components()


def validate_file_extension(filename: str) -> bool:
    """Validate file extension.

    Args:
        filename: Name of the uploaded file

    Returns:
        True if extension is allowed, False otherwise
    """
    allowed = settings.get_allowed_extensions_list()
    extension = Path(filename).suffix.lower()
    return extension in allowed


@page_router.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    """Render main page.

    Args:
        request: FastAPI request object

    Returns:
        Rendered HTML page
    """
    references = []
    if reference_manager is not None:
        try:
            references = reference_manager.list_references()
        except Exception:
            pass

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "references": references,
            "component_status": COMPONENT_STATUS,
        },
    )


@router.post("/detect")
async def detect_pose(
    image: UploadFile = File(...),
) -> JSONResponse:
    """Detect 3D pose from uploaded image.

    Args:
        image: Uploaded image file

    Returns:
        JSON response containing:
        - pose_2d: 2D keypoints (17, 3)
        - pose_3d: 3D coordinates (17, 3)
        - angles: Joint angles dictionary

    Raises:
        HTTPException: If components not initialized, file invalid, or no pose detected
    """
    # Check if components are initialized
    if detector is None:
        raise HTTPException(
            status_code=503,
            detail="YOLO11x detector not initialized. Run 'task download-models'",
        )
    if lifter is None:
        raise HTTPException(
            status_code=503,
            detail="MotionBERT lifter not initialized. Run 'task setup-motionbert'",
        )
    if angle_calculator is None:
        raise HTTPException(
            status_code=503,
            detail="Angle calculator not initialized",
        )

    # Validate file extension
    if not validate_file_extension(image.filename or ""):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {settings.allowed_extensions}",
        )

    # Save uploaded file temporarily
    file_path = settings.upload_dir / (image.filename or "temp.jpg")
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)

        # Detect 2D pose
        pose_2d = detector.detect(file_path)
        if pose_2d is None:
            raise HTTPException(
                status_code=400,
                detail="No pose detected in the image. Try adjusting lighting or camera angle.",
            )

        # Lift to 3D
        pose_3d = lifter.lift_2d_to_3d(pose_2d)

        # Calculate angles
        angles = angle_calculator.calculate_angles(pose_3d)

        return JSONResponse(
            content={
                "pose_2d": pose_2d.tolist(),
                "pose_3d": pose_3d.tolist(),
                "angles": angles,
            }
        )

    finally:
        # Always cleanup temporary file
        if file_path.exists():
            file_path.unlink()


@router.post("/compare")
async def compare_pose(
    request: Request,
    image: UploadFile = File(...),
    reference_name: str = Form(...),
) -> Response:
    """Compare uploaded pose with reference pose.

    Args:
        request: FastAPI request object
        image: Uploaded image file
        reference_name: Name of reference pose to compare against

    Returns:
        HTML fragment containing comparison results

    Raises:
        HTTPException: If components not initialized, reference not found, or no pose detected
    """
    # Check if components are initialized
    if detector is None or lifter is None or comparator is None:
        raise HTTPException(
            status_code=503,
            detail="Required components not initialized",
        )
    if angle_calculator is None:
        raise HTTPException(
            status_code=503,
            detail="Angle calculator not initialized",
        )
    if reference_manager is None:
        raise HTTPException(
            status_code=503,
            detail="Reference manager not initialized",
        )

    # Validate file extension
    if not validate_file_extension(image.filename or ""):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {settings.allowed_extensions}",
        )

    # Load reference pose
    try:
        reference = reference_manager.load(reference_name)
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"Reference pose '{reference_name}' not found",
        )

    # Save uploaded file temporarily
    file_path = settings.upload_dir / (image.filename or "temp.jpg")
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)

        # Read image as base64 for display in results
        with open(file_path, "rb") as img_file:
            image_data = base64.b64encode(img_file.read()).decode("utf-8")
            # Determine MIME type from extension
            extension = file_path.suffix.lower()
            mime_type = "image/jpeg" if extension in [".jpg", ".jpeg"] else "image/png"
            uploaded_image_url = f"data:{mime_type};base64,{image_data}"

        # Detect 2D pose
        pose_2d = detector.detect(file_path)
        if pose_2d is None:
            raise HTTPException(
                status_code=400,
                detail="No pose detected in the image. Try adjusting lighting or camera angle.",
            )

        # Lift to 3D
        pose_3d = lifter.lift_2d_to_3d(pose_2d)

        # Calculate angles
        angles = angle_calculator.calculate_angles(pose_3d)

        # Compare with reference
        reference_pose_3d = reference["pose_3d"]
        reference_angles = reference["angles"]

        metrics = comparator.compare_with_metrics(pose_3d, reference_pose_3d)
        angle_diffs = angle_calculator.compare_angles(angles, reference_angles)

        # Generate bilingual advice
        if advice_generator is not None:
            # Convert joint_distances list to dict for advice generation
            joint_distances_dict = {
                f"joint_{i}": dist for i, dist in enumerate(metrics["joint_distances"])
            }
            advice = advice_generator.generate_bilingual_advice(
                similarity_score=metrics["similarity_score"],
                angle_differences=angle_diffs,
                joint_distances=joint_distances_dict,
            )
        else:
            # Placeholder advice when Gemini is not available
            advice = {
                "ja": {
                    "overall": "Gemini APIが設定されていないため、詳細なアドバイスを生成できません。類似度スコアと角度差分を参考にしてください。",
                    "improvements": [
                        "Gemini APIキーを.envrcに設定してください",
                        "設定後、より詳細な改善アドバイスが表示されます",
                    ],
                    "priority_joints": "N/A",
                },
                "en": {
                    "overall": "Gemini API is not configured. Please refer to similarity score and angle differences.",
                    "improvements": [
                        "Configure Gemini API key in .envrc",
                        "After configuration, detailed advice will be available",
                    ],
                    "priority_joints": "N/A",
                },
            }

        return templates.TemplateResponse(
            "result.html",
            {
                "request": request,
                "reference_name": reference_name,
                "uploaded_image_url": uploaded_image_url,
                "similarity_score": metrics["similarity_score"],
                "angle_differences": angle_diffs,
                "comparison_metrics": {
                    "joint_distances": metrics["joint_distances"],
                    "mean_distance": metrics["mean_distance"],
                    "max_distance": metrics["max_distance"],
                    "max_distance_joint": metrics["max_distance_joint"],
                },
                "advice": advice,
            },
        )

    finally:
        # Always cleanup temporary file
        if file_path.exists():
            file_path.unlink()


@router.post("/reference")
async def register_reference(
    request: Request,
    image: UploadFile = File(...),
    name: str = Form(""),
) -> Response:
    """Register new reference pose.

    Args:
        request: FastAPI request object
        image: Uploaded image file
        name: Unique name for the reference pose

    Returns:
        HTML fragment with registration result

    Raises:
        HTTPException: If components not initialized, name exists, or no pose detected
    """
    # Check if components are initialized
    if detector is None or lifter is None or angle_calculator is None:
        raise HTTPException(
            status_code=503,
            detail="Required components not initialized",
        )
    if reference_manager is None:
        raise HTTPException(
            status_code=503,
            detail="Reference manager not initialized",
        )

    # Validate file extension
    if not validate_file_extension(image.filename or ""):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {settings.allowed_extensions}",
        )

    # Auto-generate name if not provided
    if not name or name.strip() == "":
        name = reference_manager.generate_next_pose_name()
    else:
        name = name.strip()

    # Check if name already exists
    if reference_manager.exists(name):
        raise HTTPException(
            status_code=400,
            detail=f"Reference pose '{name}' already exists",
        )

    # Save uploaded file temporarily
    file_path = settings.upload_dir / (image.filename or "temp.jpg")
    skeleton_path = None  # Initialize here to avoid UnboundLocalError in finally block

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)

        # Detect 2D pose with details
        detection_result = detector.detect_detailed(file_path)
        if detection_result is None:
            # Return error HTML response (status 200 so htmx displays it)
            html_content = templates.get_template("register_result.html").render(
                {
                    "request": request,
                    "success": False,
                    "name": name,
                    "error_message": "画像からポーズを検出できませんでした。照明やカメラアングルを調整してください。",
                }
            )
            return HTMLResponse(content=html_content, status_code=200)

        pose_2d, bbox, confidence = detection_result

        # Lift to 3D
        pose_3d = lifter.lift_2d_to_3d(pose_2d)

        # Calculate angles
        angles = angle_calculator.calculate_angles(pose_3d)

        # Generate skeleton visualization with bbox and confidence
        if visualizer is not None:
            try:
                skeleton_path = settings.upload_dir / f"{name}_skeleton{file_path.suffix}"
                visualizer.draw_pose(
                    file_path,
                    pose_2d,
                    skeleton_path,
                    bbox=bbox,
                    confidence=confidence,
                    show_bbox=True,
                    show_keypoint_numbers=False,
                )
            except Exception:
                # Continue without skeleton if visualization fails
                skeleton_path = None

        # Save reference pose with image and skeleton
        reference_manager.save(
            name=name,
            pose_2d=pose_2d,
            pose_3d=pose_3d,
            angles=angles,
            metadata={},
            image_path=file_path,
            skeleton_image_path=skeleton_path,
        )

        # Render template
        html_content = templates.get_template("register_result.html").render(
            {
                "request": request,
                "success": True,
                "name": name,
                "error_message": None,
            }
        )

        # Return with HX-Trigger header to automatically show skeleton modal
        return HTMLResponse(
            content=html_content,
            headers={
                "HX-Trigger": json.dumps({"showRegisteredSkeleton": {"name": name}}),
            },
        )

    except Exception as e:
        # Catch any unexpected errors and return user-friendly error message
        error_msg = f"登録中にエラーが発生しました: {str(e)}"
        html_content = templates.get_template("register_result.html").render(
            {
                "request": request,
                "success": False,
                "name": name,
                "error_message": error_msg,
            }
        )
        return HTMLResponse(content=html_content, status_code=200)

    finally:
        # Always cleanup temporary files
        if file_path.exists():
            file_path.unlink()
        if skeleton_path and skeleton_path.exists():
            skeleton_path.unlink()


@router.get("/references", response_model=None)
async def list_references(request: Request, format: str = "json") -> Response:
    """List all reference poses.

    Args:
        request: FastAPI request object
        format: Response format ('json' or 'html')

    Returns:
        JSON response or HTML fragment with list of reference poses

    Raises:
        HTTPException: If reference manager not initialized
    """
    if reference_manager is None:
        raise HTTPException(
            status_code=503,
            detail="Reference manager not initialized",
        )

    try:
        references = reference_manager.list_references()

        # Add image_filename and skeleton_filename info to each reference
        for ref in references:
            try:
                data = reference_manager.load(ref["name"])
                ref["image_filename"] = data.get("image_filename")
                ref["skeleton_filename"] = data.get("skeleton_filename")
            except Exception:
                ref["image_filename"] = None
                ref["skeleton_filename"] = None

        # Return HTML fragment for htmx requests
        if format == "html" or request.headers.get("HX-Request"):
            return templates.TemplateResponse(
                "reference_list.html",
                {
                    "request": request,
                    "references": references,
                },
            )

        # Return JSON for API requests
        return JSONResponse(content=references)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list references: {e}",
        )


@router.get("/references/{name}/image")
async def get_reference_image(name: str) -> FileResponse:
    """Get reference pose image.

    Args:
        name: Name of the reference pose

    Returns:
        Image file response

    Raises:
        HTTPException: If reference manager not initialized or image not found
    """
    if reference_manager is None:
        raise HTTPException(
            status_code=503,
            detail="Reference manager not initialized",
        )

    try:
        image_path = reference_manager.get_image_path(name)
        if image_path is None:
            raise HTTPException(
                status_code=404,
                detail=f"Image for reference pose '{name}' not found",
            )

        return FileResponse(
            path=image_path,
            media_type="image/jpeg",
            filename=image_path.name,
        )
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"Reference pose '{name}' not found",
        )


@router.get("/references/{name}/skeleton")
async def get_reference_skeleton(name: str) -> FileResponse:
    """Get reference pose skeleton visualization.

    Args:
        name: Name of the reference pose

    Returns:
        Skeleton image file response

    Raises:
        HTTPException: If reference manager not initialized or skeleton not found
    """
    if reference_manager is None:
        raise HTTPException(
            status_code=503,
            detail="Reference manager not initialized",
        )

    try:
        skeleton_path = reference_manager.get_skeleton_image_path(name)
        if skeleton_path is None:
            raise HTTPException(
                status_code=404,
                detail=f"Skeleton image for reference pose '{name}' not found",
            )

        return FileResponse(
            path=skeleton_path,
            media_type="image/jpeg",
            filename=skeleton_path.name,
        )
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"Reference pose '{name}' not found",
        )


@router.delete("/references/{name}", response_model=None)
async def delete_reference(request: Request, name: str) -> Response:
    """Delete a reference pose.

    Args:
        request: FastAPI request object
        name: Name of the reference pose to delete

    Returns:
        HTML fragment with updated reference list

    Raises:
        HTTPException: If reference manager not initialized or reference not found
    """
    if reference_manager is None:
        raise HTTPException(
            status_code=503,
            detail="Reference manager not initialized",
        )

    try:
        reference_manager.delete(name)

        # Return updated reference list
        references = reference_manager.list_references()

        # Add image_filename and skeleton_filename info to each reference
        for ref in references:
            try:
                data = reference_manager.load(ref["name"])
                ref["image_filename"] = data.get("image_filename")
                ref["skeleton_filename"] = data.get("skeleton_filename")
            except Exception:
                ref["image_filename"] = None
                ref["skeleton_filename"] = None

        return templates.TemplateResponse(
            "reference_list.html",
            {
                "request": request,
                "references": references,
            },
        )
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"Reference pose '{name}' not found",
        )
