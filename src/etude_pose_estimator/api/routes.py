"""API route handlers.

This module implements all API endpoints for pose detection, comparison,
and reference management.
"""

import shutil
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from etude_pose_estimator.config import settings
from etude_pose_estimator.core.advice import GeminiAdviceGenerator
from etude_pose_estimator.core.angle import JointAngleCalculator
from etude_pose_estimator.core.compare import ProcrustesComparator
from etude_pose_estimator.core.pose_2d import YOLO11xPoseDetector
from etude_pose_estimator.core.pose_3d import MotionBERTLifter
from etude_pose_estimator.core.reference import ReferencePoseManager

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


def initialize_components() -> dict[str, str]:
    """Initialize all components and return status messages.

    Returns:
        Dictionary with component names and initialization status
    """
    global detector, lifter, comparator, angle_calculator, advice_generator, reference_manager

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
            status["lifter"] = f"Model not found: {settings.motionbert_model_path}"
    except Exception as e:
        status["lifter"] = f"Not set up: {e}"

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
        advice_generator = GeminiAdviceGenerator(settings.gemini_api_key)
        status["advice_generator"] = "OK"
    except Exception as e:
        status["advice_generator"] = f"Error: {e}"

    # Initialize reference manager (always succeeds)
    try:
        reference_manager = ReferencePoseManager(settings.reference_dir)
        status["reference_manager"] = "OK"
    except Exception as e:
        status["reference_manager"] = f"Error: {e}"

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
    image: UploadFile = File(...),
    reference_name: str = Form(...),
) -> JSONResponse:
    """Compare uploaded pose with reference pose.

    Args:
        image: Uploaded image file
        reference_name: Name of reference pose to compare against

    Returns:
        JSON response containing:
        - similarity_score: Overall similarity (0.0-1.0)
        - angle_differences: Joint angle differences
        - comparison_metrics: Detailed comparison metrics
        - advice: Bilingual improvement advice (ja, en)

    Raises:
        HTTPException: If components not initialized, reference not found, or no pose detected
    """
    # Check if components are initialized
    if detector is None or lifter is None or comparator is None:
        raise HTTPException(
            status_code=503,
            detail="Required components not initialized",
        )
    if angle_calculator is None or advice_generator is None:
        raise HTTPException(
            status_code=503,
            detail="Angle calculator or advice generator not initialized",
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
        # Convert joint_distances list to dict for advice generation
        joint_distances_dict = {
            f"joint_{i}": dist for i, dist in enumerate(metrics["joint_distances"])
        }

        advice = advice_generator.generate_bilingual_advice(
            similarity_score=metrics["similarity_score"],
            angle_differences=angle_diffs,
            joint_distances=joint_distances_dict,
        )

        return JSONResponse(
            content={
                "similarity_score": metrics["similarity_score"],
                "angle_differences": angle_diffs,
                "comparison_metrics": {
                    "joint_distances": metrics["joint_distances"],
                    "mean_distance": metrics["mean_distance"],
                    "max_distance": metrics["max_distance"],
                    "max_distance_joint": metrics["max_distance_joint"],
                },
                "advice": advice,
            }
        )

    finally:
        # Always cleanup temporary file
        if file_path.exists():
            file_path.unlink()


@router.post("/reference")
async def register_reference(
    image: UploadFile = File(...),
    name: str = Form(...),
) -> JSONResponse:
    """Register new reference pose.

    Args:
        image: Uploaded image file
        name: Unique name for the reference pose

    Returns:
        JSON response with success message

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

    # Check if name already exists
    if reference_manager.exists(name):
        raise HTTPException(
            status_code=400,
            detail=f"Reference pose '{name}' already exists",
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

        # Save reference pose
        reference_manager.save(
            name=name,
            pose_2d=pose_2d,
            pose_3d=pose_3d,
            angles=angles,
            metadata={},
        )

        return JSONResponse(
            content={
                "name": name,
                "message": "Reference pose registered successfully",
            }
        )

    finally:
        # Always cleanup temporary file
        if file_path.exists():
            file_path.unlink()


@router.get("/references")
async def list_references() -> JSONResponse:
    """List all reference poses.

    Returns:
        JSON response with list of reference poses

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
        return JSONResponse(content=references)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list references: {e}",
        )
