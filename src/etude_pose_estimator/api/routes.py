"""API route handlers.

This module implements all API endpoints for pose detection, comparison,
and reference management.
"""

import base64
import json
import shutil
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import APIRouter, FastAPI, File, Form, HTTPException, Request, UploadFile
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


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Initialize and cleanup application resources.

    This lifespan context manager initializes all ML models and components
    on startup and stores them in app.state for use by endpoints.
    """
    # Create necessary directories
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    settings.reference_dir.mkdir(parents=True, exist_ok=True)

    status = {}

    # Initialize YOLO11x detector
    try:
        if settings.yolo_model_path.exists():
            app.state.detector = YOLO11xPoseDetector(settings.yolo_model_path)
            status["detector"] = "OK"
        else:
            app.state.detector = None
            status["detector"] = f"Model not found: {settings.yolo_model_path}"
    except Exception as e:
        app.state.detector = None
        status["detector"] = f"Error: {e}"

    # Initialize MotionBERT lifter (may fail if not set up)
    try:
        if settings.motionbert_model_path.exists():
            app.state.lifter = MotionBERTLifter(settings.motionbert_model_path)
            status["lifter"] = "OK"
        else:
            # Use placeholder implementation when model not available
            app.state.lifter = MotionBERTLifter(settings.motionbert_model_path)
            status["lifter"] = (
                f"Model not found: {settings.motionbert_model_path} (using placeholder)"
            )
    except Exception as e:
        # Use placeholder implementation on error
        app.state.lifter = MotionBERTLifter(settings.motionbert_model_path)
        status["lifter"] = f"Not set up: {e} (using placeholder)"

    # Initialize comparator (always succeeds)
    try:
        app.state.comparator = ProcrustesComparator()
        status["comparator"] = "OK"
    except Exception as e:
        app.state.comparator = None
        status["comparator"] = f"Error: {e}"

    # Initialize angle calculator (always succeeds)
    try:
        app.state.angle_calculator = JointAngleCalculator()
        status["angle_calculator"] = "OK"
    except Exception as e:
        app.state.angle_calculator = None
        status["angle_calculator"] = f"Error: {e}"

    # Initialize advice generator
    try:
        if settings.gemini_api_key:
            app.state.advice_generator = GeminiAdviceGenerator(settings.gemini_api_key)
            status["advice_generator"] = "OK"
        else:
            app.state.advice_generator = None
            status["advice_generator"] = "API key not set (placeholder advice will be used)"
    except Exception as e:
        app.state.advice_generator = None
        status["advice_generator"] = f"Error: {e} (placeholder advice will be used)"

    # Initialize reference manager (always succeeds)
    try:
        app.state.reference_manager = ReferencePoseManager(settings.reference_dir)
        status["reference_manager"] = "OK"
    except Exception as e:
        app.state.reference_manager = None
        status["reference_manager"] = f"Error: {e}"

    # Initialize pose visualizer (always succeeds)
    try:
        app.state.visualizer = PoseVisualizer()
        status["visualizer"] = "OK"
    except Exception as e:
        app.state.visualizer = None
        status["visualizer"] = f"Error: {e}"

    # Store initialization status
    app.state.component_status = status

    # Log initialization status
    print("Component initialization status:")
    for component, status_msg in status.items():
        print(f"  {component}: {status_msg}")

    yield

    # Cleanup (if needed)


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
    if request.app.state.reference_manager is not None:
        try:
            references = request.app.state.reference_manager.list_references()
        except Exception:
            pass

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "references": references,
            "component_status": request.app.state.component_status,
        },
    )


@router.post("/detect")
async def detect_pose(
    request: Request,
    image: UploadFile = File(...),
) -> JSONResponse:
    """Detect 3D pose from uploaded image.

    Args:
        request: FastAPI request object
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
    if request.app.state.detector is None:
        raise HTTPException(
            status_code=503,
            detail="YOLO11x detector not initialized. Run 'task download-models'",
        )
    if request.app.state.lifter is None:
        raise HTTPException(
            status_code=503,
            detail="MotionBERT lifter not initialized. Run 'task setup-motionbert'",
        )
    if request.app.state.angle_calculator is None:
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
        pose_2d = request.app.state.detector.detect(file_path)
        if pose_2d is None:
            raise HTTPException(
                status_code=400,
                detail="No pose detected in the image. Try adjusting lighting or camera angle.",
            )

        # Lift to 3D
        pose_3d = request.app.state.lifter.lift_2d_to_3d(pose_2d)

        # Calculate angles
        angles = request.app.state.angle_calculator.calculate_angles(pose_3d)

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
    if (
        request.app.state.detector is None
        or request.app.state.lifter is None
        or request.app.state.comparator is None
        or request.app.state.angle_calculator is None
        or request.app.state.reference_manager is None
    ):
        error_html = """
        <div class="alert alert-error">
          <svg fill="none" viewBox="0 0 24 24" class="stroke-current shrink-0 w-6 h-6">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
              d="M10 14l2-2m0 0l2-2m-2 2l-2-2m2 2l2 2m7-2a9 9 0 11-18 0 9 9 0 0118 0z"></path>
          </svg>
          <span>
            システムエラー: 必要なコンポーネントが初期化されていません。
            サーバーを再起動してください。
          </span>
        </div>
        """
        return HTMLResponse(content=error_html, status_code=503)

    # Validate file extension
    if not validate_file_extension(image.filename or ""):
        error_html = f"""
        <div class="alert alert-error">
          <svg fill="none" viewBox="0 0 24 24" class="stroke-current shrink-0 w-6 h-6">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
              d="M10 14l2-2m0 0l2-2m-2 2l-2-2m2 2l2 2m7-2a9 9 0 11-18 0 9 9 0 0118 0z"></path>
          </svg>
          <span>無効なファイル形式です。対応形式: {settings.allowed_extensions}</span>
        </div>
        """
        return HTMLResponse(content=error_html, status_code=400)

    # Load reference pose
    try:
        reference = request.app.state.reference_manager.load(reference_name)
    except FileNotFoundError:
        error_html = f"""
        <div class="alert alert-error">
          <svg fill="none" viewBox="0 0 24 24" class="stroke-current shrink-0 w-6 h-6">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
              d="M10 14l2-2m0 0l2-2m-2 2l-2-2m2 2l2 2m7-2a9 9 0 11-18 0 9 9 0 0118 0z"></path>
          </svg>
          <span>基準ポーズ「{reference_name}」が見つかりません。</span>
        </div>
        """
        return HTMLResponse(content=error_html, status_code=404)

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

        # Detect 2D pose with detailed info
        detection_result = request.app.state.detector.detect_detailed(file_path)
        if detection_result is None:
            error_html = """
            <div class="alert alert-warning">
              <svg fill="none" viewBox="0 0 24 24" class="stroke-current shrink-0 w-6 h-6">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
                  d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z">
                </path>
              </svg>
              <div>
                <div class="font-bold">ポーズを検出できませんでした</div>
                <div class="text-sm">以下を確認してください：</div>
                <ul class="text-sm list-disc list-inside mt-2">
                  <li>全身が写っていますか？</li>
                  <li>照明は十分ですか？</li>
                  <li>人物が画像の中央に配置されていますか？</li>
                  <li>背景がシンプルですか？</li>
                </ul>
              </div>
            </div>
            """
            return HTMLResponse(content=error_html, status_code=200)

        pose_2d, bbox, confidence = detection_result

        # Lift to 3D
        pose_3d = request.app.state.lifter.lift_2d_to_3d(pose_2d)

        # Calculate angles
        angles = request.app.state.angle_calculator.calculate_angles(pose_3d)

        # Compare with reference
        reference_pose_3d = reference["pose_3d"]
        reference_angles = reference["angles"]

        metrics = request.app.state.comparator.compare_with_metrics(pose_3d, reference_pose_3d)
        angle_diffs = request.app.state.angle_calculator.compare_angles(angles, reference_angles)

        # Generate skeleton visualization for uploaded image
        uploaded_skeleton_url = None
        if request.app.state.visualizer is not None:
            try:
                skeleton_path = settings.upload_dir / f"skeleton_compare_{file_path.name}"
                request.app.state.visualizer.draw_pose(
                    file_path,
                    pose_2d,
                    skeleton_path,
                    bbox=bbox,
                    confidence=confidence,
                    show_bbox=True,
                    show_keypoint_numbers=False,
                )
                # Read skeleton as base64
                with open(skeleton_path, "rb") as skel_file:
                    skeleton_data = base64.b64encode(skel_file.read()).decode("utf-8")
                    uploaded_skeleton_url = f"data:{mime_type};base64,{skeleton_data}"
                # Cleanup skeleton file
                skeleton_path.unlink()
            except Exception:
                # Continue without skeleton if visualization fails
                pass

        # Generate Japanese advice only
        if request.app.state.advice_generator is not None:
            # Convert joint_distances list to dict for advice generation
            joint_distances_dict = {
                f"joint_{i}": dist for i, dist in enumerate(metrics["joint_distances"])
            }
            ja_advice = request.app.state.advice_generator.generate_advice(
                similarity_score=metrics["similarity_score"],
                angle_differences=angle_diffs,
                joint_distances=joint_distances_dict,
                language="ja",
            )
            advice = {"ja": ja_advice}
        else:
            # Placeholder advice when Gemini is not available
            advice = {
                "ja": {
                    "overall": (
                        "Gemini APIが設定されていないため、詳細なアドバイスを生成できません。"
                        "類似度スコアと角度差分を参考にしてください。"
                    ),
                    "improvements": [
                        "Gemini APIキーを.envrcに設定してください",
                        "設定後、より詳細な改善アドバイスが表示されます",
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
                "uploaded_skeleton_url": uploaded_skeleton_url,
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
    if (
        request.app.state.detector is None
        or request.app.state.lifter is None
        or request.app.state.angle_calculator is None
    ):
        raise HTTPException(
            status_code=503,
            detail="Required components not initialized",
        )
    if request.app.state.reference_manager is None:
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
        name = request.app.state.reference_manager.generate_next_pose_name()
    else:
        name = name.strip()

    # Check if name already exists
    if request.app.state.reference_manager.exists(name):
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
        detection_result = request.app.state.detector.detect_detailed(file_path)
        if detection_result is None:
            # Return error HTML response (status 200 so htmx displays it)
            html_content = templates.get_template("register_result.html").render(
                {
                    "request": request,
                    "success": False,
                    "name": name,
                    "error_message": (
                        "画像からポーズを検出できませんでした。"
                        "照明やカメラアングルを調整してください。"
                    ),
                }
            )
            return HTMLResponse(content=html_content, status_code=200)

        pose_2d, bbox, confidence = detection_result

        # Lift to 3D
        pose_3d = request.app.state.lifter.lift_2d_to_3d(pose_2d)

        # Calculate angles
        angles = request.app.state.angle_calculator.calculate_angles(pose_3d)

        # Generate skeleton visualization with bbox and confidence
        if request.app.state.visualizer is not None:
            try:
                skeleton_path = settings.upload_dir / f"{name}_skeleton{file_path.suffix}"
                request.app.state.visualizer.draw_pose(
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
        request.app.state.reference_manager.save(
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
    if request.app.state.reference_manager is None:
        raise HTTPException(
            status_code=503,
            detail="Reference manager not initialized",
        )

    try:
        references = request.app.state.reference_manager.list_references()

        # Add image_filename and skeleton_filename info to each reference
        for ref in references:
            try:
                data = request.app.state.reference_manager.load(ref["name"])
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
async def get_reference_image(request: Request, name: str) -> FileResponse:
    """Get reference pose image.

    Args:
        request: FastAPI request object
        name: Name of the reference pose

    Returns:
        Image file response

    Raises:
        HTTPException: If reference manager not initialized or image not found
    """
    if request.app.state.reference_manager is None:
        raise HTTPException(
            status_code=503,
            detail="Reference manager not initialized",
        )

    try:
        image_path = request.app.state.reference_manager.get_image_path(name)
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
async def get_reference_skeleton(request: Request, name: str) -> FileResponse:
    """Get reference pose skeleton visualization.

    Args:
        request: FastAPI request object
        name: Name of the reference pose

    Returns:
        Skeleton image file response

    Raises:
        HTTPException: If reference manager not initialized or skeleton not found
    """
    if request.app.state.reference_manager is None:
        raise HTTPException(
            status_code=503,
            detail="Reference manager not initialized",
        )

    try:
        skeleton_path = request.app.state.reference_manager.get_skeleton_image_path(name)
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
    if request.app.state.reference_manager is None:
        raise HTTPException(
            status_code=503,
            detail="Reference manager not initialized",
        )

    try:
        request.app.state.reference_manager.delete(name)

        # Return updated reference list
        references = request.app.state.reference_manager.list_references()

        # Add image_filename and skeleton_filename info to each reference
        for ref in references:
            try:
                data = request.app.state.reference_manager.load(ref["name"])
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
