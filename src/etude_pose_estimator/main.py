"""FastAPI application entry point.

This module initializes the FastAPI application, sets up directories,
configures static file serving, and includes API routers.
"""

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from etude_pose_estimator.api import routes

# Create FastAPI application with lifespan from routes
app = FastAPI(
    title="etude-pose-estimator",
    description="Hero show pose quality management system (étude version)",
    version="0.1.0",
    lifespan=routes.lifespan,
)

# Setup templates
templates_dir = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(templates_dir))

# Setup static files (if directory exists)
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Include API routers
app.include_router(routes.router, prefix="/api", tags=["api"])
app.include_router(routes.page_router, tags=["pages"])


@app.get("/health")
async def health_check() -> JSONResponse:
    """Health check endpoint.

    Returns:
        JSON response with status
    """
    return JSONResponse(
        content={
            "status": "ok",
            "service": "etude-pose-estimator",
            "version": "0.1.0",
        }
    )
