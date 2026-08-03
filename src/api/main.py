from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from src.api.dependencies import get_metrics, get_settings
from src.api.routes import router
from src.infrastructure.predictors import ONNXInference


def create_app() -> FastAPI:
    """Fábrica da aplicação FastAPI com injeção de dependências e estado."""
    settings = get_settings()
    app = FastAPI(
        title="Triagem de Laudos Médicos",
        version="1.0.0",
        description="API de triagem de laudos médicos com NLP e ONNX.",
    )
    app.state.settings = settings
    metrics = get_metrics()
    app.state.metrics = metrics
    onnx_path: Path = settings.onnx_model_path
    app.state.inference = ONNXInference(onnx_path)
    app.middleware("http")(metrics.middleware)
    app.include_router(router)
    return app


app = create_app()


@app.get("/", include_in_schema=False)
async def root() -> HTMLResponse:
    """Página inicial simples com links úteis."""
    return HTMLResponse(
        content="""
        <html><body>
            <h1>Triagem de Laudos Médicos</h1>
            <ul>
                <li><a href="/docs">Swagger UI</a></li>
                <li><a href="/health">Health</a></li>
                <li><a href="/metrics">Metrics</a></li>
            </ul>
        </body></html>
        """
    )
