"""FastAPI application: the production HTTP surface.

Features: request-id tracing, structured logging, optional API-key auth,
latency metrics, and clean error handling. The RAG pipeline is built once at
startup and shared across requests (stateless replicas scale horizontally).
"""
import time
import uuid

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse

from .. import __version__
from ..config import Settings, get_settings
from ..logging_setup import configure_logging, get_logger
from ..models import HealthResponse, QueryRequest, QueryResponse
from .deps import Container, build_container

log = get_logger("api")


def create_app(settings: Settings | None = None, container: Container | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level)
    app = FastAPI(title="system-design-production-grade", version=__version__)
    container = container or build_container(settings)
    settings = container.settings
    app.state.container = container

    def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
        if settings.api_key and x_api_key != settings.api_key:
            raise HTTPException(status_code=401, detail="invalid or missing X-API-Key")

    @app.middleware("http")
    async def add_request_id(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID", uuid.uuid4().hex[:12])
        request.state.request_id = request_id
        start = time.perf_counter()
        response = await call_next(request)
        latency_ms = int((time.perf_counter() - start) * 1000)
        response.headers["X-Request-ID"] = request_id
        log.info(
            "request",
            extra={
                "request_id": request_id,
                "route": request.url.path,
                "latency_ms": latency_ms,
            },
        )
        return response

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception):  # noqa: ANN001
        rid = getattr(request.state, "request_id", "-")
        log.exception("unhandled error", extra={"request_id": rid})
        return JSONResponse(
            status_code=500, content={"detail": "internal error", "request_id": rid}
        )

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        c: Container = app.state.container
        return HealthResponse(
            status="ok",
            version=__version__,
            vectorstore=c.settings.vectorstore,
            embedding_provider=c.settings.embedding_provider,
            collection_count=c.store.count(),
            generator_model=c.settings.gemini_model,
        )

    @app.post("/query", response_model=QueryResponse, dependencies=[Depends(require_api_key)])
    def query(req: QueryRequest, request: Request) -> QueryResponse:
        c: Container = app.state.container
        rid = getattr(request.state, "request_id", uuid.uuid4().hex[:12])
        start = time.perf_counter()
        result = c.pipeline.query(req.question, req.history, req.top_k)
        latency_ms = int((time.perf_counter() - start) * 1000)
        log.info(
            "query",
            extra={
                "request_id": rid,
                "stage": "query",
                "grounded": result["grounded"],
                "n_sources": len(result["sources"]),
                "model": result["model"],
                "latency_ms": latency_ms,
            },
        )
        return QueryResponse(
            **result, request_id=rid, latency_ms=latency_ms
        )

    return app

