"""FastAPI app factory, router registration, X-API-Key auth, and error envelopes (API_SPEC.md)."""

from __future__ import annotations

import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from oasis.api.routes_bench import router as bench_router
from oasis.api.routes_estimate import router as estimate_router
from oasis.api.routes_ops import router as ops_router
from oasis.api.routes_runs import router as runs_router
from oasis.db.migrate import apply_migrations
from oasis.rbe.reducer import BudgetInfeasibleError


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Run database migrations on application startup."""
    apply_migrations()
    yield


def create_app() -> FastAPI:
    """Instantiate and configure the OASIS FastAPI application."""
    application = FastAPI(
        title="OASIS",
        description="Orchestrating Agents with Smart, Intelligent Scaling",
        version="0.1.0",
        lifespan=lifespan,
    )

    # CORS support for dashboard
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # -----------------------------------------------------------------
    # Authentication Middleware (X-API-Key)
    # -----------------------------------------------------------------

    @application.middleware("http")
    async def api_key_auth_middleware(request: Request, call_next: Any) -> Response:
        """Enforce static X-API-Key header except on public endpoints (docs, healthz)."""
        exempt_paths = {"/healthz", "/docs", "/openapi.json", "/redoc"}
        if request.url.path in exempt_paths:
            return await call_next(request)

        expected_key = os.environ.get("OASIS_API_KEY", "dev-local-key")
        provided_key = request.headers.get("X-API-Key")

        if not provided_key or provided_key != expected_key:
            return JSONResponse(
                status_code=401,
                content={
                    "error": {
                        "code": "authentication_required",
                        "message": "Missing or invalid X-API-Key header",
                        "details": {},
                    }
                },
            )

        return await call_next(request)

    # -----------------------------------------------------------------
    # Global Exception Handlers (Standard Error Envelope)
    # -----------------------------------------------------------------

    @application.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        """Translate FastAPI validation errors into standard error envelope."""
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "code": "validation_error",
                    "message": "Request validation failed",
                    "details": {"errors": exc.errors()},
                }
            },
        )

    @application.exception_handler(BudgetInfeasibleError)
    async def budget_infeasible_handler(
        request: Request, exc: BudgetInfeasibleError
    ) -> JSONResponse:
        """Translate budget infeasibility into standard error envelope."""
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "code": "budget_infeasible",
                    "message": str(exc),
                    "details": {"binding": "budget"},
                }
            },
        )

    @application.exception_handler(HTTPException)
    async def http_exception_handler(
        request: Request, exc: HTTPException
    ) -> JSONResponse:
        """Ensure all HTTPExceptions follow the single error envelope."""
        details: dict[str, Any] = {}
        if isinstance(exc.detail, dict) and "code" in exc.detail:
            code = str(exc.detail["code"])
            message = str(exc.detail.get("message", "An error occurred"))
            details = dict(exc.detail.get("details", {}))
        else:
            code = "error" if exc.status_code < 500 else "internal_error"
            message = str(exc.detail)

        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"code": code, "message": message, "details": details}},
        )

    # -----------------------------------------------------------------
    # Router Registrations
    # -----------------------------------------------------------------

    application.include_router(ops_router)
    application.include_router(estimate_router)
    application.include_router(runs_router)
    application.include_router(bench_router)

    return application


app = create_app()
