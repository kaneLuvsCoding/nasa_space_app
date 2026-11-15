from contextlib import asynccontextmanager
import asyncio
from datetime import datetime, timezone
import os
import logging
import tempfile
from pathlib import Path
from uuid import uuid4

from fastapi import Body, Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from sqlalchemy import select, text
from sqlalchemy.orm import Session

from .database import AstronautRow, Base, CheckInRow, InventoryRow, create_database
from .investigation import FixtureAdapter, InvestigationAdapter, UnknownFixture
from .iterative_investigation import (
    DispenseRequest,
    CrewProactiveAlertDTO,
    InvestigationSessionDTO,
    MissionContextDTO,
    MissionContextUpdate,
    StartSessionRequest,
    SubmitAnswerRequest,
    build_mission_context_dto,
    dispense_session_allocation,
    get_all_crew_proactive_alerts,
    get_latest_session,
    reset_mission_context,
    start_investigation_session,
    submit_session_answer,
    update_mission_context,
)
from .schemas import Astronaut, CheckIn, CheckInCreate, HistoryPoint, InventoryItem, InventoryUpdate, InvestigationRequest, InvestigationResponse
from .seed import seed
from .schemas import EvaluationRequest
from vendor.member3_stable.engine.models import InvestigationResponse as EngineResponse

logger = logging.getLogger(__name__)


def db(request: Request):
    with request.app.state.sessions() as session:
        yield session


def check_in_dto(row):
    # SQLite stores naive UTC timestamps; API consistently returns explicit UTC.
    record = CheckIn.model_validate(row)
    if record.recorded_at.tzinfo is None:
        record.recorded_at = record.recorded_at.replace(tzinfo=timezone.utc)
    return record


def require_astronaut(session, astronaut_id):
    row = session.get(AstronautRow, astronaut_id)
    if row is None:
        raise HTTPException(404, "Astronaut not found")
    return row


def profile(session, row):
    latest = session.scalar(select(CheckInRow).where(CheckInRow.astronaut_id == row.id).order_by(CheckInRow.recorded_at.desc(), CheckInRow.id.desc()).limit(1))
    readings = {key: getattr(latest, key) for key in ("hr", "sleep", "bp", "symptoms")} if latest else {}
    return Astronaut(id=row.id, role=row.role, baseline=row.baseline, sleepBaseline=row.sleepBaseline, provenance=latest.provenance if latest else "Synthetic profile v1", **readings)


def create_app(database_url: str | None = None, *, adapter: InvestigationAdapter | None = None, seed_demo: bool = True, investigation_mode: str | None = None, report_dir: str | Path | None = None):
    mode = investigation_mode if investigation_mode is not None else os.getenv("MEDNAV_INVESTIGATION_MODE", "fixture")
    if mode not in ("fixture", "real"):
        raise ValueError("Investigation mode must be fixture or real")
    if mode == "real" and adapter is not None:
        raise ValueError("Fixture adapter cannot be injected into real mode")
    default_path = Path(__file__).resolve().parents[1] / "mednav.db"
    url = database_url or os.getenv("MEDNAV_DATABASE_URL", f"sqlite:///{default_path.as_posix()}")
    engine, sessions = create_database(url)
    if report_dir:
        storage = Path(report_dir)
    elif os.getenv("MEDNAV_REPORT_DIR"):
        storage = Path(os.environ["MEDNAV_REPORT_DIR"])
    elif url.startswith("sqlite:///") and ":memory:" not in url:
        storage = Path(url.removeprefix("sqlite:///" )).parent / "reports"
    else:
        storage = Path(tempfile.mkdtemp(prefix="mednav-reports-"))

    @asynccontextmanager
    async def lifespan(app):
        Base.metadata.create_all(engine)
        if seed_demo:
            with sessions() as session:
                seed(session)
        with sessions() as session:
            from .reporting import backfill_investigation_medicines, transmit_pending
            backfill_investigation_medicines(session)
            transmit_pending(session, app)
        async def retry_worker():
            from .reporting import transmit_pending
            while True:
                await asyncio.sleep(2)
                def run():
                    with sessions() as session:
                        transmit_pending(session, app)
                await asyncio.to_thread(run)

        worker = asyncio.create_task(retry_worker())
        try:
            yield
        finally:
            worker.cancel()
            try:
                await worker
            except asyncio.CancelledError:
                pass
            engine.dispose()

    # Disable CDN-backed Swagger/ReDoc so even documentation endpoints are offline.
    app = FastAPI(title="MEDNAV Synthetic Backend", version="0.1.0", lifespan=lifespan, docs_url=None, redoc_url=None)
    app.state.sessions = sessions
    app.state.engine = engine
    app.state.report_dir = storage
    from .follow_ups import register_follow_ups
    register_follow_ups(app, db)
    from .reporting import register_reporting
    register_reporting(app, db)
    from .voice import router as voice_router
    app.include_router(voice_router)
    app.state.investigation_mode = mode
    from .real_engine import RealEngineService
    app.state.real_engine = RealEngineService() if mode == "real" else None
    app.state.investigation_adapter = (adapter if adapter is not None else FixtureAdapter()) if mode == "fixture" else None
    cors_origins = [
        o.strip()
        for o in os.getenv("MEDNAV_CORS_ORIGINS", "http://127.0.0.1:5173,http://localhost:5173").split(",")
        if o.strip()
    ]
    cors_regex = os.getenv("MEDNAV_CORS_ORIGIN_REGEX", r"https://.*\.vercel\.app")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_origin_regex=cors_regex or None,
        allow_methods=["GET", "POST", "PATCH", "OPTIONS"],
        allow_headers=["Content-Type", "X-MEDNAV-Officer-Key"],
    )

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, exc):
        # Do not echo medical input or non-JSON finite values in error responses.
        return JSONResponse(status_code=422, content={"detail": [{"loc": list(error["loc"]), "msg": error["msg"], "type": error["type"]} for error in exc.errors()]})

    @app.get("/api/v1/healthz")
    def healthz(session: Session = Depends(db)):
        session.execute(text("SELECT 1"))
        return {"status": "ok", "database": "sqlite", "synthetic": True, "investigation_mode": "adapter"}

    @app.get("/api/v1/astronauts", response_model=list[Astronaut])
    def astronauts(session: Session = Depends(db)):
        return [profile(session, row) for row in session.scalars(select(AstronautRow).order_by(AstronautRow.id))]

    @app.get("/api/v1/astronauts/{astronaut_id}", response_model=Astronaut)
    def astronaut(astronaut_id: str, session: Session = Depends(db)):
        return profile(session, require_astronaut(session, astronaut_id))

    @app.get("/api/v1/astronauts/{astronaut_id}/history", response_model=list[HistoryPoint])
    def history(astronaut_id: str, limit: int = Query(100, ge=1, le=1000), offset: int = Query(0, ge=0), session: Session = Depends(db)):
        require_astronaut(session, astronaut_id)
        rows = session.scalars(select(CheckInRow).where(CheckInRow.astronaut_id == astronaut_id).order_by(CheckInRow.recorded_at, CheckInRow.id).offset(offset).limit(limit))
        return [check_in_dto(row) for row in rows]

    @app.post("/api/v1/check-ins", response_model=CheckIn, status_code=201)
    def create_check_in(body: CheckInCreate, session: Session = Depends(db)):
        require_astronaut(session, body.astronaut_id)
        row = CheckInRow(**body.model_dump(), id=str(uuid4()), recorded_at=datetime.now(timezone.utc), synthetic=True, provenance="Synthetic user-entered MVP check-in")
        session.add(row)
        session.commit()
        return check_in_dto(row)

    @app.get("/api/v1/check-ins/{check_in_id}", response_model=CheckIn)
    def check_in(check_in_id: str, session: Session = Depends(db)):
        row = session.get(CheckInRow, check_in_id)
        if row is None:
            raise HTTPException(404, "Check-in not found")
        return check_in_dto(row)

    @app.get("/api/v1/inventory", response_model=list[InventoryItem])
    def inventory(session: Session = Depends(db)):
        return list(session.scalars(select(InventoryRow).order_by(InventoryRow.name)))

    @app.patch("/api/v1/inventory/{name}", response_model=InventoryItem)
    def update_inventory(name: str, body: InventoryUpdate, session: Session = Depends(db)):
        row = session.get(InventoryRow, name)
        if row is None:
            raise HTTPException(404, "Inventory item not found")
        updates = body.model_dump(exclude_unset=True)
        quantity = updates.get("quantity", row.quantity)
        available = updates.get("available", row.available)
        if quantity == 0 and available:
            raise HTTPException(422, "Zero-quantity equipment cannot be available; set available=false")
        for key, value in updates.items():
            setattr(row, key, value)
        session.commit()
        return row

    @app.post("/api/v1/investigations", response_model=InvestigationResponse)
    def investigate(body: InvestigationRequest, request: Request, session: Session = Depends(db)):
        if request.app.state.investigation_mode == "real":
            raise HTTPException(503, "Real engine is available internally; public investigation contract approval required")
        row = session.get(CheckInRow, body.check_in_id)
        if row is None:
            raise HTTPException(404, "Check-in not found")
        items = [InventoryItem.model_validate(item) for item in session.scalars(select(InventoryRow).order_by(InventoryRow.name))]
        try:
            result = request.app.state.investigation_adapter.investigate(check_in=check_in_dto(row), inventory=items, fixture_id=body.fixture_id)
        except UnknownFixture:
            raise HTTPException(422, "No demo fixture available") from None
        # Validate adapter outputs too, preserving invariants at the integration boundary.
        return InvestigationResponse.model_validate(result)

    @app.post("/api/v1/investigations/evaluate", response_model=EngineResponse)
    def evaluate(body: EvaluationRequest, session: Session = Depends(db)):
        if session.get(CheckInRow, body.check_in_id) is None:
            raise HTTPException(404, "Check-in not found")
        from .real_engine import RealEngineService, SimulationContext
        try:
            result = RealEngineService().evaluate(
                session, body.check_in_id,
                SimulationContext(communication_status=body.communication_status, emergency_override=body.emergency_override),
            )
            return EngineResponse.model_validate(result.model_dump(mode="json"))
        except (ImportError, FileNotFoundError):
            logger.exception("Stable evaluation service unavailable")
            raise HTTPException(503, "Evaluation service unavailable") from None
        except Exception:
            logger.exception("Stable evaluation failed")
            raise HTTPException(500, "Evaluation failed") from None

    @app.get("/api/v1/investigation/crew-alerts", response_model=list[CrewProactiveAlertDTO])
    def crew_alerts(session: Session = Depends(db)):
        return get_all_crew_proactive_alerts(session)

    @app.get("/api/v1/investigation/mission-context", response_model=MissionContextDTO)
    def get_mission_context(session: Session = Depends(db)):
        return build_mission_context_dto(session)

    @app.patch("/api/v1/investigation/mission-context", response_model=MissionContextDTO)
    def patch_mission_context(body: MissionContextUpdate, session: Session = Depends(db)):
        try:
            return update_mission_context(session, body)
        except LookupError as exc:
            raise HTTPException(404, str(exc)) from None

    @app.post("/api/v1/investigation/mission-context/reset", response_model=MissionContextDTO)
    def reset_mission_context_route(session: Session = Depends(db)):
        return reset_mission_context(session)

    @app.post("/api/v1/investigation/sessions", response_model=InvestigationSessionDTO, status_code=201)
    def start_session_route(body: StartSessionRequest, session: Session = Depends(db)):
        try:
            return start_investigation_session(session, body)
        except LookupError as exc:
            raise HTTPException(404, str(exc)) from None

    @app.get("/api/v1/investigation/sessions/latest/{astronaut_id}", response_model=InvestigationSessionDTO)
    def latest_session_route(astronaut_id: str, session: Session = Depends(db)):
        try:
            return get_latest_session(session, astronaut_id)
        except LookupError as exc:
            raise HTTPException(404, str(exc)) from None

    @app.post("/api/v1/investigation/sessions/{session_id}/answer", response_model=InvestigationSessionDTO)
    def answer_session_route(session_id: str, body: SubmitAnswerRequest, session: Session = Depends(db)):
        try:
            return submit_session_answer(session, session_id, body)
        except LookupError as exc:
            raise HTTPException(404, str(exc)) from None
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from None

    @app.post("/api/v1/investigation/sessions/{session_id}/dispense", response_model=InvestigationSessionDTO)
    def dispense_session_route(
        session_id: str,
        body: DispenseRequest | None = Body(default=None),
        session: Session = Depends(db),
    ):
        try:
            return dispense_session_allocation(
                session,
                session_id,
                medication_id=body.medication_id if body else None,
            )
        except LookupError as exc:
            raise HTTPException(404, str(exc)) from None
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from None

    return app


app = create_app()
