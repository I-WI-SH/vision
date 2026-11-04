"""APT training API. Run from vision with: python -m uvicorn api.main:app"""

import json
import os
import signal
import subprocess
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import PlainTextResponse
from fastapi.responses import JSONResponse


class NewlineJsonResponse(JSONResponse):
    def render(self, content) -> bytes:
        return super().render(content) + b"\n\n"

VISION_ROOT = Path(__file__).resolve().parents[1]


from .training import APTRequest, build_command, now


class JobManager:
    """One training process at a time; metadata and logs survive API restarts.

    Use one Uvicorn worker. This is a local service, not a distributed scheduler.
    """

    def __init__(self, root, storage):
        self.root = Path(root).resolve()
        self.storage = Path(storage).resolve()
        self.storage.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.jobs = {}
        self.process = None
        self.active = None
        self.watcher = None
        self.process_group = None
        self.closing = False
        for path in self.storage.glob("*/job.json"):
            job = json.loads(path.read_text(encoding="utf-8"))
            if job["status"] in ("running", "stopping"):
                # A stale PID must never be used to kill an unrelated process.
                job.update(status="interrupted", finished_at=now(),
                           error="API stopped before recording the training result")
                self._save(job)
            self.jobs[job["id"]] = job

    def _save(self, job):
        path = self.storage / job["id"] / "job.json"
        temp = path.with_suffix(".tmp")
        temp.write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8")
        temp.replace(path)

    def submit(self, request, extra_env=None):
        with self.lock:
            if self.closing or self.active is not None:
                raise HTTPException(409, "Training is already running or the service is stopping")
            job_id = uuid4().hex
            directory = self.storage / job_id
            try:
                command = build_command(request, self.root, directory / "artifacts")
            except ValueError as exc:
                raise HTTPException(422, str(exc)) from exc
            directory.mkdir()
            job = dict(id=job_id, status="running", created_at=now(), finished_at=None,
                       return_code=None, error=None, parameters=request.model_dump(),
                       command=command, output_dir=str(directory / "artifacts"))
            self._save(job)
            env = os.environ.copy()
            if extra_env:
                env.update(extra_env)
            env.update(CUDA_VISIBLE_DEVICES=request.gpu, PYTHONHASHSEED=str(request.seed),
                       CUBLAS_WORKSPACE_CONFIG=":4096:8", PYTHONUNBUFFERED="1",
                       PYTHONIOENCODING="utf-8")
            options = {"start_new_session": True} if os.name != "nt" else {
                "creationflags": subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
            }
            try:
                with (directory / "console.log").open("wb") as log:
                    process = subprocess.Popen(command, cwd=str(self.root), env=env,
                                               stdout=log, stderr=subprocess.STDOUT,
                                               stdin=subprocess.DEVNULL, **options)
                if os.name == "nt":
                    from .windows_job import WindowsJob
                    try:
                        self.process_group = WindowsJob(process)
                    except OSError:
                        process.kill()
                        process.wait()
                        raise
            except OSError as exc:
                job.update(status="failed", error=str(exc), finished_at=now())
                self._save(job)
                self.jobs[job_id] = job
                return dict(job)
            self.jobs[job_id] = job
            self.process, self.active = process, job_id
            self.watcher = threading.Thread(target=self._watch, args=(job_id, process), daemon=True)
            self.watcher.start()
            return dict(job)

    def _watch(self, job_id, process):
        code = process.wait()
        with self.lock:
            if self.process_group is not None:
                self.process_group.close()
                self.process_group = None
            job = self.jobs[job_id]
            cancelled = job["status"] == "stopping"
            job.update(status="cancelled" if cancelled else ("succeeded" if code == 0 else "failed"),
                       return_code=code, finished_at=now())
            if code != 0 and not cancelled:
                job["error"] = "Training exited with code {}; see console.log".format(code)
            self._save(job)
            self.process, self.active = None, None

    def get(self, job_id):
        with self.lock:
            if job_id not in self.jobs:
                raise HTTPException(404, "Training job not found")
            return dict(self.jobs[job_id])

    def list(self):
        with self.lock:
            return [dict(job) for job in reversed(list(self.jobs.values()))]

    def cancel(self, job_id):
        with self.lock:
            self.get(job_id)
            if job_id != self.active or self.process is None:
                raise HTTPException(409, "Training job is no longer running")
            process = self.process
            self.jobs[job_id]["status"] = "stopping"
            self._save(self.jobs[job_id])
            # Kill the process group, including PyTorch data-loader children.
            if process.poll() is None:
                try:
                    if os.name == "nt":
                        self.process_group.close()
                        self.process_group = None
                    else:
                        os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass  # Process exited between poll() and killpg().
                except (OSError, subprocess.TimeoutExpired) as exc:
                    self.jobs[job_id]["status"] = "running"
                    self._save(self.jobs[job_id])
                    raise HTTPException(500, "Could not stop training: {}".format(exc)) from exc
            return self.get(job_id)

    def close(self):
        with self.lock:
            self.closing = True
            if self.active is not None:
                process = self.process
                self.cancel(self.active)
            else:
                process = None
        if process is not None:
            process.wait(timeout=20)
        if self.watcher is not None:
            self.watcher.join(timeout=20)


def create_app(root=VISION_ROOT, storage=None):
    root = Path(root).resolve()
    storage = Path(storage or os.environ.get("APT_API_JOB_DIR", str(root / "output" / "api_jobs")))

    @asynccontextmanager
    async def lifespan(app):
        from .jobs import ManagedJobs
        manager = JobManager(root, storage)
        app.state.manager = manager
        managed = ManagedJobs(manager)
        app.state.managed = managed
        try:
            yield
        finally:
            managed.close()

    app = FastAPI(title="Vision APT Training API", version="1.1.0", lifespan=lifespan, default_response_class=NewlineJsonResponse,)

    @app.get("/health")
    def health():
        return {"status": "ok", "service": "peft", "version": "1.1.0"}

    from .jobs import register_routes
    register_routes(app)

    @app.get("/training/options")
    def options():
        return {
            "datasets": sorted(p.stem for p in (root / "configs/datasets").glob("*.yaml")),
            "configs": sorted(p.stem for p in (root / "configs/trainers/APT").glob("*.yaml")),
        }

    @app.post("/training/apt", status_code=202)
    def start(request: APTRequest):
        return app.state.manager.submit(request)

    @app.get("/training/jobs")
    def jobs():
        return app.state.manager.list()

    @app.get("/training/jobs/{job_id}")
    def job(job_id: str):
        return app.state.manager.get(job_id)

    @app.get("/training/jobs/{job_id}/logs", response_class=PlainTextResponse)
    def logs(job_id: str, tail_bytes: int = Query(default=65536, ge=1, le=1048576)):
        manager = app.state.manager
        manager.get(job_id)
        path = manager.storage / job_id / "console.log"
        if not path.exists():
            return ""
        with path.open("rb") as stream:
            stream.seek(0, 2)
            stream.seek(max(0, stream.tell() - tail_bytes))
            return stream.read().decode("utf-8", errors="replace")

    @app.post("/training/jobs/{job_id}/cancel")
    def cancel(job_id: str):
        return app.state.manager.cancel(job_id)

    return app


app = create_app()
