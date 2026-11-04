"""Durable management protocol; use one Uvicorn worker and one API instance."""

import hashlib
import json
import logging
import re
import threading
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse
from urllib.request import url2pathname

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from .training import APTRequest, build_command, now

logger = logging.getLogger(__name__)


class NewlineJsonResponse(JSONResponse):
    """响应体末尾补两个换行，沿用本服务原有格式（curl 直接看更清楚）。

    原来是从 .main 导入的，但 main 在模块级 create_app() 里又导入本模块的
    register_routes，形成循环导入——正常启动能过，直接 import api.jobs 就炸。
    这里本地定义一份，切断对 main 的依赖。
    """

    def render(self, content) -> bytes:
        return super().render(content) + b"\n\n"


class JobParams(APTRequest):
    model_uri: Optional[str] = None  # Base CLIP model, not an APT prompt checkpoint.
    robust_backbone_dir: Optional[str] = None


class SubmitJob(BaseModel):
    model_config = ConfigDict(extra="forbid")
    job_id: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    other_params: JobParams = Field(default_factory=JobParams)


class AcceptedJob(BaseModel):
    job_id: str
    status: str
    error: Optional[str] = None


class JobStatus(AcceptedJob):
    progress: Optional[int] = None
    logs: list[str] = Field(default_factory=list)
    output_uri: Optional[str] = None
    result: Optional[dict] = None


def failure(job_id, error, code=500):
    return NewlineJsonResponse(status_code=code, content={
        "job_id": job_id, "status": "failed", "error": error,
    })


class JobRejected(Exception):
    """提交被拒：参数/环境问题 400、job_id 冲突 409、服务不可用 503。

    以前一律返回 500，调用方无法区分"我传错了"和"服务挂了"，只能当服务异常重试。
    """

    def __init__(self, message, code=400):
        super().__init__(message)
        self.code = code


def local_path(uri, root, directory=False):
    parsed = urlparse(uri)
    if parsed.scheme == "file":
        if parsed.netloc not in ("", "localhost"):
            raise ValueError("文件地址只能指向算法服务器本机")
        path = Path(url2pathname(parsed.path))
    elif parsed.scheme and not re.match(r"^[A-Za-z]:[\\/]", uri):
        raise ValueError("暂不支持 {} 地址，请先下载到算法服务器本地".format(parsed.scheme))
    else:
        path = Path(uri).expanduser()
    if not path.is_absolute():
        path = root / path
    if (directory and not path.is_dir()) or (not directory and not path.is_file()):
        raise ValueError("{}不存在：{}".format("目录" if directory else "模型文件", path))
    return str(path.resolve())


class ManagedJobs:
    def __init__(self, engine):
        self.engine = engine
        self.storage = engine.storage / "managed"
        self.storage.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.stop = threading.Event()
        self.records = {}
        for path in self.storage.glob("*.json"):
            record = json.loads(path.read_text(encoding="utf-8"))
            if record["status"] == "running":
                record.update(status="failed", finished_at=now(),
                              error="算法服务上次中断，训练未完成，请使用新任务 ID 重新提交")
                self._save(record)
            self.records[record["job_id"]] = record
        self.worker = threading.Thread(target=self._work, daemon=True, name="apt-jobs")
        self.worker.start()

    def _save(self, record):
        name = hashlib.sha256(record["job_id"].encode("utf-8")).hexdigest()
        path = self.storage / (name + ".json")
        temp = path.with_suffix(".tmp")
        temp.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        temp.replace(path)

    def _reject(self, job_id, parameters, message):
        """被拒的任务也留档，再抛 JobRejected 由路由转成 4xx。

        以前提交失败只在响应体里给一句话，事后按 job_id 查不到任何记录。
        """
        record = dict(job_id=job_id, parameters=parameters, env={}, status="failed", error=message,
                      created_at=now(), finished_at=now(), engine_id=None, result=None, output_uri=None)
        self._save(record)
        self.records[job_id] = record
        raise JobRejected(message, 400)

    def enqueue(self, request):
        params = request.other_params
        training = APTRequest.model_validate(params.model_dump(exclude={"model_uri", "robust_backbone_dir"}))
        normalized = training.model_dump()
        normalized.update(model_uri=params.model_uri, robust_backbone_dir=params.robust_backbone_dir)
        with self.lock:
            old = self.records.get(request.job_id)
            if old is not None:
                if old["parameters"] != normalized:
                    raise JobRejected("任务 ID 已存在，但参数不同；请使用新的 job_id", 409)
                return {"job_id": request.job_id, "status": old["status"], "error": old["error"]}
            if self.stop.is_set():
                raise JobRejected("算法服务正在关闭，暂不接受新任务", 503)
            if not (self.engine.root / "train_api.py").is_file():
                raise JobRejected("训练入口 train_api.py 不存在，请检查部署目录", 500)
            try:
                build_command(training, self.engine.root, self.engine.storage / "validation-only")
            except ValueError as exc:
                self._reject(request.job_id, normalized, "训练配置或数据目录错误：{}".format(exc))
            env = {}
            try:
                if params.model_uri is not None:
                    env["CLIP_MODEL_PATH"] = local_path(params.model_uri, self.engine.root)
                if params.robust_backbone_dir is not None:
                    env["APT_ROBUST_BACKBONE_DIR"] = local_path(
                        params.robust_backbone_dir, self.engine.root, directory=True
                    )
            except ValueError as exc:
                self._reject(request.job_id, normalized, "模型权重路径不可用：{}".format(exc))
            record = dict(job_id=request.job_id, parameters=normalized, env=env,
                          status="queued", error=None, created_at=now(), finished_at=None,
                          engine_id=None, result=None, output_uri=None)
            self._save(record)  # Persist before reporting success.
            self.records[request.job_id] = record
            return {"job_id": request.job_id, "status": "queued", "error": None}

    def _logs(self, engine_id):
        if not engine_id:
            return []
        path = self.engine.storage / engine_id / "console.log"
        if not path.exists():
            return []
        with path.open("rb") as stream:
            stream.seek(0, 2)
            stream.seek(max(0, stream.tell() - 65536))
            # 不截断行数：管理端靠"每次拿到全量、只补新增"来合并日志，
            # 只给最后 100 行会让它在中后期停止追加、前半段直接丢失。
            return stream.read().decode("utf-8", errors="replace").splitlines()


    @staticmethod
    def _progress(status, logs):
        """把两个阶段拼成一条单调曲线：训练 0-80%、评估 80-99%、成功 100。

        以前训练按 epoch 直接算到 99、评估阶段又按 tqdm 从 9% 重来，
        管理端进度只增不减，就永远卡在 99%。这里按"最后一根进度条是否出现在
        最后一个 epoch 之后"判断当前阶段，再映射到各自的区段。
        """
        if status == "succeeded":
            return 100
        if status in ("failed", "cancelled", "interrupted"):
            return None

        last_epoch = last_bar = -1
        epoch_done = epoch_total = 0
        bar_percent = 0
        for index, line in enumerate(logs):
            matched = re.search(r"epoch \[(\d+)/(\d+)\]", line)
            if matched:
                last_epoch = index
                epoch_done, epoch_total = int(matched.group(1)), max(1, int(matched.group(2)))
                continue
            matched = re.search(r"(\d{1,3})%\|", line)
            if matched:
                last_bar = index
                bar_percent = int(matched.group(1))

        if last_bar > last_epoch:  # 评估阶段
            return min(99, 80 + bar_percent * 19 // 100)
        if last_epoch >= 0:  # 训练阶段
            return min(80, epoch_done * 80 // epoch_total)
        return None


    def get(self, job_id):
        with self.lock:
            if job_id not in self.records:
                raise KeyError(job_id)

            record = dict(self.records[job_id])

        logs = self._logs(record["engine_id"])

        # 任务失败时，如果 error 为空但日志里有 ValueError，就把它提到 error 上；
        # logs 仍返回全量——以前这里把 logs 换成只剩一条，排查时什么都看不到。
        if record["status"] == "failed" and not record["error"]:
            value_error = next(
                (
                    line.strip()
                    for line in reversed(logs)
                    if line.strip().startswith("ValueError:")
                ),
                None,
            )
            if value_error is not None:
                record["error"] = value_error

        return {
            "job_id": job_id,
            "status": record["status"],
            "progress": self._progress(record["status"], logs),
            "logs": logs,
            "error": record["error"],
            "output_uri": record["output_uri"],
            "result": record["result"],
        }

    @staticmethod
    def _read_metrics(output):
        """读产物里的 results.json 换算成管理端的指标行。

        该文件是 JSON Lines，由 trainers 在评估结束时写出：
          第 1 行 = 干净测试集（accuracy / error_rate / macro_f1）
          第 2 行 = 对抗测试集（开了 TRAINER.COOP.ATK_TEST 才有）
        只读不写，读不到就返回空——绝不影响训练结果本身。

        管理端指标表是 {name, baseline, value, threshold, passed}：算法侧只有实测值，
        基线与阈值不编造，passed 留 None，结论会落"待验证"。
        """
        path = Path(output) / "results.json"
        if not path.is_file():
            return []
        rows = []
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue
        if not rows:
            return []

        stages = [("干净测试集", rows[0]), ("对抗测试集", rows[1] if len(rows) > 1 else None)]
        metrics = []
        for suffix, row in stages:
            if not isinstance(row, dict):
                continue
            for key, name in (("accuracy", "Top-1 准确率"), ("macro_f1", "Macro-F1")):
                value = row.get(key)
                if value is None:
                    continue
                try:
                    text = "{:.2f}%".format(float(value))
                except (TypeError, ValueError):
                    text = str(value)
                metrics.append({
                    "name": "{}（{}）".format(name, suffix),
                    "baseline": "—",
                    "value": text,
                    "threshold": "—",
                    "passed": None,
                })
        return metrics

    def _finish(self, record, job):
        if job["status"] == "succeeded":
            output = Path(job["output_dir"])
            if not output.is_dir():
                raise ValueError("训练进程已退出，但未找到产物目录：{}".format(output))
            artifacts, total = [], 0
            for path in sorted(output.rglob("*")):
                if path.is_file() and not path.is_symlink():
                    size = path.stat().st_size
                    artifacts.append({"name": path.relative_to(output).as_posix(), "size_bytes": size})
                    total += size
            trainer = "SPT" if record["parameters"].get("robust") else "APT"
            record.update(status="succeeded", output_uri=str(output.resolve()), error=None,
                          result={"algorithm": trainer, "dataset": record["parameters"]["dataset"],
                                  "seed": record["parameters"]["seed"], "size_bytes": total,
                                  "artifacts": artifacts,
                                  "metrics": self._read_metrics(output)})
        else:
            logs = "\n".join(self._logs(job["id"]))
            error = "训练执行失败，退出码为 {}；请查看 logs".format(job.get("return_code"))
            if "ModuleNotFoundError" in logs or "ImportError" in logs:
                error = "训练环境缺少依赖或模块导入失败，请查看 logs 中的具体包名"
            elif "out of memory" in logs.lower():
                error = "GPU 显存不足，请减少批大小或释放显存后重试"
            elif "FileNotFoundError" in logs:
                error = "训练所需的数据或模型文件不存在，请查看 logs 中的文件路径"
            elif job["status"] in ("cancelled", "interrupted"):
                error = "训练因服务关闭或任务停止而中断，请使用新任务 ID 重新提交"
            elif job.get("error"):
                error = "训练启动或执行失败：{}".format(job["error"])
            elif "ValueError:" in logs:
                detail = next(
                    (
                        line.strip()
                        for line in reversed(logs.splitlines())
                        if "ValueError:" in line
                    ),
                    "ValueError: 未知原因",
                )
                error = f"错误：{detail}"
            record.update(status="failed", error=error)
        record["finished_at"] = now()
        self._save(record)

    def _work(self):
        while not self.stop.wait(0.1):
            record = None
            try:
                with self.lock:
                    if self.stop.is_set():
                        return
                    pending = sorted((r for r in self.records.values() if r["status"] == "queued"),
                                     key=lambda r: r["created_at"])
                    if not pending:
                        continue
                    with self.engine.lock:
                        if self.engine.active is not None:
                            continue
                        record = pending[0]
                        training = APTRequest.model_validate({
                            k: v for k, v in record["parameters"].items()
                            if k not in ("model_uri", "robust_backbone_dir")
                        })
                        job = self.engine.submit(training, extra_env=record["env"])
                        watcher = self.engine.watcher
                        record.update(engine_id=job["id"], status="running")
                        self._save(record)
                while job["status"] in ("running", "stopping"):
                    if watcher is not None:
                        watcher.join(timeout=0.1)
                    job = self.engine.get(job["id"])
                with self.lock:
                    self._finish(record, job)
            except Exception as exc:
                logger.exception("管理端训练任务执行失败")
                if record is not None:
                    with self.lock:
                        record.update(status="failed", error="任务处理失败：{}".format(exc), finished_at=now())
                        self._save(record)

    def close(self):
        with self.lock:
            self.stop.set()
        self.engine.close()
        self.worker.join(timeout=25)


def register_routes(app):

    @app.get("/jobs")
    def list_jobs():
        manager = app.state.managed

        with manager.lock:
            jobs = [
                {
                    "job_id": record["job_id"],
                    "status": record["status"],
                }
                for record in manager.records.values()
            ]

        return {
            "total": len(jobs),
            "jobs": jobs,
        }


    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc):
        if request.url.path == "/jobs" and request.method == "POST":
            body = exc.body if isinstance(exc.body, dict) else {}
            fields = [".".join(str(part) for part in e["loc"] if part != "body") for e in exc.errors()]
            job_id = body.get("job_id") if isinstance(body.get("job_id"), str) else None
            return failure(job_id, "参数错误，请检查字段及其类型或取值范围：" + "、".join(fields), code=422)
        from fastapi.exception_handlers import request_validation_exception_handler
        return await request_validation_exception_handler(request, exc)

    @app.post("/jobs", response_model=AcceptedJob)
    def submit_job(body: SubmitJob):
        try:
            return app.state.managed.enqueue(body)
        except JobRejected as exc:
            return failure(body.job_id, str(exc), code=exc.code)
        except (ValueError, OSError) as exc:
            return failure(body.job_id, "任务提交失败：{}".format(exc), code=400)
        except Exception:
            logger.exception("管理端任务提交失败")
            return failure(body.job_id, "任务提交失败，算法服务内部异常，请检查服务日志")

    @app.get("/jobs/{job_id}", response_model=JobStatus)
    def get_job(job_id: str):
        try:
            return app.state.managed.get(job_id)
        except KeyError:
            return failure(job_id, "任务不存在，请检查 job_id", code=404)
        except Exception:
            logger.exception("管理端任务查询失败")
            return failure(job_id, "任务查询失败，请检查算法服务日志")
