"""Run from the vision directory: python -m api

Defaults to 0.0.0.0:7001 with a single worker. Override with APT_API_HOST / APT_API_PORT.
State lives in ./output/api_jobs unless APT_API_JOB_DIR points elsewhere.
"""

import os

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host=os.environ.get("APT_API_HOST", "0.0.0.0"),
        port=int(os.environ.get("APT_API_PORT", 7001)),
        workers=1,  # Job state is held in memory; never raise this.
        log_level=os.environ.get("APT_API_LOG_LEVEL", "info"),
    )
