import logging
import os
from pathlib import Path
import pickle
from contextlib import asynccontextmanager
import time
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
import uvicorn

from inference.infer import FlatClassifier

logger = logging.getLogger("uvicorn")

cwd = Path(__file__).resolve().parent
target_dir = cwd / "target"

context = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    subheadings_file = target_dir / "subheadings.pkl"
    if not subheadings_file.exists():
        raise FileNotFoundError(
            f"❌ Could not find subheadings file: {subheadings_file}"
        )

    logger.info(f"💾⇨ Loading subheadings pickle file: {subheadings_file}")
    with open(subheadings_file, "rb") as fp:
        subheadings = pickle.load(fp)
    logger.info("Subheadings loaded")

    model_file = target_dir / "model.pt"
    if not model_file.exists():
        raise FileNotFoundError(f"❌ Could not find model file: {model_file}")

    context["classifier"] = FlatClassifier(model_file, subheadings)

    yield


app = FastAPI(lifespan=lifespan)


@app.middleware("http")
async def check_api_key(request: Request, call_next):
    if "API_KEY" in os.environ and not os.environ["API_KEY"] == "":
        if (
            not request.headers.get("X-Api-Key")
            or request.headers.get("X-Api-Key") != os.environ["API_KEY"]
        ):
            logger.info("Attempted access with missing or invalid API key")
            return JSONResponse(
                status_code=403,
                content={"detail": "You need an API key to access this resource"},
            )

    return await call_next(request)


@app.get("/search")  # type: ignore
async def search(q: str, digits: int = 6, limit: int = 5):
    start_time = time.time()

    if digits not in [6, 8]:
        raise HTTPException(400, "digits must be 6 or 8")

    if limit < 1 or limit > 10:
        raise HTTPException(400, "limit must be between 1 and 10")

    results = context["classifier"].classify(q, limit, digits)

    response_time = (time.time() - start_time) * 1000

    logger.info("Finished search request in %.2fms" % (response_time))

    return [{"code": result.code, "score": result.score * 1000} for result in results]


if __name__ == "__main__":
    uvicorn.run(app, port=5000)
