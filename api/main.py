"""
Main API module defining the FastAPI application and its endpoints.
"""

import csv
import resource
import time
from io import StringIO

import ollama
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from api.models.comment import Comment, CommentList
from api.nlp import (
    danger_analyzer,
    hate_analyzer,
    sentiment_analyzer,
    categories_analyzer,
)
from api.settings import settings

limiter = Limiter(key_func=get_remote_address)

app = FastAPI()

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def start_metrics() -> dict:
    return {
        "wall_time": time.perf_counter(),
        "cpu_time": time.process_time(),
        "ram_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }


def build_metrics(metrics_start: dict) -> dict:
    wall_elapsed = time.perf_counter() - metrics_start["wall_time"]
    cpu_elapsed = time.process_time() - metrics_start["cpu_time"]
    ram_delta_kb = max(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - metrics_start["ram_kb"],
        0,
    )

    return {
        "time_seconds": round(wall_elapsed, 6),
        "cpu_seconds": round(cpu_elapsed, 6),
        "ram_delta_mb": round(ram_delta_kb / 1024, 6),
    }


def map_danger_label(label: str) -> str:
    """
    Maps the danger label to a descriptive value based on the model used.
    """

    mapping = {"LABEL_0": "BAJO", "LABEL_1": "MEDIO", "LABEL_2": "ALTO"}

    return mapping.get(label)


@app.get("/")
@limiter.limit("60/minute")
async def root(request: Request):
    """
    Root endpoint returning a simple greeting message.
    """
    return {"message": "Hello World"}


@app.post("/comments/")
@limiter.limit("30/minute")
async def analyze_comment(
    request: Request,
    comment: Comment,
):
    """
    Endpoint to analyse and store a comment.
    """

    metrics_start = start_metrics()

    sentiment = sentiment_analyzer.predict(comment.content)
    hate = hate_analyzer.predict(comment.content)

    danger_label = danger_analyzer.predict(comment.content)[0]
    danger = map_danger_label(danger_label["label"])

    categories = categories_analyzer.predict(comment.content)

    return {
        "comment": comment.content,
        "sentiment": sentiment,
        "hate": hate,
        "danger": {"label": danger_label, "description": danger},
        "categories": categories,
        "metrics": build_metrics(metrics_start),
        "status": "Comment created successfully",
    }


@app.post("/summarize/")
@limiter.limit("5/minute")
async def summarize_comments(request: Request, comment_list: CommentList):
    """
    Endpoint to generate an executive summary of comments using Gemma3 via Ollama.

    Receives a list of comments and returns an institutional summary analyzing
    the overall perception, areas for improvement, and any alert comments
    requiring immediate attention.
    """

    if not comment_list.comments:
        raise HTTPException(status_code=400, detail="No comments provided")

    comments_text = ", ".join(comment_list.comments)
    prompt = settings.summarize_prompt.format(comments_text)

    try:
        response = ollama.chat(
            model="gemma3:1b",
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )

        summary = response["message"]["content"]

        return {
            "summary": summary,
            "total_comments": len(comment_list.comments),
            "status": "Summary generated successfully",
        }

    except ollama.ResponseError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Ollama service error: {str(e)}. Make sure Ollama is running and gemma3 model is available.",
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error generating summary: {str(e)}"
        )
