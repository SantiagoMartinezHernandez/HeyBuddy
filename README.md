# MUI_bestGroup
MUI SP 26 digital neuro best group

#Backend Extension, Simon (23.04):
AI Backend (Gemini Coaching)
This project includes a lightweight AI backend that generates short coaching feedback based on aggregated movement metrics (e.g. squats).
Realtime pose estimation and motion analysis are handled separately; the Gemini model is used asynchronously after a set to generate natural language feedback.
Architecture Overview

Realtime pipeline: camera + pose estimation + metric aggregation
AI pipeline: aggregated metrics → Gemini → short coaching text
The AI backend is intentionally not used in realtime to ensure low latency and cost efficiency

Tech Stack

FastAPI – HTTP API
Gemini 2.0 Flash – fast, low‑latency text generation
python‑dotenv – environment variable management
google‑genai SDK – Gemini API client

flowchart TD
    A[Camera / Sensor] --> B[Pose Estimation]
    B --> C[Movement Metrics]
    C --> D[Aggregation\n(per set)]
    D --> E[FastAPI Backend\n/ai/feedback]
    E --> F[Gemini 2.0 Flash]
    F --> G[Coaching Feedback]
