# DermaVision

DermaVision is an early AI/ML research project exploring skin-image quality review and image classification. It is **not a medical device or diagnostic system**, and it has no clinical validation or established Indian-specific validation.

## Current status

- DermNet and HAM10000 remain separate datasets for separate experiments.
- The reviewed DermNet derived dataset has train/validation/test splits. Do not change these assignments without a documented audit.
- The existing DermNet CNN V2 experiment collapsed to one class on validation (about 5.50% accuracy). Its checkpoint is preserved as a failed experiment and is deliberately not loaded by the application.
- There is no suitable DermNet or HAM10000 model enabled for predictions. The API returns `503` for prediction and Grad-CAM requests until a model passes appropriate research validation.
- The app currently accepts an image for basic format, dimensions, brightness and possible blur checks. These heuristics are not medical image-quality standards. The application does not persist uploads; its multipart server framework may buffer a request temporarily while processing it.

## Project layout

```text
backend/app/       FastAPI endpoints
backend/tests/     Backend unit tests
frontend/          React + Vite research interface
dataset/           Original and derived research datasets (not tracked in Git)
models/            Model architecture and experiment artifacts
results/            Audits and experiment results (not tracked in Git)
src/               Dataset, preprocessing, loading and training scripts
```

## Environment

The current environment was verified with Python 3.13.14, TensorFlow 2.21.0, Node 24.19.0, and npm 11.17.0. TensorFlow reports CPU execution. Python dependency versions are recorded in `requirements.txt`; the API test client dependency is in `requirements-dev.txt`.

Create or activate the project virtual environment, then install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
```

Run the API from the project root:

```powershell
python -m uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000
```

Run the React development server in another terminal:

```powershell
cd frontend
npm install
npm run dev
```

The Vite development server proxies `/api` requests to `http://127.0.0.1:8000`. The API allows the local Vite origin by default; set `DERMAVISION_CORS_ORIGINS` to a comma-separated allowlist when using a different local origin.

## API overview

- `GET /api/health` — process health.
- `GET /api/model/status` — reports that prediction is disabled and explains why.
- `GET /api/categories/dermnet` — returns the DermNet taxonomy for educational UI use, not model predictions.
- `POST /api/image/quality` — accepts one JPEG, PNG, WebP, or BMP upload up to 10 MB; does not persist it.
- `POST /api/predict` — safely returns `503` while no suitable model is available.
- `POST /api/gradcam` — safely returns `503` while no suitable model is available.

## Verification

Run backend tests from the project root:

```powershell
python -m unittest backend.tests.test_api -v
```

Build the frontend:

```powershell
cd frontend
npm run build
```

## Dataset and research safeguards

- Never train on a test split or use it for model selection.
- Keep DermNet clinical photographs and HAM10000 dermoscopic images separate.
- Do not edit or delete source dataset images or labels.
- Do not commit dataset images, derived datasets, checkpoints, or result files. Check `.gitignore` before staging changes.
- Any future model is an experimental research artifact; report its cohort, split strategy, limitations, and metrics without implying clinical performance.
- A model trained on these datasets has no established Indian-specific validation. Skin tone, acquisition devices, prevalence, and clinical setting can affect performance.

## Prior diagnostics

The class-mapping and train/validation path audit passed for the reviewed DermNet split. The V2 model's validation predictions collapsed to one class, and the small balanced subset did not overfit in the tested configurations. A synthetic 23-class pattern sanity check did train to 100% on generated images; this only shows the model can optimize a simple artificial task. Diagnostic reports are under `results/dermnet_diagnostics/`. No test split was opened during those diagnostics. Further real-data model training should wait until the input/training behavior is better understood.
