# DermaVision

DermaVision is a planned AI-powered skin disease image classification and early screening application. It is intended to explore how image processing and machine learning can support preliminary screening; it is not a substitute for diagnosis or advice from a qualified healthcare professional.

## Planned technology

- Python with TensorFlow/Keras for a convolutional neural network (CNN)
- OpenCV, NumPy, and Pandas for image and data processing
- FastAPI for a backend API
- React for the web interface

## Project structure

The repository currently contains these empty working directories:

```text
backend/    API and server-side application
dataset/    Dataset files (handle images and labels responsibly)
frontend/   React application
models/     Model artifacts created during later development
notebooks/  Exploratory analysis and learning notebooks
results/    Evaluation outputs and reports
src/        Reusable project code
```

The structure is an initial scaffold. No model or application code has been created yet.

## Development status

This project is at the setup stage. The root `requirements.txt` is currently empty, and the existing `venv` cannot start because its configured base Python executable is inaccessible. Set up a working Python installation and environment before adding project dependencies.

## Responsible use

Skin image classification is a screening aid only. Model outputs can be wrong and should not be used alone to make healthcare decisions. Use properly licensed data, protect image privacy, and seek professional medical advice for health concerns.
