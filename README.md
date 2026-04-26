# Crop Yield Planner

Crop Yield Planner is a Streamlit web app that helps users explore crop recommendations using district, rainfall, temperature, and area inputs.

The app now supports two recommendation paths:

- If the trained model files are available, it uses the saved ML model for prediction.
- If the model files are missing, it falls back to historical district-based recommendations from `final_crop.csv`.

## Main Features

- User-friendly web interface with guided inputs
- District-level defaults to reduce manual entry
- Top crop recommendations with suitability scores
- District history and comparison views
- Safe fallback mode when `.pkl` model files are not present

## Project Files

- `app.py` - main Streamlit application
- `final_crop.csv` - dataset used for fallback recommendations and summaries
- `requirements.txt` - Python dependencies
- `start_crop_app.bat` - one-click Windows setup and launcher

Optional model files:

- `crop_model.pkl`
- `scaler.pkl`
- `crop_encoder.pkl`
- `dist_encoder.pkl`

If these optional files are not in the project folder, the app still works in fallback mode.

## How To Run

### Option 1: One-click launcher

On Windows, double-click:

`start_crop_app.bat`

What it does:

1. Checks that Python is installed
2. Creates a local virtual environment in `.venv` if needed
3. Installs the required packages from `requirements.txt`
4. Starts the Streamlit web app
5. Opens the app in your default browser

The app usually opens at:

`http://localhost:8501`

### Option 2: Manual command line run

From the project folder:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m streamlit run app.py
```

### Option 3: Run with Docker

This is a good option if you want the app bundled into one portable image.

Build the image:

```powershell
docker build -t crop-yield-planner .
```

Run the container:

```powershell
docker run --rm -p 8501:8501 crop-yield-planner
```

Then open:

`http://localhost:8501`

Notes for Docker:

- Docker Desktop must be installed on the PC.
- The image includes Python and all Python package requirements.
- If you later add the optional model files (`crop_model.pkl`, `scaler.pkl`, `crop_encoder.pkl`, `dist_encoder.pkl`) into the project before building, they will also be packaged into the image.
- Streamlit is exposed on port `8501`.

## Requirements

- Windows
- Python 3.10 or newer recommended
- Internet access for first-time package installation

## Notes

- The unit for area follows the dataset field: `Area(1000 ha)`.
- If the browser does not open automatically, go to `http://localhost:8501`.
- If port `8501` is already in use, Streamlit may start on another local port.
- If double-click launch says Python was not found and opens the Microsoft Store, disable the `python.exe` alias in `Settings > Apps > Advanced app settings > App execution aliases`, or install Python from `python.org`.
- The launcher keeps Streamlit runtime files inside the project folder so it does not depend on write access to your Windows user profile.
