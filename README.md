
# Parking Lot Detection App

This is a complete end-to-end project that demonstrates using AI (DETR object detection) to analyze parking lot images, count parked vehicles, estimate free slots, and serve results via an API and a simple GUI.

---

## 🚀 Features

- 🔍 Detects cars, trucks, buses, motorcycles using a pre-trained DETR model (PyTorch + torchvision).
- 🚗 Estimates free spots based on user-defined total capacity.
- 📊 Visual output with bounding boxes and clustering into rows.
- ⚡ FastAPI server to receive images and return processed outputs (as JSON or directly as annotated images).
- 🖼 Tkinter GUI client to upload images, call the API, and view results.
- Supports parameterized thresholds, dynamic capacity, and flexible output paths.

---

## ⚙️ Installation

### 🐍 Clone the repository
```
git clone <your-repo-url>
cd parking-app
```

### 🐍 Create a virtual environment
```
python3 -m venv .venv
source .venv/bin/activate  # on macOS/Linux
.venv\Scripts\activate   # on Windows
```

### 🐍 Install all dependencies
```
pip install torch torchvision pillow matplotlib scikit-learn fastapi uvicorn requests
```

> ✅ Note: On Ubuntu, also ensure you have tkinter support installed:
```
sudo apt update
sudo apt install python3-tk
```

---

## 🚀 Usage

### ⚡ Run the FastAPI server
```
uvicorn main_api:app --host 0.0.0.0 --port 8000
```
- Visit `http://localhost:8000/docs` for the interactive Swagger UI.
- You can upload images, set `total_capacity`, `threshold`, and receive either JSON or an annotated PNG directly.

---

### 🖼 Run the Tkinter GUI client
In another terminal (with your venv activated):

```
python tk_detect_client.py
```
- Select an image file.
- Enter total capacity (e.g., `21`).
- Enter threshold (e.g., `0.5` for more sensitive detection).
- It will display the processed image right in the window.

---

## 🔍 Example run commands
```
# Activate environment
source .venv/bin/activate

# Start API server
uvicorn main_api:app --host 0.0.0.0 --port 8000

# In another terminal: run GUI
python tk_detect_client.py
```

---

## 🚀 Testing with curl or Swagger UI
```
curl -X POST "http://localhost:8000/detect"   -F "file=@res/parking1.jpg"   -F "total_capacity=21"   -F "threshold=0.5"   -F "save_output=true" --output output.png
```

Or simply visit:
```
http://localhost:8000/docs
```
and use the built-in form to POST images.

---

## 📂 Project structure
```
parking-app/
├── detect_parking/      # detection module
│   └── __init__.py
├── res/                 # sample images & outputs
│   ├── parking1.jpg
│   └── output_*.png
├── run_detection.py     # simple local test
├── main_api.py          # FastAPI server
├── tk_detect_client.py  # Tkinter GUI app
├── .venv/               # your virtual environment
└── requirements.txt     # optional, can be generated with `pip freeze`
```

---

## 📝 Generating requirements.txt
Once everything is working, you can freeze your exact packages:

```
pip freeze > requirements.txt
```

Then recreate on another machine with:

```
pip install -r requirements.txt
```

---

## ✅ Done!
You now have a full pipeline:
```
[ GUI / curl / Swagger ] -> [ FastAPI ] -> [ DETR detection ] -> [ Annotated output ]
```
Enjoy building on it!
