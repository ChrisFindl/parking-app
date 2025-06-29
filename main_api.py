from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse, FileResponse
from detect_parking import main
import shutil
import os

# INTP/A/2025 Summer Group 1

app = FastAPI()

@app.post("/detect")
async def detect_parking(
    file: UploadFile,
    total_capacity: int = Form(...),
    threshold: float = Form(0.95),
    save_output: bool = Form(True)
):
    # Save uploaded file to a temporary location
    temp_path = f"res/{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Decide output path
    output_file = f"res/output_{file.filename}" if save_output else "detection_output.png"

    # Run detection
    result = main(
        temp_path,
        total_capacity,
        threshold=threshold,
        save_output=save_output,
        output_path=output_file
    )

    # Return image directly if saved
    if save_output and os.path.exists(output_file):
        return FileResponse(
            path=output_file,
            media_type="image/png",
            filename=os.path.basename(output_file)
        )

    # Otherwise just JSON summary
    return JSONResponse(result)
