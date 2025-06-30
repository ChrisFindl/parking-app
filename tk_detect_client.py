import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox
from PIL import Image, ImageTk
import requests
import io

# INTP/A/2025 Summer Group 1

API_URL = "http://localhost:8000/detect"

def select_file():
    file_path = filedialog.askopenfilename(
        filetypes=[("Image files", "*.jpg *.jpeg *.png")]
    )
    if file_path:
        total_capacity = simpledialog.askinteger("Capacity", "Enter total capacity:")
        threshold = simpledialog.askfloat("Threshold", "Enter detection threshold (0-1):", initialvalue=0.95)
        
        if total_capacity is None or threshold is None:
            return
        
        send_to_api(file_path, total_capacity, threshold)

def send_to_api(file_path, total_capacity, threshold):
    try:
        with open(file_path, "rb") as f:
            files = {"file": f}
            data = {
                "total_capacity": total_capacity,
                "threshold": threshold,
                "save_output": True
            }
            response = requests.post(API_URL, files=files, data=data)

        if response.status_code == 200:
            # Save the received image
            output_path = "output_result.png"
            with open(output_path, "wb") as out_file:
                out_file.write(response.content)

            # Show in Tkinter
            show_image(output_path)
        else:
            messagebox.showerror("Error", f"API returned {response.status_code}")
    except Exception as e:
        messagebox.showerror("Error", str(e))

def show_image(image_path):
    img = Image.open(image_path)
    img.thumbnail((600, 400))
    img_tk = ImageTk.PhotoImage(img)

    panel.config(image=img_tk)
    panel.image = img_tk

# Create main window
root = tk.Tk()
root.title("Parking Lot Detector")

btn = tk.Button(root, text="Select Image & Run Detection", command=select_file)
btn.pack(pady=20)

panel = tk.Label(root)
panel.pack()

root.mainloop()
