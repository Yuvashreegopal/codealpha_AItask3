import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import cv2
from ultralytics import YOLO
import threading

# Load YOLOv8 model
model = YOLO("yolov8n.pt")  # You can use 'yolov8s.pt' for more accuracy

# GUI setup
root = tk.Tk()
root.title("Object Detection and Tracking - YOLOv8")
root.geometry("900x700")

# Video panel
video_panel = tk.Label(root)
video_panel.pack(pady=10)

cap = None
running = False

def start_detection():
    global cap, running
    if running:
        return
    running = True
    cap = cv2.VideoCapture(0)
    threading.Thread(target=detect_objects).start()

def stop_detection():
    global cap, running
    running = False
    if cap:
        cap.release()
    video_panel.config(image='')

def detect_objects():
    global running
    while running:
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLO detection
        results = model.track(source=frame, persist=True, show=False)
        annotated_frame = results[0].plot()

        # Convert OpenCV to ImageTk
        cv2image = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img)

        video_panel.imgtk = imgtk
        video_panel.config(image=imgtk)

    cap.release()

# Buttons
btn_frame = tk.Frame(root)
btn_frame.pack(pady=10)

start_btn = tk.Button(btn_frame, text="Start Detection", command=start_detection, font=("Arial", 12), bg="green", fg="white")
start_btn.grid(row=0, column=0, padx=10)

stop_btn = tk.Button(btn_frame, text="Stop", command=stop_detection, font=("Arial", 12), bg="red", fg="white")
stop_btn.grid(row=0, column=1, padx=10)

# Close properly
def on_closing():
    stop_detection()
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_closing)
root.mainloop()
