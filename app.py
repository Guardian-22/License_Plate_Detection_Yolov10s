import gradio as gr
from ultralytics import YOLO
import cv2

# Load the model
model = YOLO("Plate_Baseline.pt")

def predict(image):
    # Run YOLO prediction
    results = model(image)

    # YOLO returns annotated image using .plot()
    annotated_frame = results[0].plot()
    return annotated_frame

# Build Gradio interface
interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="numpy"),
    outputs=gr.Image(type="numpy"),
    title="Number Plate Detection (YOLO10s)",
    description="Upload an image to detect number plates."
)

if __name__ == "__main__":
    interface.launch()
