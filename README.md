# License Plate Detection

This project focuses on detecting vehicle license plates using a YOLOv10-Small (YOLOv10s) object detection model.
The model is trained on a high-quality license plate dataset imported from Roboflow and shows strong detection performance.

The repository includes:

📒 Google Colab Notebook for training & inference

🧠 Baseline trained model (Plate_Baseline.pt)

📊 Performance report and evaluation metrics

📁 Dataset link and details

# Dataset
Source: Roboflow

Annotations: YOLO Format

Automatic Train/Val/Test splitting via Roboflow API

Variety: Plates captured under multiple angles & lighting environments

Augmentation applied to improve robustness:

Rotation, flip, scale

Brightness/contrast

Blur & noise

Random cropping

Dataset Link:
https://app.roboflow.com/smarthsrp/license-plate-recognition-rxg4e-xyod6/2


# How to Use the Model

## Install Dependencies

```bash
pip install ultralytics
```
## Run Inference

```python
from ultralytics import YOLO

model = YOLO("Plate_Baseline.pt")
results=model.predict(source="path/to/image.jpg", save=True, conf=0.5)
results[0].show()

```
# Model Performance — Test Set
| Metric            | Value     | Meaning                                                            |
| ----------------- | --------- | ------------------------------------------------------------------ |
| **Precision (P)** | **0.99**  | Very few false positives — almost all detected plates are correct. |
| **Recall (R)**    | **0.946** | Most plates present in images were successfully detected.          |
| **mAP50**         | **0.979** | Excellent accuracy at IoU 0.5 — strong model confidence.           |
| **mAP50-95**      | **0.728** | Good performance under stricter IoU thresholds.                    |

# 📸 Sample Detection Results

Here are some examples of the model detecting license plates:

<p align="center">
  <img src="Result/single_inference_result.jpg" width="500">
  <br>
  <em>Single Image Inference</em>
</p>

<p align="center">
  <img src="Result/batch1_labels_result.jpg" width="500">
  <br>
  <em>Model performance on multiple images in a batch.</em>
</p>

<p align="center">
  <img src="Result\results.png" width="500">
  <br>
  <em>Training Metrics (Precision, Recall, mAP, Loss Curves)</em>
</p>

# Contributing

Contributions, issues, and feature requests are welcome!
Feel free to fork the repository and submit a PR 🚀  

# Acknowledgements

The license plate detection model in this project was trained using the public License Plate Recognition Dataset from Roboflow Universe.
Dataset URL:
https://universe.roboflow.com/roboflow-universe-projects/license-plate-recognition-rxg4e

## License
This project is licensed under the MIT License.  
See the [LICENSE](./LICENSE) file for details.


