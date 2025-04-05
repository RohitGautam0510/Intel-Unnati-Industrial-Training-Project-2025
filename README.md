# Retail Store Customer Tracking System

## About

A computer vision-based tracking system that uses YOLOv4-tiny and Deep SORT to monitor customer movement in a retail store, generate heatmaps, display real-time alerts, and export analytics per customer.

## Features
### 🧍‍♂️ Person Detection & Tracking
Using YOLOv4-tiny with Deep SORT for robust real-time tracking.

### 🗺️ Zone Detection & Overlays 
Tracks customer presence in zones: entrance, aisle, checkout, exit.

### ⚠️ Real-Time Alerts
Triggers on-screen alerts for prolonged dwell times (customizable).

### 📸 Heatmap Generation

Individual customer heatmaps: customer_heatmap_1.png, etc.

## Combined heatmap: 
![combined_heatmap](https://github.com/user-attachments/assets/9ae7f80d-afb8-448e-be09-b52cf50c382c)


## 📈 CSV Export

### Per-customer movement logs: 
movement_data_1.csv, etc.

### 🎥 Annotated Videos
Output videos with zone overlays, bounding boxes, and real-time alerts.


## 🚀 How to Run

### 1. Install Dependencies

pip install opencv-python numpy matplotlib seaborn deep_sort_realtime

### 2. Ensure Required Files Are Present

yolov4-tiny.weights, yolov4-tiny.cfg, and coco.names should be in the root folder.

### Videos should be stored in the video/ directory.

### 3. Run the Notebook
Open Enhancing Customer Experience with AI-Driven Insights .ipynb and run the tracking pipeline.

### The system will:

- Process all videos sequentially

* Create heatmaps and CSVs for each

+ Generate annotated videos with zone overlays and alerts

- Compile a summary combined_heatmap.png

## 📊 Outputs Explained

### File	Description
output_video_X.mp4 - Video with overlays and tracking \
customer_heatmap_X.png -	Heatmap of customer movement per video \
movement_data_X.csv -	Customer coordinates and zones per frame \
combined_heatmap.png -	Cumulative heatmap of all videos \

### 🧠 Built With
- YOLOv4-Tiny
* Deep SORT Realtime
+ OpenCV
- Matplotlib
* Seaborn

## 🔮 Future Scope

Future developments could include real-time dashboards, multi-camera support, and integration with inventory systems. Enhanced analytics like behavioral pattern recognition and product interaction tracking may provide deeper insights. The system can be extended to edge devices for on-site processing, while ensuring privacy through anonymized tracking and face blurring techniques.

## Developers of the Project

**Rohit Gautam** (Lead Developer) \
**Abrez Rizvi** (Developer) \
**Samdrub Phensah** (Developer) \
**Aashi Sachdeva** (Developer) 

 ## Affiliation 
 **CHRIST (Deemed To Be University) Delhi NCR**
