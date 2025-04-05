Retail Store Customer Tracking System
A computer vision-based tracking system that uses YOLOv4-tiny and Deep SORT to monitor customer movement in a retail store, generate heatmaps, display real-time alerts, and export analytics per customer.
Features
🧍‍♂️ Person Detection & Tracking
Using YOLOv4-tiny with Deep SORT for robust real-time tracking.

🗺️ Zone Detection & Overlays
Tracks customer presence in zones: entrance, aisle, checkout, exit.

⚠️ Real-Time Alerts
Triggers on-screen alerts for prolonged dwell times (customizable).

📸 Heatmap Generation

Individual customer heatmaps: customer_heatmap_1.png, etc.

Combined heatmap: combined_heatmap.png

📈 CSV Export
Per-customer movement logs: movement_data_1.csv, etc.

🎥 Annotated Videos
Output videos with zone overlays, bounding boxes, and real-time alerts.

📁 Project Directory Overview
bash
Copy
Edit
.
├── video/                          # Folder containing input videos
├── yolov4-tiny.cfg                 # YOLOv4-tiny configuration
├── yolov4-tiny.weights             # YOLOv4-tiny weights
├── coco.names                      # COCO class names
├── output_video_1.mp4             # Annotated video with tracking
├── customer_heatmap_1.png         # Heatmap for video 1
├── movement_data_1.csv            # Tracking data for video 1
├── combined_heatmap.png           # Summary heatmap for all videos
├── Enhancing Customer Experience with AI-Driven Insights.ipynb                # Notebook containing main script
└── ...
🚀 How to Run
1. Install Dependencies
bash
Copy
Edit
pip install opencv-python numpy matplotlib seaborn deep_sort_realtime
2. Ensure Required Files Are Present
yolov4-tiny.weights, yolov4-tiny.cfg, and coco.names should be in the root folder.

Videos should be stored in the video/ directory.

3. Run the Notebook
Open Project_1.ipynb or customer_tracking.py (if available) and run the tracking pipeline.

The system will:

Process all videos sequentially

Create heatmaps and CSVs for each

Generate annotated videos with zone overlays and alerts

Compile a summary combined_heatmap.png

📊 Outputs Explained
File	Description
output_video_X.mp4	Video with overlays and tracking
customer_heatmap_X.png	Heatmap of customer movement per video
movement_data_X.csv	Customer coordinates and zones per frame
combined_heatmap.png	Cumulative heatmap of all videos
🧠 Built With
YOLOv4-Tiny

Deep SORT Realtime

OpenCV

Matplotlib

Seaborn
