# Import necessary libraries
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from collections import defaultdict
import time
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Step 1: Load Kaggle Dataset (Customer Behavior Data)
# Replace 'customer_behavior.csv' with the actual Kaggle dataset file path.
data = pd.read_csv("Mall_Customers.csv")

# Step 2: Data Exploration and Preprocessing
print("Dataset Overview:")
print(data.head())
print(data.info())

# Handle missing values by forward filling
data.fillna(method='ffill', inplace=True)

# Normalize numerical features for clustering and ML models
scaler = StandardScaler()
numerical_columns = data.select_dtypes(include=['float64', 'int64']).columns
data[numerical_columns] = scaler.fit_transform(data[numerical_columns])

# Step 3: Customer Segmentation Using K-Means Clustering
print("\nPerforming Customer Segmentation...")
kmeans = KMeans(n_clusters=5, random_state=42)  # Define number of clusters (adjustable)
data['Cluster'] = kmeans.fit_predict(data[numerical_columns])
silhouette_avg = silhouette_score(data[numerical_columns], data['Cluster'])
print(f"Silhouette Score for Clustering: {silhouette_avg}")

# Visualize cluster distribution (optional)
print("Cluster Distribution:")
print(data['Cluster'].value_counts())

# Step 4: Prepare data for predictive analytics
# Convert 'Genre' to numeric
data['Genre'] = data['Genre'].map({'Female': 0, 'Male': 1})

# Prepare features and target
X = data[['Age', 'Annual Income (k$)', 'Genre']]  # Using these as features
y = data['Spending Score (1-100)']  # Predicting spending score

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build and train the model
model = Sequential([
    Dense(64, activation='relu', input_shape=(3,)),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
print("\nTraining the predictive model...")
model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, verbose=0)

# Evaluate the model
test_loss, test_mae = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"\nModel Performance:")
print(f"Mean Absolute Error: {test_mae:.2f}")

# Make some predictions
print("\nSample Predictions:")
sample_customers = X_test_scaled[:3]
predictions = model.predict(sample_customers, verbose=0)
for i, pred in enumerate(predictions):
    print(f"Customer {i+1} predicted spending score: {pred[0]:.2f}")

class CustomerTracker:
    def __init__(self, store_layout_zones=None):
        self.tracks = defaultdict(list)
        self.dwell_times = defaultdict(float)
        self.heatmap_data = np.zeros((480, 640))  # Adjust size based on your video
        self.store_zones = store_layout_zones or {
            'entrance': [(0, 0, 200, 480)],
            'checkout': [(440, 0, 640, 480)],
            'center': [(200, 0, 440, 480)]
        }
        self.zone_visits = defaultdict(int)
        self.last_positions = {}
        self.start_times = {}
    
    def update_tracking(self, person_id, x, y, w, h):
        center_x, center_y = x + w//2, y + h//2
        self.tracks[person_id].append((center_x, center_y))
        
        # Update heatmap
        self.heatmap_data[max(0, y):min(480, y+h), max(0, x):min(640, x+w)] += 1
        
        # Update zone analytics
        current_zone = self.get_current_zone(center_x, center_y)
        if person_id not in self.last_positions:
            self.start_times[person_id] = time.time()
        elif self.last_positions[person_id] != current_zone:
            # Zone transition
            dwell_time = time.time() - self.start_times[person_id]
            self.dwell_times[self.last_positions[person_id]] += dwell_time
            self.start_times[person_id] = time.time()
            self.zone_visits[current_zone] += 1
        
        self.last_positions[person_id] = current_zone
    
    def get_current_zone(self, x, y):
        for zone_name, boxes in self.store_zones.items():
            for (x1, y1, x2, y2) in boxes:
                if x1 <= x <= x2 and y1 <= y <= y2:
                    return zone_name
        return 'other'
    
    def generate_heatmap(self):
        plt.figure(figsize=(12, 8))
        sns.heatmap(self.heatmap_data, cmap='YlOrRd')
        plt.title('Customer Movement Heatmap')
        plt.savefig('customer_heatmap.png')
        plt.close()
    
    def get_analytics(self):
        analytics = {
            'zone_visits': dict(self.zone_visits),
            'dwell_times': {k: round(v, 2) for k, v in self.dwell_times.items()},
            'traffic_density': {
                zone: visits/max(1, sum(self.zone_visits.values()))*100 
                for zone, visits in self.zone_visits.items()
            }
        }
        return analytics

def analyze_video(video_path):
    print("\nStarting Real-Time Video Analysis...")
    
    # Load YOLO
    net = cv2.dnn.readNet("yolo_files/yolov4-tiny.weights", "yolo_files/yolov4-tiny.cfg")
    with open("yolo_files/coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    
    # Initialize tracker
    tracker = CustomerTracker()
    
    # Initialize video capture
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception("Error: Could not open video file")
    except Exception as e:
        print(f"Error loading video: {str(e)}")
        return
    
    frame_count = 0
    next_id = 0
    tracked_objects = {}
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        if frame_count % 3 != 0:  # Process every 3rd frame for efficiency
            continue
            
        height, width, _ = frame.shape
        
        # Detect objects
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)
        
        # Information to display
        class_ids = []
        confidences = []
        boxes = []
        
        # Show information on the screen
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > 0.5 and classes[class_id] == "person":
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    
                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        
        # Update tracking for this frame
        current_tracked = set()
        
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                
                # Simple tracking based on position
                tracked_id = None
                center = (x + w//2, y + h//2)
                
                # Find the closest previously tracked object
                min_dist = float('inf')
                for tid, prev_box in tracked_objects.items():
                    prev_center = (prev_box[0] + prev_box[2]//2, prev_box[1] + prev_box[3]//2)
                    dist = ((center[0] - prev_center[0])**2 + (center[1] - prev_center[1])**2)**0.5
                    if dist < min_dist and dist < 100:  # Maximum distance threshold
                        min_dist = dist
                        tracked_id = tid
                
                if tracked_id is None:
                    tracked_id = next_id
                    next_id += 1
                
                tracked_objects[tracked_id] = (x, y, w, h)
                current_tracked.add(tracked_id)
                
                # Update tracker
                tracker.update_tracking(tracked_id, x, y, w, h)
                
                # Draw bounding box and ID
                label = f"Person {tracked_id}: {confidences[i]:.2f}"
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Remove old tracks
        tracked_objects = {k: v for k, v in tracked_objects.items() if k in current_tracked}
        
        # Draw zone boundaries
        for zone_name, boxes in tracker.store_zones.items():
            for (x1, y1, x2, y2) in boxes:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, zone_name, (x1+5, y1+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # Display stats
        cv2.putText(frame, f"People Count: {len(indexes)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Display the frame
        cv2.imshow("Store Analytics", frame)
        
        # Break if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Generate final analytics
    analytics = tracker.get_analytics()
    print("\nStore Analytics Report:")
    print(f"Total Unique Customers: {next_id}")
    print("\nZone Visit Distribution:")
    for zone, visits in analytics['zone_visits'].items():
        print(f"{zone}: {visits} visits")
    print("\nAverage Dwell Times (seconds):")
    for zone, time in analytics['dwell_times'].items():
        print(f"{zone}: {time:.2f}s")
    print("\nTraffic Density (% of total traffic):")
    for zone, density in analytics['traffic_density'].items():
        print(f"{zone}: {density:.1f}%")
    
    # Generate and save heatmap
    tracker.generate_heatmap()
    print("\nHeatmap saved as 'customer_heatmap.png'")
    
    cap.release()
    cv2.destroyAllWindows()
    return analytics

def aggregate_video_analytics(video_paths):
    """Analyze multiple videos and aggregate their insights"""
    print("\nAnalyzing multiple video feeds...")
    
    aggregated_analytics = {
        'zone_visits': defaultdict(int),
        'dwell_times': defaultdict(list),
        'traffic_density': defaultdict(list),
        'total_customers': 0,
        'peak_times': defaultdict(int),
        'zone_transitions': defaultdict(int)
    }
    
    for video_path in video_paths:
        print(f"\nProcessing video: {os.path.basename(video_path)}")
        video_analytics = analyze_video(video_path)
        
        # Aggregate analytics
        for zone, visits in video_analytics['zone_visits'].items():
            aggregated_analytics['zone_visits'][zone] += visits
        
        for zone, time in video_analytics['dwell_times'].items():
            aggregated_analytics['dwell_times'][zone].append(time)
        
        for zone, density in video_analytics['traffic_density'].items():
            aggregated_analytics['traffic_density'][zone].append(density)
        
        aggregated_analytics['total_customers'] += video_analytics.get('total_customers', 0)
    
    # Calculate averages and normalize data
    final_analytics = {
        'zone_visits': dict(aggregated_analytics['zone_visits']),
        'dwell_times': {
            zone: sum(times)/len(times) 
            for zone, times in aggregated_analytics['dwell_times'].items()
        },
        'traffic_density': {
            zone: sum(densities)/len(densities)
            for zone, densities in aggregated_analytics['traffic_density'].items()
        },
        'total_customers': aggregated_analytics['total_customers']
    }
    
    return final_analytics

def generate_enhanced_insights(data, video_analytics):
    """Generate detailed insights and actionable recommendations"""
    print("\nGenerating Enhanced Store Analytics and Recommendations...")
    
    # 1. Traffic Flow Analysis
    print("\n1. Traffic Flow Analysis:")
    high_traffic_zones = []
    low_traffic_zones = []
    for zone, density in sorted(video_analytics['traffic_density'].items(), 
                              key=lambda x: x[1], reverse=True):
        print(f"{zone}: {density:.1f}% of total traffic")
        if density > 30:
            high_traffic_zones.append(zone)
        elif density < 10:
            low_traffic_zones.append(zone)
    
    # 2. Customer Engagement Analysis
    print("\n2. Customer Engagement Analysis:")
    engagement_zones = []
    for zone, time in sorted(video_analytics['dwell_times'].items(), 
                           key=lambda x: x[1], reverse=True):
        print(f"{zone}: {time:.2f} seconds average dwell time")
        if time > 30:  # More than 30 seconds considered high engagement
            engagement_zones.append(zone)
    
    # 3. Customer Segments
    print("\n3. Customer Segments:")
    spending_segments = {
        'high': len(data[data['Spending Score (1-100)'] > 75]),
        'medium': len(data[(data['Spending Score (1-100)'] > 25) & (data['Spending Score (1-100)'] <= 75)]),
        'low': len(data[data['Spending Score (1-100)'] <= 25])
    }
    for segment, count in spending_segments.items():
        print(f"{segment.title()} Spenders: {count} customers")
    
    # 4. Product Placement Strategy
    print("\n4. Product Placement Strategy:")
    recommendations = []
    
    # High-traffic area recommendations
    if high_traffic_zones:
        recommendations.append(
            f"➤ High-Impact Zones ({', '.join(high_traffic_zones)}):\n"
            "  - Place new arrivals and promotional items\n"
            "  - Install digital displays for real-time offers\n"
            "  - Position impulse purchase items"
        )
    
    # High-engagement area recommendations
    if engagement_zones:
        recommendations.append(
            f"➤ High-Engagement Zones ({', '.join(engagement_zones)}):\n"
            "  - Place complex/high-value products\n"
            "  - Install interactive product displays\n"
            "  - Position product specialists in these areas"
        )
    
    # Low-traffic area improvements
    if low_traffic_zones:
        recommendations.append(
            f"➤ Traffic Improvement Zones ({', '.join(low_traffic_zones)}):\n"
            "  - Implement attention-grabbing displays\n"
            "  - Create special promotion areas\n"
            "  - Consider layout restructuring"
        )
    
    # 5. Promotional Strategy
    print("\n5. Promotional Strategy:")
    promo_recommendations = [
        "➤ Time-Based Promotions:",
        "  - Run flash sales during peak traffic hours",
        "  - Schedule product demonstrations in high-engagement zones",
        
        "\n➤ Location-Based Promotions:",
        f"  - Position 'Deal of the Day' displays in {', '.join(high_traffic_zones[:2]) if high_traffic_zones else 'main areas'}",
        "  - Create multi-product bundles for cross-zone promotion",
        
        "\n➤ Segment-Based Promotions:",
        f"  - Target {spending_segments['high']} high-value customers with exclusive offers",
        "  - Implement loyalty program visibility in high-traffic zones"
    ]
    
    # 6. Inventory Management
    print("\n6. Inventory Management:")
    inventory_recommendations = [
        "➤ Restocking Strategy:",
        f"  - Prioritize restocking in {', '.join(high_traffic_zones)} during off-peak hours",
        "  - Implement mobile inventory checking in high-engagement zones",
        "  - Set up automated reorder triggers based on zone traffic",
        
        "\n➤ Display Management:",
        "  - Rotate product positions between high and low traffic areas",
        "  - Maintain optimal stock levels in high-engagement zones",
        "  - Implement real-time inventory tracking in high-traffic areas"
    ]
    
    # Print all recommendations
    print("\n=== ACTIONABLE RECOMMENDATIONS ===")
    print("\n" + "\n\n".join(recommendations))
    print("\n=== PROMOTIONAL STRATEGY ===")
    print("\n".join(promo_recommendations))
    print("\n=== INVENTORY MANAGEMENT ===")
    print("\n".join(inventory_recommendations))
    
    return {
        'traffic_analysis': {'high': high_traffic_zones, 'low': low_traffic_zones},
        'engagement_zones': engagement_zones,
        'customer_segments': spending_segments,
        'recommendations': recommendations
    }

def generate_insights(data, video_analytics=None):
    print("\nGenerating Comprehensive Insights...")
    
    # Customer Demographics
    print("\n1. Customer Demographics:")
    print("Gender Distribution:")
    gender_dist = data['Genre'].value_counts()
    print(gender_dist)
    print("\nAge Statistics:")
    age_stats = data['Age'].describe()
    print(age_stats)
    
    # Spending Patterns
    print("\n2. Spending Patterns:")
    print("Average Spending Score by Gender:")
    spending_by_gender = data.groupby('Genre')['Spending Score (1-100)'].mean()
    print(spending_by_gender)
    
    # Cluster Analysis
    print("\n3. Customer Segments:")
    print("Average Annual Income by Cluster:")
    income_by_cluster = data.groupby('Cluster')['Annual Income (k$)'].mean()
    print(income_by_cluster)
    
    # High-Value Customers
    high_value = data[data['Spending Score (1-100)'] > 75]
    print(f"\n4. High-Value Customers: {len(high_value)} identified")
    
    # Store Layout Optimization
    if video_analytics:
        print("\n5. Store Layout Insights:")
        print("High-Traffic Areas (based on visit frequency):")
        for zone, density in sorted(video_analytics['traffic_density'].items(), 
                                  key=lambda x: x[1], reverse=True):
            print(f"{zone}: {density:.1f}% of total traffic")
        
        print("\nCustomer Engagement Areas (based on dwell time):")
        for zone, time in sorted(video_analytics['dwell_times'].items(), 
                               key=lambda x: x[1], reverse=True):
            print(f"{zone}: {time:.2f} seconds average dwell time")
    
    # Actionable Recommendations
    print("\n6. Actionable Recommendations:")
    recommendations = []
    
    # Product Placement Recommendations
    if video_analytics:
        high_traffic_zones = [z for z, d in video_analytics['traffic_density'].items() 
                            if d > 30]  # Zones with >30% traffic
        if high_traffic_zones:
            recommendations.append(f"- Place promotional items in {', '.join(high_traffic_zones)} "
                                "to maximize visibility")
    
    # Customer Segment Recommendations
    if len(high_value) > 0:
        recommendations.append("- Implement targeted marketing for high-value customers")
    
    # Gender-Based Recommendations
    majority_gender = gender_dist.index[0]
    recommendations.append(f"- Optimize product mix for {majority_gender} customers "
                         "while maintaining diverse appeal")
    
    # Traffic Flow Recommendations
    if video_analytics and any(t < 10 for t in video_analytics['dwell_times'].values()):
        recommendations.append("- Review layout of low-dwell-time areas to improve engagement")
    
    print("\n".join(recommendations))

class CustomerTracker:
    def __init__(self, store_layout_zones=None, dwell_time_threshold=60):
        self.tracks = defaultdict(list)
        self.dwell_times = defaultdict(float)
        self.heatmap_data = np.zeros((480, 640))
        self.store_zones = store_layout_zones or {
            'entrance': [(0, 0, 200, 480)],
            'checkout': [(440, 0, 640, 480)],
            'center': [(200, 0, 440, 480)]
        }
        self.zone_visits = defaultdict(int)
        self.last_positions = {}
        self.start_times = {}
        self.dwell_time_threshold = dwell_time_threshold  # in seconds

    def update_tracking(self, person_id, x, y, w, h):
        center_x, center_y = x + w // 2, y + h // 2
        self.tracks[person_id].append((center_x, center_y))
        
        # Update heatmap
        self.heatmap_data[max(0, y):min(480, y + h), max(0, x):min(640, x + w)] += 1
        
        # Update zone analytics
        current_zone = self.get_current_zone(center_x, center_y)
        if person_id not in self.last_positions:
            self.start_times[person_id] = time.time()
        elif self.last_positions[person_id] != current_zone:
            # Zone transition
            dwell_time = time.time() - self.start_times[person_id]
            self.dwell_times[self.last_positions[person_id]] += dwell_time
            self.start_times[person_id] = time.time()
            self.zone_visits[current_zone] += 1
            
            # Check for alerts
            if self.dwell_times[self.last_positions[person_id]] > self.dwell_time_threshold:
                self.trigger_alert(self.last_positions[person_id], dwell_time)

        self.last_positions[person_id] = current_zone

    def trigger_alert(self, zone, dwell_time):
        print(f"Alert: Customer has been in {zone} for {dwell_time:.2f} seconds. Consider providing assistance or restocking.")

    def get_current_zone(self, x, y):
        for zone_name, boxes in self.store_zones.items():
            for (x1, y1, x2, y2) in boxes:
                if x1 <= x <= x2 and y1 <= y <= y2:
                    return zone_name
        return 'other'
# Process all videos in the folder
video_folder = "E:/Projects/Rohit/video"
video_paths = [
    os.path.join(video_folder, "HD CCTV Camera video 3MP 4MP iProx CCTV HDCCTVCamerasnet retail store-3-23.mp4"),
    os.path.join(video_folder, "HD CCTV Camera video 3MP 4MP iProx CCTV HDCCTVCamerasnet retail store.mp4"),
    os.path.join(video_folder, "new_codec_HD CCTV Camera video 3MP 4MP iProx CCTV HDCCTVCamerasnet retail store-3-23.mp4")
]

# Analyze all videos and generate comprehensive insights
aggregated_analytics = aggregate_video_analytics(video_paths)
enhanced_insights = generate_enhanced_insights(data, aggregated_analytics)
generate_insights(data, aggregated_analytics)
