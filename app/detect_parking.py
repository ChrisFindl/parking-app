import torch
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import requests
import numpy as np
import sklearn.cluster
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import torchvision.ops as ops  # For NMS

urls = {
    "parking1.jpg": "https://www.torontomu.ca/content/dam/parking/public-parking/parking-pkg.jpg",
    #"parking2.jpg": "https://www.reliance-foundry.com/wp-content/uploads/parking-lot-safety.jpg",
}

for filename, url in urls.items():
    r = requests.get(url)
    with open(filename, 'wb') as f:
        f.write(r.content)
    print(f"Downloaded {filename}")

# Hardcoded total capacities for your parking lot images
total_capacity_map = {
    "parking1.jpg": 21,  # hardcoded total slots for parking1
    #"parking2.jpg": 89,  # hardcoded total slots for parking2
}

# Load pretrained DETR model
model = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=True)
model.eval()

# Change this to test a specific image
image_path = "parking1.jpg"  # or "parking1.jpg", "parking2.jpg"

img = Image.open(image_path).convert("RGB")

transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225])
])
img_tensor = transform(img).unsqueeze(0)

# Inference
with torch.no_grad():
    outputs = model(img_tensor)

# COCO label map
CLASSES = [
    'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
    'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
    'N/A', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

# Process model outputs
probs = outputs['pred_logits'].softmax(-1)[0, :, :-1]
boxes = outputs['pred_boxes'][0]
#Change threshold for detection confidence across different images
threshold = 0.5

keep = probs.max(-1).values > threshold
labels = probs[keep].argmax(-1)
filtered_classes = [CLASSES[i] for i in labels.tolist()]

# Filter for vehicle-related classes
vehicle_labels = ['car', 'truck', 'bus', 'motorcycle']
vehicle_indices = [i for i, cls in enumerate(filtered_classes) if cls in vehicle_labels]
vehicle_boxes = boxes[keep][vehicle_indices]
probs_keep = probs[keep].max(-1).values[vehicle_indices]

# Apply Non-Maximum Suppression to remove duplicates
nms_threshold = 0.5
keep_indices = ops.nms(vehicle_boxes, probs_keep, nms_threshold)
vehicle_boxes = vehicle_boxes[keep_indices]
probs_keep = probs_keep[keep_indices]

img_width, img_height = img.size

# Get (x, y) centers of each vehicle box in pixel space
vehicle_centers = [(box[0].item() * img_width, box[1].item() * img_height) for box in vehicle_boxes]
vehicle_centers = np.array(vehicle_centers)

if len(vehicle_centers) >= 2:
    best_k = 1
    best_score = -1

    if len(vehicle_centers) > 2:
        for k in range(2, min(9, len(vehicle_centers))):
            kmeans_test = KMeans(n_clusters=k, n_init='auto')
            labels_k = kmeans_test.fit_predict(vehicle_centers[:, 1].reshape(-1, 1))
            score = silhouette_score(vehicle_centers[:, 1].reshape(-1, 1), labels_k)
            if score > best_score:
                best_score = score
                best_k = k

        kmeans = KMeans(n_clusters=best_k, n_init='auto')
        row_labels = kmeans.fit_predict(vehicle_centers[:, 1].reshape(-1, 1))
    else:
        kmeans = KMeans(n_clusters=1, n_init='auto')
        row_labels = kmeans.fit_predict(vehicle_centers[:, 1].reshape(-1, 1))
else:
    row_labels = np.zeros(len(vehicle_centers), dtype=int)

# Use hardcoded total capacity instead of estimating
total_capacity = total_capacity_map.get(image_path, 20)  # default fallback

# Count detected vehicles
vehicle_count = len(vehicle_boxes)

# Calculate free spots
free_spots = total_capacity - vehicle_count

# Visual Debug Output
fig, ax = plt.subplots(figsize=(12, 8))
ax.imshow(img)

colors = plt.cm.get_cmap('tab10', np.unique(row_labels).size)

for i, (box, label) in enumerate(zip(vehicle_boxes, row_labels)):
    cx, cy, w, h = box.tolist()
    x = (cx - w / 2) * img_width
    y = (cy - h / 2) * img_height
    rect = patches.Rectangle((x, y), w * img_width, h * img_height,
                             linewidth=2, edgecolor=colors(label), facecolor='none')
    ax.add_patch(rect)
    ax.text(x, y - 5, f"Row {label+1}", color=colors(label), fontsize=8)

ax.set_title(f"Detected Vehicles: {vehicle_count} | Total Capacity: {total_capacity} | Free Spots: {free_spots}", color='green')
ax.axis('off')
plt.tight_layout()
plt.show()

print(f"Detected parked vehicles: {vehicle_count}")
print(f"Total capacity (hardcoded): {total_capacity}")
print(f"Estimated free spots: {free_spots}")
