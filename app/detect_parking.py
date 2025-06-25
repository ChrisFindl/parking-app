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

urls = {
    "parking1.jpg": "https://www.torontomu.ca/content/dam/parking/public-parking/parking-pkg.jpg",
    #"parking2.jpg": "https://www.reliance-foundry.com/wp-content/uploads/parking-lot-safety.jpg",
}


for filename, url in urls.items():
    r = requests.get(url)
    with open(filename, 'wb') as f:
        f.write(r.content)
    print(f"Downloaded {filename}")


# Load pretrained DETR model
model = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=True)
model.eval()

# Load and transform image
image_path = "parking1.jpg"  # or parking2.jpg, parking3.jpg
#image_path = "parking2.jpg"  # Change to the desired image
#image_path = "parking3.jpg"  # Change to the desired image

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
threshold = 0.5

keep = probs.max(-1).values > threshold
labels = probs[keep].argmax(-1)
filtered_classes = [CLASSES[i] for i in labels.tolist()]

# Filter for vehicle-related classes
vehicle_labels = ['car', 'truck', 'bus']
vehicle_indices = [i for i, cls in enumerate(filtered_classes) if cls in vehicle_labels]
vehicle_boxes = boxes[keep][vehicle_indices]

img_width, img_height = img.size

# Get (x, y) centers of each vehicle box in pixel space
vehicle_centers = [(box[0].item() * img_width, box[1].item() * img_height) for box in vehicle_boxes]
vehicle_centers = np.array(vehicle_centers)

if len(vehicle_centers) >= 2:
    # Estimate number of rows (you can tune this or make dynamic based on spread)
    # Try clustering from 2 to 8 rows and choose the best one using silhouette score
    best_k = 1
    best_score = -1

    if len(vehicle_centers) > 2:
        for k in range(2, min(9, len(vehicle_centers))):
            kmeans_test = KMeans(n_clusters=k, n_init='auto')
            labels = kmeans_test.fit_predict(vehicle_centers[:, 1].reshape(-1, 1))
            score = silhouette_score(vehicle_centers[:, 1].reshape(-1, 1), labels)
            if score > best_score:
                best_score = score
                best_k = k

        kmeans = KMeans(n_clusters=best_k, n_init='auto')
        row_labels = kmeans.fit_predict(vehicle_centers[:, 1].reshape(-1, 1))
    else:
        row_labels = np.zeros(len(vehicle_centers), dtype=int)
        row_labels = kmeans.fit_predict(vehicle_centers[:, 1].reshape(-1, 1))

    estimated_capacity = 0
    for row_id in np.unique(row_labels):
        row_cars = vehicle_centers[row_labels == row_id]
        row_cars = row_cars[np.argsort(row_cars[:, 0])]  # sort by X position
        if len(row_cars) > 1:
            x_diffs = np.diff(row_cars[:, 0])
            avg_spacing = np.median(x_diffs)
            row_capacity = round((row_cars[-1, 0] - row_cars[0, 0]) / avg_spacing) + 1
        else:
            row_capacity = 1
        estimated_capacity += row_capacity
else:
    estimated_capacity = 10  # fallback

# ------------------ These are option calculations based on image size ---------------
# The first one is more accurate (provides total capacity), the second one is a rough estimate

# Optional: estimate free spots if you know total
#** This stuff works best for image 1
# total_capacity = 20
# free_spots = total_capacity - count
# print(f"Estimated free spots: {free_spots}")

# image_width, image_height = img.size
# pixels_per_car = 150000  # Tweak this number
# total_capacity = round((image_width * image_height) / pixels_per_car)
# print(f"Estimated total capacity: {total_capacity}")
#------------------------------------------------------------------------------------

# Final count
vehicle_count = len(vehicle_boxes)
free_spots = estimated_capacity - vehicle_count

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

ax.set_title(f"Detected Vehicles: {vehicle_count} | Estimated Capacity: {estimated_capacity} | Free Spots: {free_spots}")
ax.axis('off')
plt.tight_layout()
plt.show()

print(f"Detected parked vehicles: {vehicle_count}")
print(f"Estimated total capacity: {estimated_capacity}")
print(f"Estimated free spots: {free_spots}")
