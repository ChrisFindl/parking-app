import torch
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt
import requests

urls = {
    "parking1.jpg": "https://www.torontomu.ca/content/dam/parking/public-parking/parking-pkg.jpg",
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

img = Image.open(image_path)

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

# Count cars, trucks, and buses
count = sum(cls in ['car', 'truck', 'bus'] for cls in filtered_classes)

print(f"Detected parked vehicles: {count}")

# Optional: estimate free spots if you know total
total_capacity = 20
free_spots = total_capacity - count
print(f"Estimated free spots: {free_spots}")
