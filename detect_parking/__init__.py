import torch
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import os

def main(image_path, total_capacity, threshold=0.95, save_output=False, output_path="detection_output.png"):
    """
    Runs parking lot detection on an image.

    Args:
        image_path (str): Path to the parking lot image.
        total_capacity (int): Total parking slots.
        threshold (float): Confidence threshold for detection.
        save_output (bool): Whether to save the annotated image.
        output_path (str): Where to save the output image.

    Returns:
        dict: Summary of detection counts.
    """
    # -------------------------------------------
    # 1. Load image and prepare transform
    # -------------------------------------------
    img = Image.open(image_path).convert("RGB")
    img_width, img_height = img.size

    transform = T.Compose([
        T.Resize(800),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225])
    ])
    img_tensor = transform(img).unsqueeze(0)

    # -------------------------------------------
    # 2. Load pretrained DETR model
    # -------------------------------------------
    model = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=True)
    model.eval()

    # -------------------------------------------
    # 3. Run inference
    # -------------------------------------------
    with torch.no_grad():
        outputs = model(img_tensor)

    # -------------------------------------------
    # 4. Process outputs to filter vehicles
    # -------------------------------------------
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

    probs = outputs['pred_logits'].softmax(-1)[0, :, :-1]
    boxes = outputs['pred_boxes'][0]

    keep = probs.max(-1).values > threshold
    labels = probs[keep].argmax(-1)
    filtered_classes = [CLASSES[i] for i in labels.tolist()]

    vehicle_labels = ['car', 'truck', 'bus', 'motorcycle']
    vehicle_indices = [i for i, cls in enumerate(filtered_classes) if cls in vehicle_labels]
    vehicle_boxes = boxes[keep][vehicle_indices]

    # -------------------------------------------
    # 5. Cluster into rows
    # -------------------------------------------
    vehicle_centers = [(box[0].item() * img_width, box[1].item() * img_height) for box in vehicle_boxes]
    vehicle_centers = np.array(vehicle_centers)

    if len(vehicle_centers) >= 2:
        best_k, best_score = 1, -1
        if len(vehicle_centers) > 2:
            for k in range(2, min(9, len(vehicle_centers))):
                labels_k = KMeans(n_clusters=k, n_init='auto').fit_predict(vehicle_centers[:, 1].reshape(-1, 1))
                score = silhouette_score(vehicle_centers[:, 1].reshape(-1, 1), labels_k)
                if score > best_score:
                    best_score, best_k = score, k
        row_labels = KMeans(n_clusters=best_k, n_init='auto').fit_predict(vehicle_centers[:, 1].reshape(-1, 1))
    else:
        row_labels = np.zeros(len(vehicle_centers), dtype=int)

    # -------------------------------------------
    # 6. Estimate free spots
    # -------------------------------------------
    vehicle_count = len(vehicle_boxes)
    free_spots = total_capacity - vehicle_count

    # -------------------------------------------
    # 7. Plot results
    # -------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(img)
    cmap = plt.cm.get_cmap('tab10', np.unique(row_labels).size)

    for i, (box, label) in enumerate(zip(vehicle_boxes, row_labels)):
        cx, cy, w, h = box.tolist()
        x = (cx - w / 2) * img_width
        y = (cy - h / 2) * img_height
        rect = patches.Rectangle((x, y), w * img_width, h * img_height,
                                 linewidth=2, edgecolor=cmap(label), facecolor='none')
        ax.add_patch(rect)
        ax.text(x, y - 5, f"Row {label+1}", color=cmap(label), fontsize=8)

    ax.set_title(f"Detected Vehicles: {vehicle_count} | Capacity: {total_capacity} | Free: {free_spots}")
    ax.axis('off')
    plt.tight_layout()

    if save_output:
        plt.savefig(output_path)
        print(f"Saved detection output to {output_path}")
    plt.close(fig)  # Important for API memory cleanup

    # -------------------------------------------
    # 8. Return summary
    # -------------------------------------------
    return {
        "detected": vehicle_count,
        "capacity": total_capacity,
        "free": free_spots,
        "output_file": output_path if save_output else None
    }
