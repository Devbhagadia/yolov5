import os
import torch
import cv2
import numpy as np
from yolov5 import YOLOv5
from deep_sort_pytorch.deep_sort import DeepSort
import sys

def resize_with_aspect_ratio(image, new_shape):
    h, w = image.shape[:2]
    new_w, new_h = new_shape
    scale = min(new_w / w, new_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    new_w = new_w - (new_w % 32)
    new_h = new_h - (new_h % 32)

    resized_image = cv2.resize(image, (new_w, new_h))
    canvas = np.full((new_h, new_w, 3), 128, dtype=np.uint8)
    top = (new_h - resized_image.shape[0]) // 2
    left = (new_w - resized_image.shape[1]) // 2
    canvas[top:top + resized_image.shape[0], left:left + resized_image.shape[1]] = resized_image

    return canvas

def process_video(video_file, output_dir, weights):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = YOLOv5(weights, device=device)
    deepsort = DeepSort("C:\\Users\\Dev\\yolov5\\models\\osnet_x1_0_imagenet.pth")

    print(f"Processing {video_file}...")
    source = video_file
    output_file = os.path.join(output_dir, os.path.basename(video_file).replace('.mp4', '_output.mp4'))

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Failed to open video {video_file}")
        return

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        img = resize_with_aspect_ratio(frame, (416, 416))  # Use a size divisible by 32
        img = img / 255.0
        img = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0).to(device)

        results = model.predict(img)
        predictions = results[0].cpu().numpy()

        if len(predictions):
            bbox_xyxy = predictions[:, :4]
            confs = predictions[:, 4]
            clss = predictions[:, 5]

            # Filter predictions based on confidence threshold
            conf_threshold = 0.5  # Adjust as needed
            mask = confs > conf_threshold
            bbox_xyxy = bbox_xyxy[mask]
            confs = confs[mask]
            clss = clss[mask]

            if len(bbox_xyxy) > 0:
                outputs, _ = deepsort.update(bbox_xyxy, confs, clss, frame)

                if outputs is not None and len(outputs) > 0:
                    for output in outputs:  # Iterate over each detected object
                        if len(output) >= 6:  # Ensure there's enough data (bbox and track ID)
                            # Extract bbox and track ID
                            bbox = output[:4].astype(int)
                            x1, y1, x2, y2 = bbox
                            track_id = int(output[5].item())  # Use .item() to extract scalar from numpy array

                            # Convert to tuples for cv2.rectangle and cv2.putText
                            pt1 = (x1, y1)
                            pt2 = (x2, y2)
                            cv2.rectangle(frame, pt1, pt2, (0, 255, 0), 2)
                            cv2.putText(frame, f'ID: {track_id}', (pt1[0], pt1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        out.write(frame)

    # Release resources after processing
    cap.release()
    out.release()
    print(f"Finished processing {video_file}. Saved to {output_file}.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python new_detect.py <video_file>")
        sys.exit(1)

    video_file = sys.argv[1]
    output_dir = "C:\\Users\\Dev\\yolov5\\outputs"
    weights = "C:\\Users\\Dev\\yolov5\\yolov5s.pt"
    os.makedirs(output_dir, exist_ok=True)

    process_video(video_file, output_dir, weights)
