import os
import cv2

# Paths
images_dir = './data/images'
labels_dir = './data/labels'
output_dir = './cropped_heads'



# Classes (as per your labels)
HELMET_CLASS = 0
NO_HELMET_CLASS = 1

IMG_SIZE = 128

os.makedirs(os.path.join(output_dir, 'helmet'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'no_helmet'), exist_ok=True)

def yolo_to_bbox(line, img_w, img_h):
    parts = line.strip().split()
    class_id = int(parts[0])
    x_center = float(parts[1]) * img_w
    y_center = float(parts[2]) * img_h
    w = float(parts[3]) * img_w
    h = float(parts[4]) * img_h

    x1 = max(int(x_center - w/2), 0)
    y1 = max(int(y_center - h/2), 0)
    x2 = min(int(x_center + w/2), img_w-1)
    y2 = min(int(y_center + h/2), img_h-1)
    return class_id, x1, y1, x2, y2

def main():
    label_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]

    for label_file in label_files:
        image_file = label_file.replace('.txt', '.jpg')  # Adjust if image format different
        img_path = os.path.join(images_dir, image_file)
        label_path = os.path.join(labels_dir, label_file)

        img = cv2.imread(img_path)
        if img is None:
            print(f"Image not found: {img_path}")
            continue

        h, w = img.shape[:2]
        with open(label_path, 'r') as f:
            lines = f.readlines()

        for i, line in enumerate(lines):
            class_id, x1, y1, x2, y2 = yolo_to_bbox(line, w, h)
            crop = img[y1:y2, x1:x2]

            if crop.size == 0:
                continue

            crop = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))

            if class_id == HELMET_CLASS:
                save_folder = os.path.join(output_dir, 'helmet')
            elif class_id == NO_HELMET_CLASS:
                save_folder = os.path.join(output_dir, 'no_helmet')
            else:
                continue

            save_path = os.path.join(save_folder, f"{image_file.split('.')[0]}_{i}.jpg")
            cv2.imwrite(save_path, crop)

        print(f"Processed {image_file}")

if __name__ == "__main__":
    main()
