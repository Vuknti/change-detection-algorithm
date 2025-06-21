import cv2
import os
import numpy as np

def find_image_pairs(input_dir):
    files = os.listdir(input_dir)
    base_names = set(f.split('~')[0] for f in files if '~2' in f)
    pairs = [(os.path.join(input_dir, f"{name}.jpg"), os.path.join(input_dir, f"{name}~2.jpg")) for name in base_names]
    return pairs

def detect_changes(before_img, after_img):
    before_gray = cv2.cvtColor(before_img, cv2.COLOR_BGR2GRAY)
    after_gray = cv2.cvtColor(after_img, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(before_gray, after_gray)
    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5,5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, kernel)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    output = after_img.copy()
    for cnt in contours:
        if cv2.contourArea(cnt) > 500:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 0, 255), 2)
    return output

def process_folder(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    pairs = find_image_pairs(input_dir)
    for before_path, after_path in pairs:
        before = cv2.imread(before_path)
        after = cv2.imread(after_path)
        result = detect_changes(before, after)
        base_name = os.path.basename(before_path).replace('.jpg', '')
        output_path = os.path.join(output_dir, f"{base_name}_diff.jpg")
        cv2.imwrite(output_path, result)
        print(f"Saved: {output_path}")

if __name__ == "__main__":
    input_folder = "input"
    output_folder = "output"
    process_folder(input_folder, output_folder)
