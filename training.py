import cv2
import os
import numpy as np

# Hàm trích xuất đặc trưng của ảnh sử dụng HOG
def extract_features(image_path):
    # Đọc ảnh
    image = cv2.imread(image_path)
    if image is None:
        print(f"Không thể đọc được ảnh: {image_path}")
        return None
    
    # Chuyển ảnh sang grayscale nếu ảnh là RGB hoặc BGR
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Kích thước chuẩn để HOG hoạt động hiệu quả
    image = cv2.resize(image, (64, 128))  # Kích thước phổ biến cho HOG là 64x128
    
    # Cấu hình các tham số HOG
    hog = cv2.HOGDescriptor(
        _winSize=(64, 128),
        _blockSize=(16, 16),
        _blockStride=(8, 8),
        _cellSize=(8, 8),
        _nbins=9 
    )
    
    # Tính toán vector đặc trưng HOG cho ảnh
    features = hog.compute(image)
    features = features.flatten()  # Chuyển thành mảng một chiều
    return features

# Hàm lưu đặc trưng của ảnh cùng nhãn vào file
def save_features_to_file(features, label, file_path='train_data.txt'):
    with open(file_path, 'a') as f:
        f.write(','.join(map(str, features)) + f',{label}\n')

# Hàm chính thực hiện dán nhãn tự động cho ảnh
def main():
    train_dir = 'D:/XLA/New2/Animals'  # Thay thế bằng đường dẫn thực tế
    
    for folder_name in os.listdir(train_dir):
        folder_path = os.path.join(train_dir, folder_name)
        
        if os.path.isdir(folder_path):
            label = folder_name
            
            for image_name in os.listdir(folder_path):
                image_path = os.path.join(folder_path, image_name)
                
                if os.path.isfile(image_path) and image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    features = extract_features(image_path)
                    if features is not None:
                        save_features_to_file(features, label)
                    else:
                        print(f"Không thể trích xuất đặc trưng cho ảnh: {image_path}")

if __name__ == "__main__":
    main()
