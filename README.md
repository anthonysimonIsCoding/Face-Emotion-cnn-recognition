# HỆ THỐNG NHẬN DIỆN GIỚI TÍNH VÀ CẢM XÚC KHUÔN MẶT THỜI GIAN THỰC  
**Kèm giải thích trực quan bằng Grad-CAM**

**Kiến trúc mini-Xception nhẹ • FER2013 & IMDB-WIKI • TensorFlow 2.12 • OpenCV**

## 1. Tóm tắt (Abstract)

Dự án triển khai một hệ thống nhận diện khuôn mặt thời gian thực có khả năng đồng thời dự đoán **giới tính** (nam/nữ) và **bảy loại cảm xúc cơ bản** (giận dữ, chán ghét, sợ hãi, vui vẻ, buồn bã, ngạc nhiên, trung tính) dựa trên kiến trúc **mini-Xception** – một biến thể nhẹ của Xception (Arriaga et al., 2017).  

Hệ thống tích hợp kỹ thuật **Grad-CAM** (thông qua thư viện `tf-keras-vis`) để sinh ra các bản đồ nhiệt giải thích trực quan, chỉ rõ những vùng trên khuôn mặt đóng góp lớn nhất vào quyết định của mô hình. V

## Các tính năng chính

| Tính năng                                           | Trạng thái |
|-----------------------------------------------------|------------|
| Nhận diện giới tính (2 lớp)                         | Hoàn thiện |
| Nhận diện 7 cảm xúc cơ bản (FER2013)                | Hoàn thiện |
| Phát hiện khuôn mặt thời gian thực (Haar Cascade)   | Hoàn thiện |
| Trực quan hóa Grad-CAM cho cả hai nhiệm vụ          | Hoàn thiện |
| Xử lý ảnh tĩnh và lưu kết quả chi tiết              | Hoàn thiện |
| Hỗ trợ nhiều khuôn mặt trong một khung hình         | Hoàn thiện |
| Cơ chế flip + average giảm bias giới tính           | Hoàn thiện |
| Hướng dẫn huấn luyện lại trên FER2013 và IMDB-WIKI  | Hoàn thiện |

## 2. Cấu trúc thư mục
<pre>
Face-Emotion-cnn-recognition/
├── datasets/
│   ├── emotion/
│   │   ├── test/
│   │   └── train/
│   └── gender/
│       └── imdb_crop/
├── images/
├── model/
│   ├── detection_model/haarcascade_frontalface_default.xml   ← Mô hình Haar Cascade phát hiện khuôn mặt
│   ├── gender_model/simple_CNN.81-0.96.hdf5                  ← Mô hình CNN đã huấn luyện để phân loại giới tính
│   └── emotion_model/fer2013_mini_XCEPTION.102-0.66.hdf5     ← Mô hình CNN (Mini XCEPTION) phân loại cảm xúc
├── environment.yml
├── gradcam_output.jpg
├── image_emotion_gender_demo.py
├── image_gradcam_demo.py                                 ← 🖼️ Demo Ảnh tĩnh + Grad-CAM chi tiết + lưu kết quả khoa học
├── README.md
├── train_emotion_classifier.py                           ← ⚙️ Tệp mã nguồn để Huấn luyện lại mô hình phân loại cảm xúc (FER2013)
├── train_gender_classifier_imdb.py                       ← ⚙️ Tệp mã nguồn để Huấn luyện lại mô hình phân loại giới tính (IMDB-WIKI)
├── video_emotion_color_demo.py
├── video_emotion_gender_demo.py
└── video_gradcam_demo.py                                 ← 🎥 Demo chính: Webcam thời gian thực + Hiển thị Grad-CAM (trực quan nhất)
</pre>

## 3. Hướng dẫn cài đặt

```bash
git clone https://github.com/<username>/Face-Emotion-cnn-recognition.git
cd Face-Emotion-cnn-recognition
conda env create -f environment.yml
conda activate faceai-gradcam

```

## Môi trường thực thi

```bash
name: faceai-gradcam
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.10
  - pip
  - opencv
  - numpy
  - matplotlib
  - scipy
  - scikit-learn
  - pillow
  - pip:
    - tensorflow==2.12.0          # Phiên bản tương thích hoàn hảo với model cũ
    - tf-keras-vis==0.8.7         # Grad-CAM visualization
    - tqdm
    - seaborn
    - pandas
```

## 4. Các lệnh khởi chạy

| Mục đích | Lệnh thực thi |
| :--- | :--- |
| **Webcam thời gian thực + Grad-CAM** | `python video_gradcam_demo.py` |
| Webcam + đổi màu khung theo cảm xúc | `python video_emotion_color_demo.py` |
| Ảnh tĩnh + **Grad-CAM chi tiết** + lưu kết quả | `python image_gradcam_demo.py "đường_dẫn_ảnh.jpg"` |
| Huấn luyện lại mô hình cảm xúc (FER2013) | `python train_emotion_classifier.py` |
| Huấn luyện lại mô hình giới tính (IMDB-WIKI) | `python train_gender_classifier_imdb.py` |

## 5. Tập dữ liệu (Datasets)

### 5.1. FER2013 – Facial Expression Recognition 2013  
**Nguồn**: Kaggle – Challenges in Representation Learning  
**Link tải**: https://www.kaggle.com/datasets/msambare/fer2013  

| Thông tin                  | Chi tiết                                      |
|----------------------------|-----------------------------------------------|
| Tổng số ảnh                | 35.887 ảnh (grayscale, 48×48 pixel)           |
| Số lớp                     | 7 (angry, disgust, fear, happy, sad, surprise, neutral) |
| Phân chia gốc              | 28.709 train / 3.589 validation / 3.589 test  |
| Đặc điểm                   | Ảnh đã được căn chỉnh và crop khuôn mặt       |
| Độ khó                     | Có nhiều ảnh nhiễu, ánh sáng kém, góc nghiêng |
| Độ chính xác SOTA (2025)   | 73–75 % (private test)                        |

> **Lưu ý**: Trong dự án này, tập validation được dùng làm tập test để đánh giá cuối cùng.

### 5.2. IMDB-WIKI (crop face only) – Nhận diện giới tính & tuổi  
**Nguồn**: Computer Vision Laboratory, ETH Zurich  
**Link tải**: https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/  

| Thông tin                  | Chi tiết                                      |
|----------------------------|-----------------------------------------------|
| IMDB                       | 460.723 ảnh khuôn mặt (đã crop) từ 20.284 ngôi sao |
| WIKI                       | 62.328 ảnh khuôn mặt từ Wikipedia             |
| Tổng (sau lọc)             | ~500.000 ảnh chất lượng cao                   |
| Nhãn                      | Giới tính (male/female), tuổi thực, face_score |
| Độ phân giải               | Đa dạng (thường ≥ 64×64)                      |
| Độ chính xác gender SOTA   | 98.0–98.5 % trên tập test riêng               |

> Trong dự án, chỉ sử dụng **IMDB crop** và áp dụng bộ lọc:  
> `face_score > 3.0` và `second_face_score is NaN` → loại bỏ ảnh nhiễu và ảnh có nhiều khuôn mặt.
> <img width="321" height="32" alt="image" src="https://github.com/user-attachments/assets/62113d9c-f4a7-4c91-9f4f-eddd519d916a" />





