import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image  # tf.keras
from statistics import mode

# ---------------------------
# Hyper-parameters
# ---------------------------
frame_window = 10  # Số frames average probs (smooth label)
emotion_offsets = (20, 40)  # Expand crop emotion (ngang, dọc)

# ---------------------------
# Load full mini_XCEPTION emotion model (full model .hdf5)
# ---------------------------
emotion_model = load_model("model/emotion_model/fer2013_mini_XCEPTION.102-0.66.hdf5", compile=False)
print("✅ Emotion model loaded successfully!")

# ---------------------------
# Load Haar Cascade for face detection
# ---------------------------
face_haar_cascade = cv2.CascadeClassifier('model/detection_model/haarcascade_frontalface_default.xml')

# ---------------------------
# Emotion labels
# ---------------------------
emotion_labels = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')

# Target size từ model (dynamic, ko hardcoded)
emotion_target_size = emotion_model.input_shape[1:3]  # (48,48)

# Preprocess utils (v2=True cho emotion [-1,1])
def preprocess_input(x, v2=True):
    x = x.astype('float32')
    x = x / 255.0
    if v2:
        x = x - 0.5
        x = x * 2.0
    return x

# Apply offsets (expand crop)
def apply_offsets(face_coordinates, offsets):
    x, y, width, height = face_coordinates
    x_off, y_off = offsets
    return (x - x_off, x + width + x_off, y - y_off, y + height + y_off)

# Starting lists for calculating modes (history probs cho smooth)
emotion_history = []  # Probs [7 classes] qua frames

# ---------------------------
# Start webcam
# ---------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Cannot open webcam!")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        continue
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))  # Params tốt hơn
    
    for (x, y, w, h) in faces:
        # Vẽ bbox mặt (màu default)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        # Apply offsets để crop lớn hơn cho emotion
        ex1, ex2, ey1, ey2 = apply_offsets((x, y, w, h), emotion_offsets)
        
        # Đảm bảo không vượt frame
        ex1 = max(0, ex1)
        ey1 = max(0, ey1)
        ex2 = min(gray.shape[1], ex2)
        ey2 = min(gray.shape[0], ey2)
        
        # Crop vùng cho emotion model: ảnh xám, resize về emotion_target_size
        roi_gray_emotion = gray[ey1:ey2, ex1:ex2]
        img_emotion = cv2.resize(roi_gray_emotion, emotion_target_size)
        
        # Chuẩn bị cho emotion model (preprocess v2=True)
        img_emotion = preprocess_input(img_emotion, v2=True)
        img_emotion = np.expand_dims(img_emotion, axis=0)
        img_emotion = np.expand_dims(img_emotion, axis=-1)  # Channel dim cho grayscale
        
        # Predict emotion
        emotion_pred = emotion_model.predict(img_emotion)
        emotion_probs = emotion_pred[0]  # [7 classes probs]
        emotion_label_arg = np.argmax(emotion_probs)
        emotion_text = emotion_labels[emotion_label_arg]
        emotion_probability = emotion_probs[emotion_label_arg]  # Prob của class max
        
        # Smooth emotion probs (average qua window)
        emotion_history.append(emotion_probs)
        if len(emotion_history) > frame_window:
            emotion_history.pop(0)
        avg_emotion_probs = np.mean(emotion_history, axis=0)
        emotion_mode_arg = np.argmax(avg_emotion_probs)
        emotion_mode = emotion_labels[emotion_mode_arg]
        emotion_mode_prob = avg_emotion_probs[emotion_mode_arg]  # Prob smoothed
        
        # Chọn màu dựa trên emotion_mode (như code cũ, scale bằng prob smoothed)
        if emotion_mode == 'angry':
            color = emotion_mode_prob * np.asarray((255, 0, 0))
        elif emotion_mode == 'sad':
            color = emotion_mode_prob * np.asarray((0, 0, 255))
        elif emotion_mode == 'happy':
            color = emotion_mode_prob * np.asarray((255, 255, 0))
        elif emotion_mode == 'surprise':
            color = emotion_mode_prob * np.asarray((0, 255, 255))
        else:  # disgust, fear, neutral
            color = emotion_mode_prob * np.asarray((0, 255, 0))
        
        color = color.astype(int).tolist()
        
        # Vẽ bounding box với màu
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        
        # Vẽ kết quả (chỉ emotion_mode)
        cv2.putText(frame, f"{emotion_mode}", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    
    # Show the frame
    resized_frame = cv2.resize(frame, (1000, 700))
    cv2.imshow('Emotion Recognition', resized_frame)
    
    if cv2.waitKey(10) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()