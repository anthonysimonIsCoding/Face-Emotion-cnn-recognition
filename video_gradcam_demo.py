import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image  # tf.keras
from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
from tf_keras_vis.utils.scores import CategoricalScore

# ---------------------------
# Hyper-params
# ---------------------------
frame_window = 10  # Số frames average probs (smooth label)
gender_offsets = (30, 60)  # Expand crop gender (ngang, dọc)
emotion_offsets = (20, 40)  # Expand crop emotion

# ---------------------------
# Load models
# ---------------------------
gender_model = load_model("./model/gender_model/simple_CNN.81-0.96.hdf5", compile=False)
print("✅ Gender model loaded!")

emotion_model = load_model("model/emotion_model/fer2013_mini_XCEPTION.102-0.66.hdf5", compile=False)
print("✅ Emotion model loaded!")

face_haar_cascade = cv2.CascadeClassifier('model/detection_model/haarcascade_frontalface_default.xml')

# ---------------------------
# Start webcam
# ---------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Cannot open webcam!")
    exit()

gender_labels = ('Female', 'Male')
emotion_labels = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')

# Target sizes
gender_target_size = gender_model.input_shape[1:3]
emotion_target_size = emotion_model.input_shape[1:3]

# Preprocess
def preprocess_input(x, v2=True):
    x = x.astype('float32')
    x = x / 255.0
    if v2:
        x = x - 0.5
        x = x * 2.0
    return x

# Apply offsets
def apply_offsets(face_coordinates, offsets):
    x, y, width, height = face_coordinates
    x_off, y_off = offsets
    return (x - x_off, x + width + x_off, y - y_off, y + height + y_off)

# GradCAM layer detect
def get_conv_layer(model):
    for layer in reversed(model.layers):
        if any(k in layer.name.lower() for k in ['conv', 'separable', 'depthwise']):
            idx = -model.layers.index(layer) - 1
            print(f"Using emotion layer idx {idx}: {layer.name}")
            return idx
    return -1

emotion_layer_idx = get_conv_layer(emotion_model)

# GradCAM + Guided Backprop simulate (tf_keras_vis có guided)
gradcam = Gradcam(emotion_model, model_modifier=ReplaceToLinear(), clone=True)

# Threshold gender
FEMALE_THRESHOLD = 0.3
MALE_THRESHOLD = 0.6

# History buffers smooth
gender_history = []
emotion_history = []

while True:
    ret, frame = cap.read()
    if not ret:
        continue
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))
    
    for (x, y, w, h) in faces:
        # Vẽ bbox
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        # Crop gender với offsets
        gx1, gx2, gy1, gy2 = apply_offsets((x, y, w, h), gender_offsets)
        gx1 = max(0, gx1); gx2 = min(frame.shape[1], gx2)
        gy1 = max(0, gy1); gy2 = min(frame.shape[0], gy2)
        roi_color_gender = frame[gy1:gy2, gx1:gx2]
        roi_color_rgb = cv2.cvtColor(roi_color_gender, cv2.COLOR_BGR2RGB)
        img_gender = cv2.resize(roi_color_rgb, gender_target_size)
        img_gender = preprocess_input(img_gender, v2=False)
        img_gender = np.expand_dims(img_gender, 0)
        
        # Predict gender
        gender_pred = gender_model.predict(img_gender)
        female_prob = gender_pred[0][0]
        male_prob = 1 - female_prob
        
        # Smooth gender
        gender_history.append(female_prob)
        if len(gender_history) > frame_window:
            gender_history.pop(0)
        avg_female = np.mean(gender_history)
        avg_male = 1 - avg_female
        
        if avg_female > FEMALE_THRESHOLD:
            gender_label = "Female"
        elif avg_male > MALE_THRESHOLD:
            gender_label = "Male"
        else:
            gender_label = "Unknown"
        
        # Crop emotion với offsets
        ex1, ex2, ey1, ey2 = apply_offsets((x, y, w, h), emotion_offsets)
        ex1 = max(0, ex1); ex2 = min(gray.shape[1], ex2)
        ey1 = max(0, ey1); ey2 = min(gray.shape[0], ey2)
        roi_gray_emotion = gray[ey1:ey2, ex1:ex2]
        img_emotion = cv2.resize(roi_gray_emotion, emotion_target_size)
        img_emotion = preprocess_input(img_emotion, v2=True)
        img_emotion = np.expand_dims(img_emotion, 0)
        img_emotion = np.expand_dims(img_emotion, -1)
        
        # Predict emotion
        emotion_pred = emotion_model.predict(img_emotion)
        emotion_probs = emotion_pred[0]
        
        # Smooth emotion
        emotion_history.append(emotion_probs)
        if len(emotion_history) > frame_window:
            emotion_history.pop(0)
        avg_emotion_probs = np.mean(emotion_history, axis=0)
        emotion_label_index = np.argmax(avg_emotion_probs)
        emotion_label = emotion_labels[emotion_label_index]
        
        # GradCAM heatmap (guided-like: dùng tf_keras_vis + relu clip positive)
        score = CategoricalScore(emotion_label_index)
        cam = gradcam(score, img_emotion, penultimate_layer=emotion_layer_idx)
        heatmap = np.squeeze(cam[0])
        heatmap = np.maximum(heatmap, 0)  # Guided-like: chỉ positive activations
        
        # Resize & normalize
        heatmap = cv2.resize(heatmap, (w, h))
        h_max = np.max(heatmap)
        if h_max > 0:
            heatmap = (heatmap / h_max) * 255
        else:
            heatmap = np.zeros((h, w), dtype=np.uint8)
        heatmap = np.uint8(heatmap)
        heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # Overlay
        frame[y:y + h, x:x + w] = cv2.addWeighted(frame[y:y + h, x:x + w], 0.7, heatmap_colored, 0.3, 0)
        
        # Vẽ label (color gender)
        color_label = (0, 255, 0) if gender_label == "Unknown" else ((0, 0, 255) if gender_label == "Female" else (255, 0, 0))
        cv2.putText(frame, f"{gender_label} | {emotion_label}", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color_label, 2)
    
    # Show frame
    resized_frame = cv2.resize(frame, (1000, 700))
    cv2.imshow('Gender + Emotion with Guided Grad-CAM', resized_frame)
    
    if cv2.waitKey(10) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()