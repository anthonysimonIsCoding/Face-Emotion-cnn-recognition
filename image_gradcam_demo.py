import sys
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
from tf_keras_vis.utils.scores import CategoricalScore

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Tạo folder
os.makedirs("images/crops", exist_ok=True)

# UTILS (giữ nguyên)
def load_image(image_path, grayscale=False, target_size=None):
    if grayscale:
        pil_image = image.load_img(image_path, color_mode='grayscale', target_size=target_size)
    else:
        pil_image = image.load_img(image_path, target_size=target_size)
    return image.img_to_array(pil_image)

def load_detection_model(model_path):
    detection_model = cv2.CascadeClassifier(model_path)
    return detection_model

def detect_faces(detection_model, gray_image_array):
    return detection_model.detectMultiScale(
        gray_image_array, 
        scaleFactor=1.1,   
        minNeighbors=3,    
        minSize=(30, 30)   
    )

def draw_bounding_box(face_coordinates, image_array, color):
    x, y, w, h = face_coordinates
    cv2.rectangle(image_array, (x, y), (x + w, y + h), color, 2)

def apply_offsets(face_coordinates, offsets):
    x, y, width, height = face_coordinates
    x_off, y_off = offsets
    return (x - x_off, x + width + x_off, y - y_off, y + height + y_off)

def draw_text(coordinates, image_array, text, color, x_offset=0, y_offset=0,
              font_scale=2, thickness=2):
    x, y = coordinates[:2]
    cv2.putText(image_array, text, (x + x_offset, y + y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, color, thickness, cv2.LINE_AA)

def preprocess_input(x, v2=True):
    x = x.astype('float32')
    x = x / 255.0
    if v2:
        x = x - 0.5
        x = x * 2.0
    return x

# LẤY PATH ẢNH
if len(sys.argv) < 2:
    print("❌ Thiếu path ảnh!")
    sys.exit()

image_path = sys.argv[1]
print("📸 Loading:", image_path)

# LOAD MODELS
detection_model_path = "model/detection_model/haarcascade_frontalface_default.xml"
gender_model_path = "./model/gender_model/simple_CNN.81-0.96.hdf5"
emotion_model_path = "model/emotion_model/fer2013_mini_XCEPTION.102-0.66.hdf5"

face_detection = load_detection_model(detection_model_path)
gender_classifier = load_model(gender_model_path, compile=False)
emotion_classifier = load_model(emotion_model_path, compile=False)

print("✅ Models loaded!")

gender_labels = ("Female", "Male")
emotion_labels = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')

gender_target_size = gender_classifier.input_shape[1:3]
emotion_target_size = emotion_classifier.input_shape[1:3]

# GradCAM layer detect
def get_conv_layer(model):
    for layer in reversed(model.layers):
        if any(k in layer.name.lower() for k in ['conv', 'separable', 'depthwise']):
            idx = -model.layers.index(layer) - 1
            print(f"Using layer idx {idx}: {layer.name}")
            return idx
    print("Warning: Default to -1")
    return -1

gender_layer_idx = get_conv_layer(gender_classifier)
emotion_layer_idx = get_conv_layer(emotion_classifier)

gradcam_gender = Gradcam(gender_classifier, model_modifier=ReplaceToLinear(), clone=True)
gradcam_emotion = Gradcam(emotion_classifier, model_modifier=ReplaceToLinear(), clone=True)

# LOAD IMAGE
rgb_image = load_image(image_path, grayscale=False)
gray_image = load_image(image_path, grayscale=True)
gray_image = np.squeeze(gray_image).astype('uint8')

img = cv2.cvtColor(rgb_image.astype('uint8'), cv2.COLOR_RGB2BGR)
orig = img.copy()

faces = detect_faces(face_detection, gray_image)
if len(faces) == 0:
    print("❌ Không tìm thấy mặt nào!")
    sys.exit()

print(f"🔍 Detected {len(faces)} faces")

# THRESHOLD
FEMALE_THRESHOLD = 0.3
MALE_THRESHOLD = 0.6

for i, face_coordinates in enumerate(faces):
    x, y, w, h = face_coordinates
    if w < 40 or h < 40:
        continue

    # DYNAMIC OFFSETS
    gender_ox = min(50, w // 3)
    gender_oy = min(50, h // 3)
    gender_offsets = (gender_ox, gender_oy)

    emotion_offsets = (0, 0)

    gx1, gx2, gy1, gy2 = apply_offsets(face_coordinates, gender_offsets)
    gx1, gx2 = max(0, gx1), min(rgb_image.shape[1], gx2)
    gy1, gy2 = max(0, gy1), min(rgb_image.shape[0], gy2)
    rgb_face_crop = rgb_image[gy1:gy2, gx1:gx2]

    ex1, ex2, ey1, ey2 = apply_offsets(face_coordinates, emotion_offsets)
    ex1, ex2 = max(0, ex1), min(gray_image.shape[1], ex2)
    ey1, ey2 = max(0, ey1), min(gray_image.shape[0], ey2)
    gray_face_crop = gray_image[ey1:ey2, ex1:ex2]

    try:
        if rgb_face_crop.size == 0 or gray_face_crop.size == 0:
            continue

        # SAVE CROPS
        rgb_crop_bgr = cv2.cvtColor(rgb_face_crop.astype('uint8'), cv2.COLOR_RGB2BGR)
        cv2.imwrite(f"images/crops/face{i+1}_gender_crop.png", rgb_crop_bgr)
        gray_crop_resized = cv2.resize(gray_face_crop.astype('uint8'), (emotion_target_size[1], emotion_target_size[0]))
        cv2.imwrite(f"images/crops/face{i+1}_emotion_crop.png", gray_crop_resized)

        # RESIZE
        rgb_face = cv2.resize(rgb_face_crop.astype('uint8'), (gender_target_size[1], gender_target_size[0]))
        gray_face = cv2.resize(gray_face_crop.astype('uint8'), (emotion_target_size[1], emotion_target_size[0]))

        # GENDER PREDICT + FLIP
        rgb_face_flipped = cv2.flip(rgb_face, 1)
        rgb_face_orig = preprocess_input(rgb_face, v2=False)
        rgb_face_orig = np.expand_dims(rgb_face_orig, 0)
        gender_pred_orig = gender_classifier.predict(rgb_face_orig)
        female_orig = gender_pred_orig[0][0]

        rgb_face_flip = preprocess_input(rgb_face_flipped, v2=False)
        rgb_face_flip = np.expand_dims(rgb_face_flip, 0)
        gender_pred_flip = gender_classifier.predict(rgb_face_flip)
        female_flip = gender_pred_flip[0][0]

        avg_female = (female_orig + female_flip) / 2
        avg_male = 1 - avg_female

        if avg_female > FEMALE_THRESHOLD:
            gender_text = "Female"
            gender_class = 0
        elif avg_male > MALE_THRESHOLD:
            gender_text = "Male"
            gender_class = 1
        else:
            gender_text = "Unknown"
            gender_class = 0

        # EMOTION PREDICT
        gray_face_input = preprocess_input(gray_face, v2=True)
        gray_face_input = np.expand_dims(gray_face_input, 0)
        gray_face_input = np.expand_dims(gray_face_input, -1)
        emotion_pred = emotion_classifier.predict(gray_face_input)[0]
        emotion_idx = np.argmax(emotion_pred)
        emotion_text = emotion_labels[emotion_idx]

        # DRAW BBOX + LABEL
        color = (0, 255, 255) if gender_text == "Unknown" else ((0, 0, 255) if gender_text == "Female" else (255, 0, 0))
        draw_bounding_box(face_coordinates, img, color)
        draw_text(face_coordinates, img, str(i+1), color, 0, 20, 0.7, 1)
        draw_text(face_coordinates, img, gender_text, color, 0, -20, 0.7, 1)
        draw_text(face_coordinates, img, emotion_text, color, 0, -50, 0.7, 1)

        # GRAD-CAM GENDER
        gender_input = rgb_face_orig
        gender_score = CategoricalScore(gender_class)
        gender_cam = gradcam_gender(gender_score, gender_input, penultimate_layer=gender_layer_idx)
        gender_heatmap = gender_cam[0]
        print(f"Face {i+1} Gender heatmap shape: {gender_heatmap.shape}, min={np.min(gender_heatmap):.2f}, max={np.max(gender_heatmap):.2f}")

        gender_heatmap = cv2.resize(gender_heatmap, (w, h))
        gender_heatmap = np.maximum(gender_heatmap, 0)
        h_max = np.max(gender_heatmap)
        if h_max > 0:
            gender_heatmap = (gender_heatmap / h_max) * 255
        else:
            print("Warning: Gender heatmap max=0, skipping overlay")
            gender_heatmap = np.zeros((h, w), dtype=np.uint8)
        gender_heatmap = np.uint8(gender_heatmap)
        cv2.imwrite(f"images/crops/face{i+1}_gender_heatmap_raw.png", gender_heatmap)
        gender_heatmap_colored = cv2.applyColorMap(gender_heatmap, cv2.COLORMAP_JET)
        gender_overlay = cv2.addWeighted(orig[y:y+h, x:x+w], 0.85, gender_heatmap_colored, 0.15, 0)
        cv2.imwrite(f"images/crops/face{i+1}_gender_gradcam.png", gender_overlay)

        # GRAD-CAM EMOTION (FIX: Squeeze an toàn cho 3D/4D)
        emotion_input = gray_face_input
        emotion_score = CategoricalScore(emotion_idx)
        emotion_cam = gradcam_emotion(emotion_score, emotion_input, penultimate_layer=emotion_layer_idx)
        # FIX: Squeeze để lấy (H,W) từ bất kỳ shape nào (1,H,W) hoặc (1,H,W,1)
        emotion_heatmap = np.squeeze(emotion_cam[0])
        print(f"Face {i+1} Emotion heatmap shape after squeeze: {emotion_heatmap.shape}, min={np.min(emotion_heatmap):.2f}, max={np.max(emotion_heatmap):.2f}")

        emotion_heatmap = cv2.resize(emotion_heatmap, (w, h))
        emotion_heatmap = np.maximum(emotion_heatmap, 0)
        e_max = np.max(emotion_heatmap)
        if e_max > 0:
            emotion_heatmap = (emotion_heatmap / e_max) * 255
        else:
            print("Warning: Emotion heatmap max=0, skipping overlay")
            emotion_heatmap = np.zeros((h, w), dtype=np.uint8)
        emotion_heatmap = np.uint8(emotion_heatmap)
        cv2.imwrite(f"images/crops/face{i+1}_emotion_heatmap_raw.png", emotion_heatmap)
        emotion_heatmap_colored = cv2.applyColorMap(emotion_heatmap, cv2.COLORMAP_JET)
        emotion_overlay = cv2.addWeighted(orig[y:y+h, x:x+w], 0.85, emotion_heatmap_colored, 0.15, 0)
        cv2.imwrite(f"images/crops/face{i+1}_emotion_gradcam.png", emotion_overlay)

        # BLEND nhẹ lên full img
        blended = cv2.addWeighted(img[y:y+h, x:x+w], 0.7, gender_heatmap_colored, 0.15, 0)
        img[y:y+h, x:x+w] = cv2.addWeighted(blended, 0.85, emotion_heatmap_colored, 0.15, 0)

        print(f"Face {i+1}: Gender {gender_text} | Emotion {emotion_text} | GradCAM ready!")

    except Exception as e:
        print(f"Error face {i+1}: {e}")
        continue

# SAVE + SHOW
cv2.imwrite("images/gradcam_output.jpg", img)
print("🔥 Saved: gradcam_output.jpg | Raw heatmaps in crops/*_heatmap_raw.png (check values >0?)")
cv2.imshow("GradCAM Result", img)
cv2.waitKey(0)
cv2.destroyAllWindows()