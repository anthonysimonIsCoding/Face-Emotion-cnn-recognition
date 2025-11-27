import sys
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Tạo folder cho crops nếu chưa có
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

# LOAD MODELS
detection_model_path = "model/detection_model/haarcascade_frontalface_default.xml"
gender_model_path = "./model/gender_model/simple_CNN.81-0.96.hdf5"
emotion_model_path = "model/emotion_model/fer2013_mini_XCEPTION.102-0.66.hdf5"

face_detection = load_detection_model(detection_model_path)
gender_classifier = load_model(gender_model_path, compile=False)
emotion_classifier = load_model(emotion_model_path, compile=False)

gender_labels = ("Female", "Male")
emotion_labels = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')

gender_target_size = gender_classifier.input_shape[1:3]
emotion_target_size = emotion_classifier.input_shape[1:3]

# LOAD IMAGE
image_path = sys.argv[1]
rgb_image = load_image(image_path, grayscale=False)
gray_image = load_image(image_path, grayscale=True)
gray_image = np.squeeze(gray_image).astype('uint8')

faces = detect_faces(face_detection, gray_image)
print(f"🔍 Detected {len(faces)} faces total")

# THRESHOLD
FEMALE_THRESHOLD = 0.3
MALE_THRESHOLD = 0.6

for i, face_coordinates in enumerate(faces):
    x, y, w, h = face_coordinates
    if w < 40 or h < 40:
        print(f"Face {i+1}: Too small, skipping")
        continue
    print(f"Face {i+1} position: x={x}, y={y}, w={w}, h={h}")

    # DYNAMIC OFFSETS (mới: dựa face size, tránh dính)
    gender_ox = min(50, w // 3)  # ~1/3 width, max 50
    gender_oy = min(50, h // 3)  # ~1/3 height
    gender_offsets = (gender_ox, gender_oy)
    print(f"Face {i+1}: Dynamic offsets = {gender_offsets} (ko dính nữa)")

    emotion_offsets = (0, 0)  # Giữ nguyên

    x1, x2, y1, y2 = apply_offsets(face_coordinates, gender_offsets)
    x1, x2 = max(0, x1), min(rgb_image.shape[1], x2)
    y1, y2 = max(0, y1), min(rgb_image.shape[0], y2)
    rgb_face_crop = rgb_image[y1:y2, x1:x2]

    x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
    x1, x2 = max(0, x1), min(gray_image.shape[1], x2)
    y1, y2 = max(0, y1), min(gray_image.shape[0], y2)
    gray_face_crop = gray_image[y1:y2, x1:x2]

    try:
        if rgb_face_crop.size == 0 or gray_face_crop.size == 0:
            print(f"Face {i+1}: Crop empty, skipping")
            continue

        # SAVE CROP (sạch hơn giờ)
        rgb_crop_bgr = cv2.cvtColor(rgb_face_crop.astype('uint8'), cv2.COLOR_RGB2BGR)
        cv2.imwrite(f"images/crops/face{i+1}_gender_crop.png", rgb_crop_bgr)
        gray_crop = cv2.resize(gray_face_crop.astype('uint8'), (48, 48))
        cv2.imwrite(f"images/crops/face{i+1}_emotion_crop.png", gray_crop)
        print(f"Face {i+1}: Saved clean crops!")

        rgb_face = rgb_face_crop.astype('uint8')
        gray_face = gray_face_crop.astype('uint8')

        gender_resize = (gender_target_size[1], gender_target_size[0])
        emotion_resize = (emotion_target_size[1], emotion_target_size[0])
        rgb_face = cv2.resize(rgb_face, gender_resize)
        gray_face = cv2.resize(gray_face, emotion_resize)

        # FLIP + AVG (giữ để thử)
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
        print(f"Face {i+1} - Orig F={female_orig:.2f}, Flipped F={female_flip:.2f}, Avg F={avg_female:.2f}")

        if avg_female > FEMALE_THRESHOLD:
            gender_text = "Female"
        elif avg_male > MALE_THRESHOLD:
            gender_text = "Male"
        else:
            gender_text = "Unknown"

        print(f"Face {i+1} - Final: {gender_text}")

        # Emotion
        gray_face = preprocess_input(gray_face, v2=True)
        gray_face = np.expand_dims(gray_face, 0)
        gray_face = np.expand_dims(gray_face, -1)
        emotion_prediction = emotion_classifier.predict(gray_face)
        emotion_label_arg = np.argmax(emotion_prediction)
        emotion_text = emotion_labels[emotion_label_arg]
        print(f"Face {i+1} - Emotion: {emotion_text}")

        # Color
        if gender_text == "Unknown":
            color = (0, 255, 255)
        elif gender_text == "Female":
            color = (0, 0, 255)
        else:
            color = (255, 0, 0)

        # Draw
        draw_bounding_box(face_coordinates, rgb_image, color)
        draw_text(face_coordinates, rgb_image, str(i+1), color, 0, 20, 1, 2)
        draw_text(face_coordinates, rgb_image, gender_text, color, 0, -20, 1, 2)
        draw_text(face_coordinates, rgb_image, emotion_text, color, 0, -50, 1, 2)

    except Exception as e:
        print(f"Error processing face {i+1}: {e}")
        continue

# Save
bgr_image = cv2.cvtColor(rgb_image.astype('uint8'), cv2.COLOR_RGB2BGR)
cv2.imwrite("images/fixed_predict.png", bgr_image)
print("🔥 Saved: images/fixed_predict.png | Crops sạch ở images/crops/!")
cv2.imshow("Result", bgr_image)
cv2.waitKey(0)