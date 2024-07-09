import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image
from hand_pose import extract_hand_landmarks
from preprocess import preprocess_landmarks
from model import build_model

# Load custom Odia font
odia_font_path = "Anek_Odia/AnekOdia-VariableFont_wdth,wght.ttf"
odia_font = ImageFont.truetype(odia_font_path, 32)

# Load the model
model = build_model()
# Assume the model is pre-trained and load the weights if available
# model.load_weights('path_to_model_weights.h5')

# Mapping digits to Odia numerals
odia_numerals = ['୦', '୧', '୨', '୩', '୪', '୫', '୬', '୭', '୮', '୯']

# Predict digit
def predict_digit(landmarks, model):
    landmarks = preprocess_landmarks(landmarks)
    landmarks = np.expand_dims(landmarks, axis=0)
    prediction = model.predict([landmarks, landmarks])
    digit = np.argmax(prediction)
    return digit

# Real-time prediction
cap = cv2.VideoCapture(0)
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    landmarks, output_image = extract_hand_landmarks(frame)
    if landmarks:
        digit = predict_digit(landmarks, model)
        odia_digit = odia_numerals[digit]

        # Convert frame to PIL image
        pil_image = Image.fromarray(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_image)
        draw.text((50, 50), odia_digit, font=odia_font, fill=(255, 0, 0))

        # Convert back to OpenCV image
        output_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    cv2.imshow('Air-Writing Recognition', output_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
