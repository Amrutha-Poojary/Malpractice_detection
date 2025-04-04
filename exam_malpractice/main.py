import cv2 
import numpy as np
from tensorflow.keras.models import load_model


model_path = './exam_malpractice_model.h5'
model = load_model(model_path)  


img_height, img_width = 32, 32


class_names = {0: "Non-Suspicious", 1: "Suspicious"}


cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break


    resized_frame = cv2.resize(frame, (img_height, img_width))
    preprocessed_frame = np.expand_dims(resized_frame / 255.0, axis=0)

    prediction = model.predict(preprocessed_frame)
    predicted_class = (prediction[0] > 0.5).astype(int)[0]
    predicted_label = class_names.get(predicted_class, "Unknown")

    cv2.putText(
        frame,
        f"Prediction: {predicted_label}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0) if predicted_class == 1 else (0, 0, 255),
        2,
        cv2.LINE_AA
    )

    # Show the frame
    cv2.imshow("Webcam - Suspicious Activity Detection", frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1000 // 30) & 0xFF == ord('q'):  # Adjust frame rate (e.g., 30 FPS)
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
