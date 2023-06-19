import cv2
import streamlit as st
import numpy as np
import tempfile
import mediapipe as mp
import tensorflow.keras
import tensorflow as tf
from gtts import gTTS
from playsound import playsound
import os
import time

model = keras.models.load_model('modelWeights.h5')
actions = np.array([
    'a', 'b', 'c',
    '0', '1', '2', '3',
    'hello', 'goodbye', 'please', 'thank you',
    'yes', 'no', 'help',
    'red', 'blue', 'green',
    'mother', 'father',
    'eat', 'drink',
    'welcome', 'to', 'our', 'graduation', 'project', 'discussion',
    'idle'
])


def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results


mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

colors = [(245, 117, 16), (117, 245, 16), (16, 117, 245),
          (100, 255, 150), (255, 255, 23), (0, 0, 0), (255, 255, 255)]
sequence = []
sentence = []
predictions = []
threshold = 0.8


def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten(
    ) if results.pose_landmarks else np.zeros(33 * 4)  # to avoid errors
    # flatten will put all landmarks into 1D array
    righthand = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten(
    ) if results.right_hand_landmarks else np.zeros(21 * 3)  # to avoid errors
    lefthand = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten(
    ) if results.left_hand_landmarks else np.zeros(21 * 3)  # to avoid errors
    return np.concatenate([pose, righthand, lefthand])


def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0, 60 + num * 40), (int(prob * 100),
                                                         90 + num * 40), colors[num % actions.size], -1)
        cv2.putText(output_frame, actions[num % actions.size], (
            0, 85 + num * 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    return output_frame


def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(
                                  color=(80, 22, 10), thickness=1, circle_radius=1),
                              mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=1, circle_radius=1))
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(
                                  color=(121, 22, 76), thickness=1, circle_radius=1),
                              mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=1, circle_radius=1))
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(
                                  color=(245, 117, 66), thickness=1, circle_radius=1),
                              mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=1, circle_radius=1))


cap = cv2.VideoCapture(0)
st.title("Sign Language Recogniser")
frame_placeholder = st.empty()
stop = st.button("Stop")
st.write('Select the options you want')
landMarkBool = st.checkbox('Draw landmarks')
probViz = st.checkbox('prob viz')
ttsBool = st.checkbox('TTS')

sequence = []
sentence = []
predictions = []
threshold = 0.5
# cap = cv2.VideoCapture(0)
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        ##################################
        # Make detections
        image, results = mediapipe_detection(frame, holistic)
        if landMarkBool:
            draw_landmarks(image, results)

        # 2. Prediction logic
        if len(sequence) == 0:
            cv2.putText(image, 'TAKING NEXT WORD', (120, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
            # Show to screen
            frame1 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame1, channels="RGB")
            # cv2.imshow('OpenCV Feed', image)
            cv2.waitKey(2000)
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-45:]
        if len(sequence) == 45:
            print("predicting")
            res = model.predict(np.expand_dims(
                sequence, axis=0), verbose=None)[0]
            if ttsBool:
                mytext = actions[np.argmax(res)]
                language = 'en'
                myobj = gTTS(text=mytext, lang=language, slow=False)
                Address = mytext + ".mp3"
                myobj.save(Address)
                time.sleep(1)
                playsound(Address)
            sequence = []
            print("---->", actions[np.argmax(res)])
            predictions.append(np.argmax(res))
            sentence.append(actions[np.argmax(res)])
            # 3. Viz logic

            if len(sentence) > 5:
                sentence = sentence[-5:]

        cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
        cv2.putText(image, ' '.join(sentence), (3, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        # Viz probabilities
        if probViz:
            image = prob_viz(res, actions, image, colors)

        cv2.imshow('OpenCV Feed', image)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        if not ret:
            st.write("Video Capture has ended")
            break

        frame1 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame1, channels="RGB")
        # cv2.imshow('OpenCV Feed', image)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()