import streamlit as st
from ultralytics import YOLO
import cv2
from ultralytics.yolo.v8.detect.predict import DetectionPredictor
import time
from playsound import playsound
from camera_input_live import camera_input_live

model = YOLO('../Models/final.pt')
# src = 'download.jpg'
ptime = time.process_time_ns()


def predict():
    global probs
    results = model.predict(source="0", show=True)
    for result in results:
        probs = result.probs
        if (float(probs.data[1]) >= 0.50) and (ptime >= 2000000000):
            st.text('Slippery surface detected')
            # playsound('/Models/beep-01a (1).wav')


def main():
    # title
    st.title('See Ya Sense')
    # button
    st.button('Check', on_click=predict, type='primary')


if __name__ == '__main__':
    main()
