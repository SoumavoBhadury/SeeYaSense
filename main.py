import streamlit as st
from ultralytics import YOLO
from ultralytics.yolo.v8.detect.predict import DetectionPredictor


model = YOLO('../Models/final.pt')
def predict():
    results = model.predict(source="0", show=True, conf=0.5)
    print(results)
    #print(type(results))





def main():

    #title
    st.title('See Ya Sense')
    #button
    st.button('Check ', on_click=predict())
    predict()

if __name__ == '__main__':
    main()
