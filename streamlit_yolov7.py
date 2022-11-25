import singleinference_yolov7
from singleinference_yolov7 import SingleInference_YOLOV7
import os
import streamlit as st
import logging
import requests
from PIL import Image
from io import BytesIO
import numpy as np
import cv2
class Streamlit_YOLOV7(SingleInference_YOLOV7):
    '''
    streamlit app that uses yolov7
    '''
    def __init__(self,):
        self.logging_main=logging
        self.logging_main.basicConfig(level=self.logging_main.DEBUG)

    def new_yolo_model(self,img_size,path_yolov7_weights,path_img_i,device_i='cpu'):
        '''
        SimpleInference_YOLOV7
        created by Steven Smiley 2022/11/24

        INPUTS:
        VARIABLES                    TYPE    DESCRIPTION
        1. img_size,                    #int#   #this is the yolov7 model size, should be square so 640 for a square 640x640 model etc.
        2. path_yolov7_weights,         #str#   #this is the path to your yolov7 weights 
        3. path_img_i,                  #str#   #path to a single .jpg image for inference (NOT REQUIRED, can load cv2matrix with self.load_cv2mat())

        OUTPUT:
        VARIABLES                    TYPE    DESCRIPTION
        1. predicted_bboxes_PascalVOC   #list#  #list of values for detections containing the following (name,x0,y0,x1,y1,score)

        CREDIT
        Please see https://github.com/WongKinYiu/yolov7.git for Yolov7 resources (i.e. utils/models)
        @article{wang2022yolov7,
            title={{YOLOv7}: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors},
            author={Wang, Chien-Yao and Bochkovskiy, Alexey and Liao, Hong-Yuan Mark},
            journal={arXiv preprint arXiv:2207.02696},
            year={2022}
            }
        
        '''
        super().__init__(img_size,path_yolov7_weights,path_img_i,device_i=device_i)
    def main(self):
        st.title('Custom YoloV7 Object Detector')
        st.subheader(""" Upload an image and run YoloV7 on it.  
        This model was trained to detect the following classes from a drone's vantage point. 
        Notice where the model fails.
        (i.e. objects too close up & too far away):\n""")
        st.markdown(
            """
        <style>
        .reportview-container .markdown-text-container {
            font-family: monospace;
        }
        .sidebar .sidebar-content {
            background-image: linear-gradient(#2e7bcf,#2e7bcf);
            color: black;
        }
        .Widget>label {
            color: green;
            font-family: monospace;
        }
        [class^="st-b"]  {
            color: green;
            font-family: monospace;
        }
        .st-bb {
            background-color: black;
        }
        .st-at {
            background-color: green;
        }
        footer {
            font-family: monospace;
        }
        .reportview-container .main footer, .reportview-container .main footer a {
            color: black;
        }
        header .decoration {
            background-image: None);
        }


        </style>
        """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <style>
            .reportview-container {
                background: url("https://raw.githubusercontent.com/stevensmiley1989/STREAMLIT_YOLOV7/main/misc/IMG_0512_reduced.JPG")
            }
        .sidebar .sidebar-content {
                background: url("https://raw.githubusercontent.com/stevensmiley1989/STREAMLIT_YOLOV7/main/misc/IMG_0512_reduced.JPG")
            }
            </style>
            """,
            unsafe_allow_html=True
        )
        text_i_list=[]
        for i,name_i in enumerate(self.names):
            #text_i_list.append(f'id={i} \t \t name={name_i}\n')
            text_i_list.append(f'{i}: {name_i}\n')
        st.selectbox('Classes',tuple(text_i_list))
        self.conf_selection=st.selectbox('Confidence Threshold',tuple([0.1,0.25,0.5,0.75,0.95]))
        
        self.response=requests.get(self.path_img_i)

        self.img_screen=Image.open(BytesIO(self.response.content))

        st.image(self.img_screen, caption=self.capt, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
        st.markdown('YoloV7 on streamlit.  Demo of object detection with YoloV7 with a web application.')
        self.im0=np.array(self.img_screen.convert('RGB'))
        self.load_image_st()
        predictions = st.button('Predict on the image?')
        if predictions:
            self.predict()
            predictions=False

    def load_image_st(self):
        uploaded_img=st.file_uploader(label='Upload an image')
        if type(uploaded_img) != type(None):
            self.img_data=uploaded_img.getvalue()
            st.image(self.img_data)
            self.im0=Image.open(BytesIO(self.img_data))#.convert('RGB')
            self.im0=np.array(self.im0)

            return self.im0
        elif type(self.im0) !=type(None):
            return self.im0
        else:
            return None
    
    def predict(self):
        self.conf_thres=self.conf_selection
        st.write('Loading image')
        self.load_cv2mat(self.im0)
        st.write('Making inference')
        self.inference()

        self.img_screen=Image.fromarray(self.image).convert('RGB')
        
        self.capt='DETECTED:'
        if len(self.predicted_bboxes_PascalVOC)>0:
            for item in self.predicted_bboxes_PascalVOC:
                name=str(item[0])
                conf=str(round(100*item[-1],2))
                self.capt=self.capt+ ' name='+name+' confidence='+conf+'%, '
        st.image(self.img_screen, caption=self.capt, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
        self.image=None
    

if __name__=='__main__':
    app=Streamlit_YOLOV7()

    #INPUTS for YOLOV7
    img_size=640
    path_yolov7_weights="weights/best.pt"
    path_img_i="https://raw.githubusercontent.com/stevensmiley1989/STREAMLIT_YOLOV7/main/test_images/DJI_0028_fps24_frame00000040.jpg"
    #INPUTS for webapp
    app.capt="Initial Image"
    app.new_yolo_model(img_size,path_yolov7_weights,path_img_i)
    app.conf_thres=0.65
    app.load_model() #Load the yolov7 model
    
    app.main()

