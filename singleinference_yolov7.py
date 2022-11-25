
import random
import numpy as np
import os
import sys
import torch
import cv2
import logging
#

class SingleInference_YOLOV7:
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
    def __init__(self,
    img_size, path_yolov7_weights, 
    path_img_i='None',
    device_i='0',
    conf_thres=0.25,
    iou_thres=0.5):

        self.conf_thres=conf_thres
        self.iou_thres=iou_thres
        self.clicked=False
        self.img_size=img_size
        #self.path_yolov7=path_yolov7
        self.path_yolov7_weights=path_yolov7_weights
        self.path_img_i=path_img_i
        #sys.path.append(self.path_yolov7)
        from utils.general import check_img_size, non_max_suppression, scale_coords
        from utils.torch_utils import select_device
        from models.experimental import attempt_load
        self.scale_coords=scale_coords
        self.non_max_suppression=non_max_suppression
        self.select_device=select_device
        self.attempt_load=attempt_load
        self.check_img_size=check_img_size

        #Initialize
        self.predicted_bboxes_PascalVOC=[]
        self.im0=None
        self.im=None
        self.device = self.select_device(device_i) #gpu 0,1,2,3 etc or '' if cpu
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA
        self.logging=logging
        #if os.path.exists('logs')==False:
        #    os.makedirs('logs')
        #self.logging.basicConfig(filename='logs/'+str(self.__class__.__name__)+'.log',filemode='w',format='%(name)s - %(levelname)s - %(message)s',level=self.logging.ERROR)
        self.logging.basicConfig(level=self.logging.DEBUG)



    def load_model(self):
        '''
        Loads the yolov7 model

        self.path_yolov7_weights = r"/example_path/my_model/best.pt"
        self.device = '0' for gpu cuda 0, '' for cpu

        '''
        # Load model
        self.model = self.attempt_load(self.path_yolov7_weights, map_location=self.device)  # load FP32 model
        self.stride = int(self.model.stride.max())  # model stride
        self.img_size = self.check_img_size(self.img_size, s=self.stride)  # check img_size
        if self.half:
            self.model.half() # to FP16

        # Get names and colors
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]

        # Run inference
        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, self.img_size, self.img_size).to(self.device).type_as(next(self.model.parameters())))  # run once

    def read_img(self,path_img_i):
        '''
        Reads a single path to a .jpg file with OpenCV

        path_img_i = r"/example_path/img_example_i.jpg"

        '''
        #Read path_img_i
        if type(path_img_i)==type('string'):
            if os.path.exists(path_img_i):
                self.path_img_i=path_img_i
                self.im0=cv2.imread(self.path_img_i)
                print('self.im0.shape',self.im0.shape)
                #self.im0=cv2.resize(self.im0,(self.img_size,self.img_size))
            else:
                log_i=f'read_img \t Bad path for path_img_i:\n {path_img_i}'
                self.logging.error(log_i)
        else:
            log_i=f'read_img \t Bad type for path_img_i\n {path_img_i}'
            self.logging.error(log_i)


    def load_cv2mat(self,im0=None):
        '''
        Loads an OpenCV matrix
        
        im0 = cv2.imread(self.path_img_i)

        '''
        if type(im0)!=type(None):
            self.im0=im0
        if type(self.im0)!=type(None):
            self.img=self.im0.copy()    
            self.imn = cv2.cvtColor(self.im0, cv2.COLOR_BGR2RGB)
            self.img=self.imn.copy()
            self.image = self.img.copy()
            self.image, self.ratio, self.dwdh = self.letterbox(self.image,auto=False)
            self.image_letter=self.image.copy()
            self.image = self.image.transpose((2, 0, 1))

            self.image = np.expand_dims(self.image, 0)
            self.image = np.ascontiguousarray(self.image)
            self.im = self.image.astype(np.float32)
            self.im = torch.from_numpy(self.im).to(self.device)
            self.im = self.im.half() if self.half else self.im.float()  # uint8 to fp16/32
            self.im /= 255.0  # 0 - 255 to 0.0 - 1.0
            if self.im.ndimension() == 3:
                self.im = self.im.unsqueeze(0)
        else:
            log_i=f'load_cv2mat \t Bad self.im0\n {self.im0}'
            self.logging.error(log_i)


    def inference(self):
        '''
        Inferences with the yolov7 model, given a valid input image (self.im)
        '''
        # Inference
        if type(self.im)!=type(None):
            self.outputs = self.model(self.im, augment=False)[0]
            # Apply NMS
            self.outputs = self.non_max_suppression(self.outputs, self.conf_thres, self.iou_thres, classes=None, agnostic=False)
            img_i=self.im0.copy()
            self.ori_images = [img_i]
            self.predicted_bboxes_PascalVOC=[]
            for i,det in enumerate(self.outputs):
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    #det[:, :4] = self.scale_coords(self.im.shape[2:], det[:, :4], self.im0.shape).round()
                    #Visualizing bounding box prediction.
                    batch_id=i
                    self.image = self.ori_images[int(batch_id)]
                    print(self.image.shape)
                    for j,(*bboxes,score,cls_id) in enumerate(reversed(det)):
                        x0=float(bboxes[0].cpu().detach().numpy())
                        y0=float(bboxes[1].cpu().detach().numpy())
                        x1=float(bboxes[2].cpu().detach().numpy())
                        y1=float(bboxes[3].cpu().detach().numpy())
                        self.box = np.array([x0,y0,x1,y1])
                        self.box -= np.array(self.dwdh*2)
                        self.box /= self.ratio
                        self.box = self.box.round().astype(np.int32).tolist()
                        cls_id = int(cls_id)
                        score = round(float(score),3)
                        name = self.names[cls_id]
                        self.predicted_bboxes_PascalVOC.append([name,x0,y0,x1,y1,score]) #PascalVOC annotations
                        color = self.colors[self.names.index(name)]
                        name += ' '+str(score)
                        cv2.rectangle(self.image,self.box[:2],self.box[2:],color,2)
                        cv2.putText(self.image,name,(self.box[0], self.box[1] - 2),cv2.FONT_HERSHEY_SIMPLEX,0.75,[225, 255, 255],thickness=2)
        else:
            log_i=f'Bad type for self.im\n {self.im}'
            self.logging.error(log_i)

    def show(self):
        '''
        Displays the detections if any are present
        '''
        if len(self.predicted_bboxes_PascalVOC)>0:
            self.TITLE='Press any key or click mouse to quit'
            cv2.namedWindow(self.TITLE)
            cv2.setMouseCallback(self.TITLE,self.onMouse)
            while cv2.waitKey(1) == -1 and not self.clicked:
                #print(self.image.shape)
                cv2.imshow(self.TITLE, self.image)
            cv2.destroyAllWindows()
            self.clicked=False
        else:
            log_i=f'Nothing detected for {self.path_img_i} \n \t w/ conf_thres={self.conf_thres} & iou_thres={self.iou_thres}'
            self.logging.debug(log_i)

    def letterbox(self,im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleup=True, stride=32):
        '''
        Formats the image in letterbox format for yolov7
        '''
        # Resize and pad image while meeting stride-multiple constraints
        shape = im.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return im, r, (dw, dh)
    def onMouse(self,event,x,y,flags,param):
        '''
        Handles closing example window
        '''
        if event==cv2.EVENT_LBUTTONUP:
            self.clicked=True

if __name__=='__main__':  

    #INPUTS
    img_size=640
    path_yolov7_weights="weights/best.pt"
    path_img_i=r"test_images/DJI_0028_fps24_frame00000040.jpg"

    #INITIALIZE THE app
    app=SingleInference_YOLOV7(img_size,path_yolov7_weights,path_img_i,device_i='0',conf_thres=0.25,iou_thres=0.5)

    #LOAD & INFERENCE
    app.load_model() #Load the yolov7 model
    app.read_img(path_img_i) #read in the jpg image from the full path, note not required if you want to load a cv2matrix instead directly
    app.load_cv2mat() #load the OpenCV matrix, note could directly feed a cv2matrix here as app.load_cv2mat(cv2matrix)
    app.inference() #make single inference
    app.show() #show results
    print(f'''
    app.predicted_bboxes_PascalVOC\n
    \t name,x0,y0,x1,y1,score\n
    {app.predicted_bboxes_PascalVOC}''') 








