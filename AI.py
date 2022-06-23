# -*- coding: utf-8 -*-
"""
@created 2022년 06월 20일 월요일
@author 허진경, 홍영복 
"""
import os
import time
import tensorflow as tf
from absl.flags import FLAGS
import core.utils as utils
from tensorflow.python.saved_model import tag_constants 
from core.config import cfg
import cv2
import numpy as np
from tensorflow.compat.v1 import ConfigProto
from shapely.geometry import LineString
from tools import preprocessing
from mqtt.Camerapublisher import ImageMqttPublisher 
from absl import app, flags


os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# comment out below line to enable tensorflow logging outputs 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
physical_devices = tf.config.experimental.list_physical_devices('GPU') 

if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True) 

def mask_polylines(cap, camera_name, bus):
    pts=[] #점찍은좌표를저장할리스트
# setMouseCallback 함수

    def draw_circle(event, x, y, flags, param):
    
        if event == cv2.EVENT_LBUTTONDBLCLK: 
            ix, iy = x, y
            cv2.circle(img, (ix, iy), 3, (0, 255, 0), -1)
            pts.append([ix, iy])
            
        cv2.namedWindow("image")
        cv2.setMouseCallback("image", draw_circle)
        ret, img = cap.read()
        h,w,c = img.shape
        cv2.rectangle(img, (185, 130), (400 + 100 * int(bus), 230), 
                      (0, 0, 0), -1)
        cv2.putText(img, f'{"Bus" * int(bus)} Roi', (200, 200),
                    cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 255), 5)

        if h > 1000:
            img = cv2.resize(img, (int(w / 2), int(h / 2)))
        while True:
            cv2.imshow("image", img) # 동영상 이미지를 보여주는 화면
            # cv2.imshow("bg_mask",bg_mask) 
            #마스크로 저장할 이미지를 보여주는 화면 k = cv2.waitKey(1) & 0xFF
            if k == ord("s"):
                pts = np.array(pts, dtype=np.int32)
                cv2.polylines(img, [pts], True, (0, 0, 255), 2) 
            elif k == 27:
                break 
    cv2.destroyAllWindows()
    if h>1000:
        pts *= 2

# numpy 파일로 저장
    if bus:
        np.save("./pts/"+"bus_"+camera_name,pts)
    else:
        np.save("./pts/" + camera_name, pts)
    return pts
def borderline(pts): 
    border = []
    for i in range(len(pts) - 1): 
        border.append(LineString([pts[i], pts[i + 1]]))
    border.append(LineString([pts[-1], pts[0]])) 
    return border

def point_in_border(point, pts):
    border = borderline(pts)
    com_line = LineString([(0, point[1]), (point[0], point[1])]) 
    count = 0

    for line in border:
        if com_line.intersection(line):
            count += 1
    if count % 2 == 0:
        return False
    else:
        return True
# -------------------------------------------------------------------------- 
flags.DEFINE_string('video', './video/F20009_1_202011260930.avi', 'path to input folder') 
flags.DEFINE_string('camera', '165.032.105.25', 'camera ip address') 
flags.DEFINE_float('iou', 0.1, 'iou threshold')
flags.DEFINE_float('score', 0.01, 'score threshold')
flags.DEFINE_boolean('dont_show', True, 'show video outputww')
flags.DEFINE_string('IP', '165.132.105.25', 'server ip address')
# rtsp://reputer:reputer@rtsp://reputer01.iptimecam.com:20001/stream_ch00_1 # 재배기 1(정면)
def main(_argv): 
    print("start")
    s_t = time.time()
    video_path = FLAGS.video
    # load configuration for object detector
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS) 
    input_size = 608
    # otherwise load standard tensorflow saved model
    saved_model_loaded = tf.saved_model.load('./checkpoints/yolov4-608',
                                             tags=[tag_constants.SERVING])
    infer = saved_model_loaded.signatures['serving_default']
    video_name = FLAGS.camera # 165.132.105.25
    camera_name = FLAGS.camera # 165.132.105.25 person 111.202.15.53, car F18003_3, person F20009_1 # begin video capture
    vid = cv2.VideoCapture("rtsp://reputer:reputer@reputer01.iptimecam.com:20001/stream_ch00_1") 
    try:
        pts = np.load("./pts/" + camera_name + ".npy") 
    except:
        mask_polylines(vid, camera_name, False)
        pts = np.load("./pts/" + camera_name + ".npy")
    print("car video")
    
    nms_max_overlap = 0.95
    color = {'person': (131, 224, 112), 'bicycle': (51, 221, 255),
             'motorbike': (61, 61, 245)}
    allowed_classes = color.keys()
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)) 
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)) 
    vid_fps = int(vid.get(cv2.CAP_PROP_FPS))
    codec = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(video_path[:-len(v)] + "detection" + f'/{video_name}_detection.avi',
                          (width, height))
    frame_cnt = 0 print(video_path)
    imageMqttPusblisher = ImageMqttPublisher(FLAGS.IP, 1883, "/camerapub")
    imageMqttPusblisher.connect()
# while video is running
    while True:
            start_time = time.time() ret, frame = vid.read() 
        if not ret:
            print('Video has ended or failed, try a different video format!')
            break
        # save frame_roi
        frame_roi = frame.copy()
        cv2.polylines(frame_roi, [pts], True,
                      (0, 0, 255), 2)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_data = cv2.resize(frame,
                                (input_size, input_size)) # ,interpolation = cv2.INTER_AREA image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        # run detections on tflite if flag is set
        batch_data = tf.constant(image_data)
        pred_bbox = infer(batch_data)
        for key, value in pred_bbox.items():
            boxes = value[:, :, 0:4]
            pred_conf = value[:, :, 4:]
            boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
                boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
                scores=tf.reshape(
                    pred_conf, (tf.shape(pred_conf)[0], -1,
                                tf.shape(pred_conf)[-1])),
                                max_output_size_per_class=50,
                                max_total_size=50,
                                iou_threshold=FLAGS.iou,
                                score_threshold=FLAGS.score )
# convert data to numpy arrays and slice out unused elements 
        num_objects = valid_detections.numpy()[0]
        bboxes = boxes.numpy()[0]
        bboxes = bboxes[0:int(num_objects)]
        scores = scores.numpy()[0]
        scores = scores[0:int(num_objects)]
        classes = classes.numpy()[0]
        classes = classes[0:int(num_objects)]
# format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, width, height 
        original_h, original_w, _ = frame.shape
        bboxes = utils.format_boxes(bboxes, original_h, original_w)
        mw = bboxes[:, 2].mean()
        #read in all class names from config
        class_names = utils.read_class_names(cfg.YOLO.CLASSES)
        # loop through objects and use class index to get class name, allow only classes in
allowed_classes list names = []
        deleted_indx = []
        for i in range(num_objects):
            class_indx = int(classes[i])
            class_name = class_names[class_indx] 
            w = bboxes[i][2] - bboxes[i][0]
            h = bboxes[i][3] - bboxes[i][1]
            if class_name not in allowed_classes:
                deleted_indx.append(i)
            elif (w > 300) or (h > 300):
                deleted_indx.append(i) else:
                    names.append(class_name) 
        names = np.array(names)
        # delete detections that are not in allowed_classes
        bboxes = np.delete(bboxes, deleted_indx, axis=0)
        scores = np.delete(scores, deleted_indx, axis=0)
        classes = np.delete(classes, deleted_indx, axis=0)
        indices = preprocessing.non_max_suppression(bboxes, classes, nms_max_overlap, scores) 
        if frame_cnt % 3 == 0:
            count = {cn:0 for cn in color.keys()} 
        for ind in indices:
            bbox = bboxes[ind] 
            class_name = names[ind]
            x, y, w, h = bbox            
            if (w > 300) or (h > 300):
                continue
            
            center = (x + int(w / 2), y + int(h / 2)) 
            if point_in_border(center, pts):
                if FLAGS.person == False:
                    if point_in_border(center,bpts):
                        if w > mw:
                            class_name = "bus"
                # draw bbox on screen
                cv2.rectangle(frame, (int(x), int(y)), (int(x) + int(w),
                                                        int(y) + int(h)), color[class_name], 1)
                cv2.putText(frame, class_name, (int(bbox[0]), int(bbox[1]) + 10),
                            0, 0.5, (0, 0, 0), 2)
                if frame_cnt%3 == 0:
                    count[class_name] += 1
        text_height = 50
        cv2.rectangle(frame, (50, 0), (450, 20 + len(color.keys())*50),
                      (0, 0, 0), -1)
        for cn in color.keys():
            cv2.putText(frame, f'{cn} : {count[cn]}', (60, text_height),
                        cv2.FONT_HERSHEY_COMPLEX, 1.5, color[cn], 2)
            text_height += 50
        result = np.asarray(frame)
        result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        # out.write(result)
        result = cv2.resize(result, None, fx=0.5, fy= 0.5) 
        imageMqttPusblisher.sendBase64(result)
        # calculate frames per second of running detections
        if not FLAGS.dont_show: 
            if height>720:
                result = cv2.resize(result, (int(width/2),
                                             int(height/2)))
            cv2.imshow("Output Video", result)
        # if output flag is set, save video file 
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break_flag = 0 
            # out.release()
            break
        fps = 1.0 / (time.time() - start_time)
            # print("{0} {1} : {2:.2f}fps".format(FLAGS.camera, frame_cnt, fps), end='\r') 
        frame_cnt += 1
    imageMqttPusblisher.disconnect() 
        # xml 생성
    vid.release()
    # out.release()
    cv2.destroyAllWindows() 
    print(time.time() - s_t) 
    cv2.destroyAllWindows()
if __name__ == '__main__':
    try:
        app.run(main) 
    except SystemExit:
        pass
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    