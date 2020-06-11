# USAGE
# python detect_mask_video.py

# import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import os, sys

import onnx
import vision.utils.box_utils_numpy as box_utils
from caffe2.python.onnx import backend

# onnx runtime
import onnxruntime as ort

def onnx_predict(width, height, confidences, boxes, prob_threshold, iou_threshold=0.3, top_k=-1):
    boxes = boxes[0]
    confidences = confidences[0]
    picked_box_probs = []
    picked_labels = []
    for class_index in range(1, confidences.shape[1]):
        probs = confidences[:, class_index]
        mask = probs > prob_threshold
        probs = probs[mask]
        if probs.shape[0] == 0:
            continue
        subset_boxes = boxes[mask, :]
        box_probs = np.concatenate([subset_boxes, probs.reshape(-1, 1)], axis=1)
        box_probs = box_utils.hard_nms(box_probs,
                                       iou_threshold=iou_threshold,
                                       top_k=top_k,
                                       )
        picked_box_probs.append(box_probs)
        picked_labels.extend([class_index] * box_probs.shape[0])
    if not picked_box_probs:
        return np.array([]), np.array([]), np.array([])
    picked_box_probs = np.concatenate(picked_box_probs)
    picked_box_probs[:, 0] *= width
    picked_box_probs[:, 1] *= height
    picked_box_probs[:, 2] *= width
    picked_box_probs[:, 3] *= height
    return picked_box_probs[:, :4].astype(np.int32), np.array(picked_labels), picked_box_probs[:, 4]

def detect_and_predict_mask(frame, faceNet, maskNet):
    # grab the dimensions of the frame and then construct a blob
    # from it
    
    # 1A. Normal FD
    """
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
        (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the face detections
    faceNet.setInput(blob)

    detections = faceNet.forward()

    # initialize our list of faces, their corresponding locations,
    # and the list of predictions from our face mask network
    faces = []
    locs = []
    preds = []

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the detection
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the confidence is
        # greater than the minimum confidence
        if confidence > args["confidence"]:
            # compute the (x, y)-coordinates of the bounding box for
            # the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # ensure the bounding boxes fall within the dimensions of
            # the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # extract the face ROI, convert it from BGR to RGB channel
            # ordering, resize it to 224x224, and preprocess it
            width, height = endX-startX, endY-startY
            x1, x2 = int(startX*0.9), int(endX*1.1)
            y1, y2 = int(startY*0.9), int(endY*1.1)
            face = frame[y1:y2, x1:x2]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
#             face = np.expand_dims(face, axis=0)

            # add the face and bounding boxes to their respective lists
            faces.append(face)
            locs.append((x1, y1, x2, y2))

    """

    ## 1B. Ultra Fast ONNX FD
    orig_frame = frame.copy()
    (h, w) = orig_frame.shape[:2]
    frame = cv2.resize(frame, (640, 480))
    image_mean = np.array([127, 127, 127])
    frame = (frame - image_mean) / 128
    frame = np.transpose(frame, [2, 0, 1])
    frame = np.expand_dims(frame, axis=0)
    frame = frame.astype(np.float32)

    confidences, boxes = ort_session.run(None, {input_name: frame})
    boxes, labels, probs = onnx_predict(orig_frame.shape[1], orig_frame.shape[0], confidences, boxes, threshold)

    print(labels)
    preds, locs = [], []
    for box, lab, prob in zip(boxes, labels, probs):
        # only make a predictions if at least one face was detected
        (startX, startY, endX, endY) = box[:]
        # ensure the bounding boxes fall within the dimensions of
        # the frame
        (startX, startY) = (max(0, startX), max(0, startY))
        (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

        # extract the face ROI, convert it from BGR to RGB channel
        # ordering, resize it to 224x224, and preprocess it
        width, height = endX-startX, endY-startY
        x1, x2 = int(startX*0.95), int(endX*1.05)
        y1, y2 = int(startY*0.95), int(endY*1.05)
        face = orig_frame[y1:y2, x1:x2]
        locs.append((x1, y1, x2, y2))
        print("FACE:", face.shape, lab, box, prob)
        # face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        # face = cv2.resize(face, (224, 224))
        # face = img_to_array(face)
        # face = preprocess_input(face)
#             face = np.expand_dims(face, axis=0)

        # add the face and bounding boxes to their respective lists
        faceBlob = cv2.dnn.blobFromImage(face, 1.0, (224, 224),
            (0,0,0))

        # pass the blob through the network and obtain the face detections
        maskNet.setInput(faceBlob)
        isMask = maskNet.forward()[0][0]
        if isMask >=0: preds.append(True)
        else: preds.append(False)


    # return a 2-tuple of the face locations and their corresponding
    # locations
    return (locs, preds)

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", type=str,
    default="face_detector",
    help="path to face detector model directory")
ap.add_argument("-m", "--model", type=str,
    default="mask_detector.model",
    help="path to trained face mask detector model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
    help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

label_path = "models/voc-model-labels.txt"
threshold = 0.6
onnx_path = "models/onnx/version-RFB-640.onnx"
class_names = [name.strip() for name in open(label_path).readlines()]

predictor = onnx.load(onnx_path)
onnx.checker.check_model(predictor)
onnx.helper.printable_graph(predictor.graph)
predictor = backend.prepare(predictor, device="CPU")  # default CPU

ort_session = ort.InferenceSession(onnx_path)
input_name = ort_session.get_inputs()[0].name

"""
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"],
    "res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
"""

# load the face mask detector model from disk
print("[INFO] loading face mask detector model...")
maskNet = cv2.dnn.readNet("face_detector/deploy1.prototxt", "face_detector/face_mask.caffemodel")

# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
#vs = VideoStream(src="rtsp://admin:admin123@202.83.161.162:554/main_stream").start()
vs = VideoStream(src=0).start()
# vs = cv2.VideoCapture("rtsp://admin:admin123@202.83.161.162:554/main_stream")
time.sleep(2.0)

writer = None

# loop over the frames from the video stream
while True:
    # grab the frame from the threaded video stream and resize it
    # to have a maximum width of 400 pixels
    frame = vs.read()
    frame = imutils.resize(frame, width=800)

    if writer == None:
        fps = 20
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        H, W = frame.shape[:2]
        writer = cv2.VideoWriter('output.mp4', fourcc, 20,
            (W, H), True)

    # detect faces in the frame and determine if they are wearing a
    # face mask or not
    (locs, preds) = detect_and_predict_mask(frame, None, maskNet)

    # loop over the detected face locations and their corresponding
    # locations
    for (box, pred) in zip(locs, preds):
        # unpack the bounding box and predictions
        (startX, startY, endX, endY) = box

        # determine the class label and color we'll use to draw
        # the bounding box and text
        label = "Mask" if pred else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

        # include the probability in the label
#         label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

        # display the label and bounding box rectangle on the output
        # frame
        cv2.putText(frame, label, (startX, startY - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    writer.write(frame)
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# do a bit of cleanup
cv2.destroyAllWindows()
writer.release()
vs.stop()
