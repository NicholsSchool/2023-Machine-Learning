from cscore import CameraServer
from ntcore import NetworkTableInstance

import cv2
import json
import numpy as np
import time

# Python code to run the classifier:
import re
import os
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter
from pycoral.adapters import common
from pycoral.adapters import classify

print("Connecting to Network Tables")
inst = ntcore.NetworkTableInstance.getDefault()
table = inst.getTable("pieces")

piece = table.getStringTopic("piece").publish()
xMin = table.getIntegerTopic("xMin").publish()
yMin = table.getIntegerTopic("yMin").publish()
xMax = table.getIntegerTopic("xMax").publish()
yMax = table.getIntegerTopic("yMax").publish()

inst.startClient4("example client")
inst.setServerTeam(4930)
inst.startDSClient()

# the TFLite converted to be used with edgetpu
modelPath = "Models/detect_edgetpu.tflite"
# The path to labels.txt that was downloaded with your model
labelPath = "Models/labelmap.pbtxt"

min_conf_threshold = 0.5

team = 4930
server = False

# This function takes in a TFLite Interpter and Image, and returns classifications
def classifyImage(interpreter, image):
    size = common.input_size(interpreter)
    common.set_input(interpreter, cv2.resize(image, size, fx=0, fy=0,
                                             interpolation=cv2.INTER_CUBIC))
    interpreter.invoke()

    return classify.get_classes(interpreter)


def main():
    with open('/boot/frc.json') as f:
        config = json.load(f)
    camera = config['cameras'][0]

    width = camera['width']
    height = camera['height']


    CameraServer.startAutomaticCapture()

    input_stream = CameraServer.getVideo()
    output_stream = CameraServer.putVideo('Processed', width, height)
    img = np.zeros(shape=(height, width, 3), dtype=np.float32)

    # Table for vision output information
    # start NetworkTables
    ntinst = NetworkTableInstance.getDefault()
    if server:
        print("Setting up NetworkTables server")
        ntinst.startServer()
    else:
        print("Setting up NetworkTables client for team {}".format(team))
        ntinst.startClient4("wpilibpi")
        #ntinst.setServerTeam(team)
        ntinst.startDSClient()
    #vision_nt = NetworkTables.getTable('Vision')
    vision_nt = ntinst.getTable('Vision')

    # Load your model onto the TF Lite Interpreter
    interpreter = make_interpreter(modelPath)
    interpreter.allocate_tensors()
    labels = read_label_file(labelPath)

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]

    print(width)
    print(height)

    floating_model = (input_details[0]['dtype'] == np.float32)

    input_mean = 127.5
    input_std = 127.5

    outname = output_details[0]['name']

    if ('StatefulPartitionedCall' in outname): # This is a TF2 model
        boxes_idx, classes_idx, scores_idx = 1, 3, 0
    else: # This is a TF1 model
        boxes_idx, classes_idx, scores_idx = 0, 1, 2



    # Wait for NetworkTables to start
    time.sleep(0.5)

    prev_time = time.time()

    while True:
        start_time = time.time()

        frame_time, input_img = input_stream.grabFrame(img)

        output_img = np.copy(input_img)
        frame_rgb = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (width, height))
        input_data = np.expand_dims(frame_resized, axis=0)

        # Notify output of error and skip iteration
        if frame_time == 0:
            output_stream.notifyError(input_stream.getError())
            continue

        if floating_model:
            input_data = (np.float32(input_data) - input_mean) / input_std

        interpreter.set_tensor(input_details[0]['index'],input_data)
        interpreter.invoke()

        # Retrieve detection results
        boxes = interpreter.get_tensor(output_details[boxes_idx]['index'])[0] # Bounding box coordinates of detected objects
        classes = interpreter.get_tensor(output_details[classes_idx]['index'])[0] # Class index of detected objects
        scores = interpreter.get_tensor(output_details[scores_idx]['index'])[0] # Confidence of detected objects

        imH = 120
        imW = 160


        for i in range(len(scores)):
            if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):

                # Get bounding box coordinates and draw box
                # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
                ymin = int(max(1,(boxes[i][0] * imH)))
                xmin = int(max(1,(boxes[i][1] * imW)))
                ymax = int(min(imH,(boxes[i][2] * imH)))
                xmax = int(min(imW,(boxes[i][3] * imW)))
                
                cv2.rectangle(output_img, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)

                # Draw label
                object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
                

                print('piece: ' + str( object_name ) )
                print('ymin: ' + str( ymin ))
                print('xmin: ' + str(xmin))
                print('ymax: ' + str(ymax))
                print('xmax: ' + str(xmax))

                piece.set(object_name)
                xMin.set(xmin)
                yMin.set(ymin)
                xMax.set(xmax)
                yMax.set(ymax)
    


        # Draw framerate in corner of frame

        processing_time = start_time - prev_time
        prev_time = start_time

        fps = 1 / processing_time
        cv2.putText(output_img, str(round(fps, 1)), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
        output_stream.putFrame(output_img)

main()