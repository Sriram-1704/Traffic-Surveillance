import cv2
import numpy as np

classesFile = "Helmet/Models/obj.names";
classes = None
frame_count = 0 
frame_count_out=0  

confThreshold = 0.5  
nmsThreshold = 0.4   

def getOutputsNames(net):
    layersNames = net.getLayerNames()
    return [layersNames[int(i) - 1] for i in net.getUnconnectedOutLayers()]

modelConfiguration = "Helmet/Models/yolov5-obj.cfg";
modelWeights = "Helmet/Models/yolov5-obj_2400.weights";

net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

def drawPred(classId, conf, left, top, right, bottom,frame,option):
    global frame_count
    #cv.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 3)
    label = '%.2f' % conf
    if classes:
        assert(classId < len(classes))
        label = '%s:%s' % (classes[classId], label)
    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    label_name,label_conf = label.split(':')
    print(label_name+" === "+str(conf)+"==  "+str(option))
    if label_name == 'Helmet' and conf > 0.30:
        cv2.rectangle(frame, (left, top - round(1.5*labelSize[1])), (left + round(1.5*labelSize[0]), top + baseLine), (255, 255, 255), cv2.FILLED)
        cv2.putText(frame, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 1)
            

def postprocess(frame, outs, option):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    global frame_count_out
    frame_count_out=0
    classIds = []
    confidences = []
    boxes = []
    classIds = []
    confidences = []
    boxes = []
    cc = 0
    for out in outs:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                #print(classIds)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    indices = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    count_person=0 # for counting the classes in this loop.
    for i in indices:
        #i = i[0]
        box = boxes[int(i)]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        frame_count_out = drawPred(classIds[int(i)], confidences[int(i)], left, top, left + width, top + height,frame,option)
        my_class='Helmet'      
        unknown_class = classes[classId]
        print("===="+str(unknown_class))
        if my_class == unknown_class:
            count_person += 1
            
    print(str(frame_count_out))
    if count_person == 0 and option == 1:
        cv2.putText(frame, "Helmet Not detected", (10, 50), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,0), 2)
    return frame

def getHelmet(frame):
    blob = cv2.dnn.blobFromImage(frame, 1/255, (416, 416), [0,0,0], 1, crop=False)
    net.setInput(blob)
    outs = net.forward(getOutputsNames(net))
    return postprocess(frame, outs,0)
