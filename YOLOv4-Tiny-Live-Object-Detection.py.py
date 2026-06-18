#!/usr/bin/env python
# coding: utf-8

# In[55]:


import cv2 
import numpy as np
import matplotlib.pyplot as plt
import sys
get_ipython().run_line_magic('matplotlib', 'inline')


# In[56]:


objectnessThreshold=0.5
confThreshold=0.5
nmsThreshold=0.4
inpWidth=416
inpHeight=416

classesFile=r"D:\Opencv oct-Nov\15_V4\coco.names"
def getOutputsNames(net):
    layersNames = net.getLayerNames()
    return [layersNames[i - 1]
            for i in net.getUnconnectedOutLayers()]
modelConfiguration =R"D:\Opencv oct-Nov\15_V4\yolov4-tiny.cfg"
modelWeights =R"D:\Opencv oct-Nov\15_V4\yolov4-tiny.weights"
print(classes)


# In[57]:


net=cv2.dnn.readNetFromDarknet(modelConfiguration,modelWeights)


# In[58]:


def getOutputsNames(net):
    layersNames=net.getLayerNames()
    return[layersNames[i-1] for i in net.getUnconnectedOutLayers()]


# In[59]:


FONTFACE = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.7
THICKNESS = 1


def display_objects(frame, outs):
    """Remove the bounding boxes with low confidence using non-maxima suppression."""
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    # Scan through all the bounding boxes output from the network and keep only the
    # ones with high confidence scores. Assign the box's class label as the class with the highest score.
    classIds = []
    confidences = []
    boxes = []
    
    # Loop through all outputs.
    for out in outs:
        for detection in out:
            if detection[4] > objectnessThreshold:
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
                    confidences.append(float(confidence))
                    boxes.append([left, top, width, height])

    # Perform non maximum suppression to eliminate redundant overlapping boxes with
    # lower confidences.
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
  
    for i in indices:
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        cv2.rectangle(frame, (left, top), (left + width, top + height), (255, 255, 255), 2)
        label = "{}:{:.2f}".format(classes[classIds[i]], confidences[i])
        display_text(frame, label, left, top)


# In[60]:


def display_text(im, text, x, y):
    """Draw text onto image at location."""
    
    # Get text size 
    textSize = cv2.getTextSize(text, FONTFACE, FONT_SCALE, THICKNESS)
    dim = textSize[0]
    baseline = textSize[1]
            
    # Use text size to create a black rectangle. 
    cv2.rectangle(im, (x,y), (x + dim[0], y + dim[1] + baseline), (0,0,0), cv2.FILLED);
    # Display text inside the rectangle.
    cv2.putText(im, text, (x, y + dim[1]), FONTFACE, FONT_SCALE, (0, 255, 255), THICKNESS, cv2.LINE_AA)


# In[62]:


ig=r"D:\Opencv oct-Nov\15_V4\traffic.jpg"
frame=cv2.imread(ig)
blob=cv2.dnn.blobFromImage(frame,1/255,(inpWidth,inpHeight),[0,0,0], 1, crop=False)
net.setInput(blob)
outs=net.forward(getOutputsNames(net))
display_objects(frame, outs)

# Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
t, _ = net.getPerfProfile()
label = 'Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency())
cv2.putText(frame, label, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1, cv2.LINE_AA)
plt.figure(figsize=[10,10])
plt.imshow(frame[..., ::-1])
plt.show()
print(label)


# In[ ]:





# In[ ]:




