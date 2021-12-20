# # externalLibraries.py
# Import PyTorch Library
import torch
from torch import nn

# Import external libraries
import argparse
import numpy as np
import opencv_wrapper as cvw
from skimage.filters import threshold_local
import json
import random
from string import ascii_uppercase, digits, punctuation
import colorama
import regex


# # PreProcessing.py
"""
get_info() function reads the image using openCV and performs thresholding, dilation, noise removal, and
contouring to finally retrieve bounding boxes from the contour.
Below are some of the additional available functions from openCV for preprocessing:
Median filter: median filter blurs out noises by taking the medium from a set of pixels
cv2.medianBlur()
Dilation and erosion: dilation adds pixels to boundaries of pixels, erosion removes it
cv2.dilate()
cv2.erode()
cv2.opening() #This is an erosion followed by a dilation
"""

def get_info(path):
    font     = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.5
    fontColor  = (255,0,0)
    lineType = 1

    #Threshold
    image = cv2.imread(path)

    height,width,channel = image.shape
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    T = threshold_local(gray, 15, offset = 6, method = "gaussian") # generic, mean, median, gaussian
    thresh = (gray > T).astype("uint8") * 255
    thresh = ~thresh

    #Dilation
    kernel =np.ones((1,1), np.uint8)
    ero = cv2.erode(thresh, kernel, iterations= 1)
    img_dilation = cv2.dilate(ero, kernel, iterations=1)

    # Remove noise
    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(img_dilation, None, None, None, 8, cv2.CV_32S)
    sizes = stats[1:, -1] #get CC_STAT_AREA component
    final = np.zeros((labels.shape), np.uint8)
    for i in range(0, nlabels - 1):
        if sizes[i] >= 10:   #filter small dotted regions
            final[labels == i + 1] = 255

    #Find contours
    kern = np.ones((5,15), np.uint8)
    img_dilation = cv2.dilate(final, kern, iterations = 1)
    contours, hierarchy = cv2.findContours(img_dilation, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    # Map contours to bounding rectangles, using bounding_rect property
    rects = map(lambda c: cv2.boundingRect(c), contours)
    # Sort rects by top-left x (rect.x == rect.tl.x)
    sorted_rects = sorted(rects, key =lambda r: r[0])
    sorted_rects = sorted(sorted_rects, key =lambda r: r[1])

    etfo=''
    for rect in sorted_rects:
        x,y,w,h = rect
        if(w<20 or h<20):
            continue
        temp = image[y:y+h, x:x+w]
        temp = cv2.cvtColor(temp, cv2.COLOR_BGR2RGB)
        hi = pytesseract.image_to_data(temp, config=r'--psm 6')
        hi = hi.split()
        ind = 22
        while(True):
            if (ind>len(hi)):
                break
            if(int(hi[ind])==-1):
                ind+=11
            else:
                etfo=etfo+hi[ind+1]
                etfo=etfo+" "
                x+=len(hi[ind+1])*20
                ind+=12
        etfo=etfo+'\n'
    return etfo
	

# # LSTM.py
"""
The following LSTM is the model to use after the OCR extraction, where it predicts the key-value pairs after the texts
are extracted using the previous get_info() method.
"""
class ExtractLSTM(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers=2, bidirectional=True)
        self.linear = nn.Linear(hidden_size * 2, 5)

    def forward(self, inpt):
        embedded = self.embed(inpt)
        feature, _ = self.lstm(embedded)
        oupt = self.linear(feature)
        return oupt
		
		
# # pipeline.py
VOCAB= ascii_uppercase+digits+punctuation+" \t\n"

#Change to CUDA to run using GPU
device = 'cpu'

def get_test_data(etfo):  
    text = etfo
    text_tensor = torch.zeros(len(text), 1, dtype=torch.long)
    text_tensor[:, 0] = torch.LongTensor([VOCAB.find(c) for c in text])
    return text_tensor.to(device)

etfo = get_info('invoice.png')
# etfo = get_info('X51005621482.jpeg')
etfo = etfo.upper()
text_tensor = get_test_data(etfo)
temp = []
for i in range(len(text_tensor)):
    if text_tensor[i]>=0 and text_tensor[i]<=70:
        temp.append([int(text_tensor[i])])
text_tensor = torch.LongTensor(temp)

#model initialization
hidden_size = 256
device= torch.device('cpu')
model = ExtractLSTM(len(VOCAB), 16, hidden_size).to(device)
model.load_state_dict(torch.load('model.pth'))
result = test(model)
print(result)