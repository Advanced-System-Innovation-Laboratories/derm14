# -*- coding: utf-8 -*-

import numpy as np
from tqdm import tqdm
import os
import cv2
from skimage.transform import resize
#%%
Dir = "FILE DESTINATION"

#%%
def get_data(Dir):
    X = []
    y = []
    counter = 0
    for nextDir in os.listdir(Dir):
        if not nextDir.startswith('.'):
            if nextDir.startswith('ak'):
                label = 0
            elif nextDir.startswith('df'):
                label = 1 
            elif nextDir.startswith('sk'):
                label = 2
            elif nextDir.startswith('vasc'):
                label = 3
            elif nextDir.startswith('bcc'):
                label = 4
            elif nextDir.startswith('bw'):
                label = 5
            elif nextDir.startswith('scc'):
                label = 6
            elif nextDir.startswith('bnv'):
                label = 7
            elif nextDir.startswith('knv'):
                label = 8
            elif nextDir.startswith('lnv'):
                label = 9
            elif nextDir.startswith('nv'):
                label = 10
            elif nextDir.startswith('snv'):
                label = 11
            elif nextDir.startswith('lm'):
                label = 12
            elif nextDir.startswith('mel'):
                label = 13
            temp = Dir + nextDir

            for file in tqdm(os.listdir(temp)):
                img = cv2.imread(temp + '/' + file)
                if img is not None:
                    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                    img = resize(img, (X, Y, Z)) #Change X, Y, Z parameters according to your deep learning model.
                    #img_file = scipy.misc.imresize(arr=img_file, size=(150, 150, 3))
                    img = np.asarray(img)
                    X.append(img)
                    y.append(label)
                    counter += 1

    X = np.asarray(X)
    y = np.asarray(y)
    return X, y
#%%
X, y = get_data(Dir)
print("Your np arrays has been successfully generated.")
#%%
np.save("SAVING DIRECTORY_X.npy", X)
np.save("SAVING DIRECTORY_Y.npy", y)
print("Your np arrays has been successfully saved.")
#%%
