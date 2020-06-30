import os
import glob
import numpy as np
import cv2

directory=os.getcwd()
directory=directory+"/faces_extracted"
list_directory=[x[0] for x in os.walk(directory)]
del list_directory[0]

X=list()
Y=list()

for directory in list_directory:
    filenames=glob.glob(directory+"/*")
    label=os.path.basename(directory)
    images=[cv2.imread(filename) for filename in filenames]
    for image in images:
        image=image.astype('float32')
        X.append(image)
        Y.append(label)

Z=zip(Y,X)
np.savez_compressed('data.npz',Z)




