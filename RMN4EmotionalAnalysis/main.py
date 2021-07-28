import cv2
import numpy as np
from rmn import RMN
from os import listdir
from os.path import isfile, join

m = RMN()
mypath = 'examples/'
frames = [f for f in listdir(mypath) if isfile(join(mypath, f))]
count = 0
for frame in frames:
    print(frame)
    image = cv2.imread(mypath + frame)
    results = m.detect_emotion_for_single_frame(image)
    print(results)
    image = m.draw(image, results)
    cv2.imwrite('examples/results/output_' + str(count) + '.png', image)
    count += 1
