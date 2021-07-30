import cv2
import numpy as np
from rmn import RMN
from os import listdir
from os.path import isfile, join


def main_test():
    m = RMN()
    mypath = 'examples/'
    frames = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    count = 0
    emo_dict = {'angry': 0, 'disgust': 0, 'fear': 0, 'happy': 0, 'sad': 0, 'surprise': 0, 'neutral': 0}
    for frame in frames:
        print(frame)
        image = cv2.imread(mypath + frame)
        results = m.detect_emotion_for_single_frame(image)
        proba_list = results[0]['proba_list']
        emo_label = results[0]['emo_label']
        emo_dict[emo_label] = emo_dict[emo_label] + 1
        print(results)
        image = m.draw(image, results)
        cv2.imwrite('examples/results/output_' + str(count) + '.png', image)
        count += 1
    emo_voted = max(emo_dict, key=emo_dict.get)
    votes = max(emo_dict.values())
    print('Emotion voted is ' + emo_voted + ' with '+str(votes) + ' votes')


if __name__ == '__main__':
    main_test()
