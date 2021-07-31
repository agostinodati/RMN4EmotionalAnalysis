import cv2
import numpy as np
from rmn import RMN
import os
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
    emo_voted = max(emo_dict, key=emo_dict.get) # NB: se ci sono più elementi max, prende il primo che incontra. Loggare quando vi sono più max eventualmente
    votes = emo_dict[emo_voted]
    print('Emotion voted is ' + emo_voted + ' with '+str(votes) + ' votes')


def main():
    '''
    Per ogni sottocartella X di "cohn-kanade-images", devo aprire ogni sotto cartella Y e votare le immagini contenute.
    Fatto ciò, verifico il risultato andando a prendere corrispendente dentro la sottocartella Y dentro "Emotion_labels".
    Quindi conviene avere due liste con tutte le sottocartelle del dataset. Oppure modificare i nomi delle sotto-sottocartelle
    effettuando un po' di preprocessing
    '''
    pathDataset = 'D:/CKdataset/cohn-kanade-images/'
    pathLabelDir = 'D:/CKdataset/Emotion_labels/Emotion/'
    listDir = os.listdir(pathDataset)
    for dir in listDir:
        pathDir = pathDataset + dir + '/'
        if os.path.isdir(pathDir):
            listSubDir = os.listdir(pathDir)
            for subdir in listSubDir:
                pathSubDir = pathDir + subdir + '/'
                if os.path.isdir(pathSubDir) and subdir != '.DS_Store':
                    listFrames = os.listdir(pathSubDir)
                    for frame in listFrames:
                        print(pathSubDir + frame)
                pathLabel = pathLabelDir + dir + '/' + subdir + '/'
                if os.path.isdir(pathLabel):
                    emotion_label_filepath = pathLabel + os.listdir(pathLabel)[0]
                    print(emotion_label_filepath)
                    with open(pathLabel, 'r') as f:
                        label = f.readline()



if __name__ == '__main__':
    main()
