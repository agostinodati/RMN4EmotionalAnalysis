import cv2
from shutil import copyfile
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


def ck_preprocessor(pathDataset = 'D:/CKdataset/cohn-kanade-images/', pathLabelDir = 'D:/CKdataset/Emotion_labels/Emotion/', newPath = 'C:/CKpreprocessed/'):
    '''
    Create a new dataset starting from CK. In this new dataset will be stored only the sequences of images
    that have a emotion label. Sequences labelled with "contempt" will be ignored by the classifier.
    :param pathDataset: Path of CK dataset with all subdirs
    :param pathLabelDir: Path of emotion label's dir
    :param newPath:  Path where new dataset will be created
    '''
    if os.path.isdir(newPath) is not True:
        os.mkdir(newPath)

    ckEmotionDict = {1: 'angry', 2: 'contempt', 3: 'disgust', 4: 'fear', 5: 'happy', 6: 'sad', 7: 'surprise'}

    missingLabel = 0
    totalElements = 0
    contemptCount = 0
    countDict = {'angry': 0, 'contempt': 0, 'disgust': 0, 'fear': 0, 'happy': 0, 'sad': 0, 'surprise': 0}

    listDir = os.listdir(pathDataset)
    for dir in listDir:
        pathDir = pathDataset + dir + '/'
        if os.path.isdir(pathDir):
            listSubDir = os.listdir(pathDir)
            for subdir in listSubDir:
                pathSubDir = pathDir + subdir + '/'
                pathLabel = pathLabelDir + dir + '/' + subdir + '/'
                if os.path.isdir(pathLabel):
                    print(os.listdir(pathLabel))
                    listFileName = os.listdir(pathLabel)
                    totalElements = totalElements + 1
                    if listFileName.__len__() == 0:
                        # Not all the expressions are labelled
                        print('Skipped this becouse the label is missing')
                        print(pathLabel)
                        missingLabel = missingLabel + 1
                        break
                    labelFileName = listFileName[0]
                    emotion_label_filepath = pathLabel + labelFileName
                    print(emotion_label_filepath)
                    with open(emotion_label_filepath, 'r') as f:
                        label = f.readline()
                        label = int(label.split('.')[0])
                        if label == 2:
                            print('Contempt found')
                            contemptCount = contemptCount + 1
                        labelName = ckEmotionDict[label]
                        print(label)

                    # Count all the label for the log
                    countDict[labelName] = countDict[labelName] + 1

                    # Create a new dir passing the label in the title and keeping the names of the original dir and subdir
                    newDir = labelName + '_' + dir + '_' + subdir
                    newDirPath = newPath + newDir + '/'
                    if os.path.isdir(newDirPath) is not True:
                        os.mkdir(newDirPath)

                if os.path.isdir(pathSubDir) and subdir != '.DS_Store':
                    listFrames = os.listdir(pathSubDir)
                    for frame in listFrames:
                        # Copy the images in the dir into the path created above
                        copyfile(pathSubDir + frame, newDirPath + frame)
                        #print(pathSubDir + frame)
    datasetLogHead = 'angry,contempt,disgust,fear,happy,sad,surprise,missing,total\n'
    datasetLog = str(countDict['angry']) + ',' + str(countDict['contempt']) + ',' + str(countDict['disgust']) + ',' + str(countDict['fear']) + ',' + str(countDict['happy']) + ',' + str(countDict['sad']) + ',' + str(countDict['surprise']) + ',' + str(missingLabel) + ',' + str(totalElements)

    csvLogDataset = newPath + 'logDataset.csv'
    with open(csvLogDataset, 'w') as f:
        f.write(datasetLogHead)
        f.write(datasetLog)

    print('Total: ' + str(totalElements))
    print('Missing: ' + str(missingLabel))
    print('Contempt: ' + str(contemptCount))


if __name__ == '__main__':
    ck_preprocessor()
