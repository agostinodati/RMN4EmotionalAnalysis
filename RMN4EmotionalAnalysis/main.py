import cv2
from shutil import copyfile
from rmn import RMN
import os
from os import listdir
from os.path import isfile, join
import csv
import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix
import time


def ck_preprocessor(pathDataset='D:/CKdataset/cohn-kanade-images/', pathLabelDir='D:/CKdataset/Emotion_labels/Emotion/',
                    newPath='C:/CKpreprocessed/'):
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
                        # print(pathSubDir + frame)
    datasetLogHead = 'angry,contempt,disgust,fear,happy,sad,surprise,missing,total\n'
    datasetLog = str(countDict['angry']) + ',' + str(countDict['contempt']) + ',' + str(
        countDict['disgust']) + ',' + str(countDict['fear']) + ',' + str(countDict['happy']) + ',' + str(
        countDict['sad']) + ',' + str(countDict['surprise']) + ',' + str(missingLabel) + ',' + str(totalElements)

    csvLogDataset = newPath + 'logDataset.csv'
    with open(csvLogDataset, 'w') as f:
        f.write(datasetLogHead)
        f.write(datasetLog)

    print('Total: ' + str(totalElements))
    print('Missing: ' + str(missingLabel))


def ck_rmn_classify(dataset_path='C:/CKpreprocessed/', contempt_as_neutral=False):
    dataset_log = dataset_path + 'logDataset.csv'
    if contempt_as_neutral is True:
        results_file_name = 'results_contempt_as_neutral.txt'
    else:
        results_file_name = 'results_contempt_ignored.txt'
    labels = []
    y_true = []
    occurrences = []
    y_pred = []
    votes_emotions = {'angry': 0, 'neutral': 0, 'disgust': 0, 'fear': 0, 'happy': 0, 'sad': 0, 'surprise': 0}
    count_emotions = {'angry': 0, 'neutral': 0, 'disgust': 0, 'fear': 0, 'happy': 0, 'sad': 0, 'surprise': 0}
    classifier_rmn = RMN()

    start_time = time.time()

    with open(dataset_log) as csv_file:
        # Extract labels from the log file
        csv_log_dataset = csv.reader(csv_file, delimiter=',')
        head = csv_log_dataset.__next__()
        body = csv_log_dataset.__next__()
        labels = head[:-2]
        labels[1] = 'neutral'
        occurrences = body[:-2]
        occurrences = [int(i) for i in occurrences]  # Convert the values from string to int

    dataset_path_seq = dataset_path + 'sequences/'
    list_ck_dirs = os.listdir(dataset_path_seq)

    count = 0
    for dir in list_ck_dirs:
        if os.path.isdir(dataset_path_seq + dir) is False:
            break
        frame_sequence = os.listdir(dataset_path_seq + dir)
        start = int(frame_sequence.__len__() / 2)
        frame_sequence = frame_sequence[start:]
        true_label = dir.split('_')[0]
        if true_label == 'contempt':
            if contempt_as_neutral is True:
                y_true.append('neutral')
            else:
                break
        else:
            y_true.append(true_label)

        for frame_name in frame_sequence:
            frame = cv2.imread(dataset_path_seq + dir + '/' + frame_name)
            results = classifier_rmn.detect_emotion_for_single_frame(frame)
            label_rmn = results[0]['emo_label']
            votes_emotions[label_rmn] = votes_emotions[label_rmn] + 1
        emotion_voted = max(votes_emotions,
                            key=votes_emotions.get)  # NB: se ci sono più elementi max, prende il primo che incontra. Loggare quando vi sono più max eventualmente
        y_pred.append(emotion_voted)
        print(count)
        count += 1
        votes_emotions = {'angry': 0, 'neutral': 0, 'disgust': 0, 'fear': 0, 'happy': 0, 'sad': 0,
                          'surprise': 0}  # Reset
        count_emotions[emotion_voted] = count_emotions[emotion_voted] + 1
        '''
        if count == 20:
            break
        '''

    # y_pred = [count_emotions['angry'], count_emotions['neutral'], count_emotions['disgust'], count_emotions['fear'], count_emotions['happy'], count_emotions['sad'], count_emotions['surprise']]

    confusionMatrix = confusion_matrix(y_true, y_pred, labels=labels)
    print(confusionMatrix)

    precision_micro = metrics.precision_score(y_true, y_pred, labels=labels, average='micro')
    precision_macro = metrics.precision_score(y_true, y_pred, labels=labels, average='macro')

    jaccard_micro = metrics.jaccard_score(y_true, y_pred, labels=labels,
                                          average='micro')  # In binary and multiclass classification, this function is equivalent to the accuracy_score
    jaccard_macro = metrics.jaccard_score(y_true, y_pred, labels=labels, average='macro')

    recall_micro = metrics.recall_score(y_true, y_pred, labels=labels, average='micro')
    recall_macro = metrics.recall_score(y_true, y_pred, labels=labels, average='macro')

    f1_score_micro = metrics.f1_score(y_true, y_pred, labels=labels, average='micro')
    f1_score_macro = metrics.f1_score(y_true, y_pred, labels=labels, average='macro')

    balanced_accuracy = metrics.balanced_accuracy_score(y_true, y_pred)

    metrics_dict = {'Precision Micro': precision_micro,
                    'Precision Macro': precision_macro,
                    'Jaccard Micro': jaccard_micro,
                    'Jaccard Macro': jaccard_macro,
                    'Recall Micro': recall_micro,
                    'Recall Macro': recall_macro,
                    'F1-Score Micro': f1_score_micro,
                    'F1-Score Macro': f1_score_macro,
                    'Balanced Accuracy': balanced_accuracy}

    classification_report = metrics.classification_report(y_true, y_pred, labels=labels)
    print(classification_report)

    execution_time = time.time() - start_time

    print('Execution time: ' + str(execution_time) + ' seconds.')

    with open(dataset_path + results_file_name, 'w') as f:
        f.write('Classification Report\n')
        f.write(classification_report)
        f.write('\n\n')

        for name, value in metrics_dict.items():
            print(f'{name:10} = {value:.2f}')
            print('\n')
            f.write(f'{name:10} = {value:.2f}')
            f.write('\n')
        f.write('\n\nExecution time: ' + str(int(execution_time)) + ' seconds')

    print(count_emotions)


if __name__ == '__main__':
    # ck_preprocessor()
    ck_rmn_classify()
