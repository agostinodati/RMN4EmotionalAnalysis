import cv2
from shutil import copyfile
from rmn import RMN
import os
import csv
import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix
from scipy import ndimage
import time
import numpy as np


def ck_preprocessor(path_dataset='D:/CKdataset/cohn-kanade-images/', path_label_dir='D:/CKdataset/Emotion_labels/Emotion/',
                    new_path='C:/CKpreprocessed/'):
    '''
    Create a new dataset starting from CK. In this new dataset will be stored only the sequences of images
    that have a emotion label. Sequences labelled with "contempt" will be ignored by the classifier.
    :param path_dataset: Path of CK dataset with all subdirs
    :param path_label_dir: Path of emotion label's directory
    :param new_path:  Path where new dataset will be created
    '''
    if os.path.isdir(new_path) is not True:
        os.mkdir(new_path)
        log_path = new_path
        new_path = new_path + 'sequences/'
        os.mkdir(new_path)

    ck_emotion_dict = {1: 'angry', 2: 'contempt', 3: 'disgust', 4: 'fear', 5: 'happy', 6: 'sad', 7: 'surprise'}

    missing_label = 0
    total_elements = 0
    count_dict = {'angry': 0, 'contempt': 0, 'disgust': 0, 'fear': 0, 'happy': 0, 'sad': 0, 'surprise': 0}


    countTotal = 0
    list_dir = os.listdir(path_dataset)
    for dir in list_dir:
        path_dir = path_dataset + dir + '/'
        if os.path.isdir(path_dir):
            list_sub_dir = os.listdir(path_dir)
            for subdir in list_sub_dir:
                countTotal += 1
                path_sub_dir = path_dir + subdir + '/'
                path_label = path_label_dir + dir + '/' + subdir + '/'
                if os.path.isdir(path_label):
                    print(os.listdir(path_label))
                    list_file_name = os.listdir(path_label)
                    total_elements = total_elements + 1
                    if list_file_name.__len__() == 0:
                        # Not all the expressions are labelled
                        print('Skipped this because the label is missing')
                        print(path_label)
                        missing_label = missing_label + 1
                        break
                    label_file_name = list_file_name[0]
                    emotion_label_filepath = path_label + label_file_name
                    print(emotion_label_filepath)
                    with open(emotion_label_filepath, 'r') as f:
                        label = f.readline()
                        label = int(label.split('.')[0])
                        label_name = ck_emotion_dict[label]
                        print(label)

                    # Count all the label for the log
                    count_dict[label_name] = count_dict[label_name] + 1

                    # Create a new dir passing the label in the title and keeping the names of the original dir and subdir
                    new_dir = label_name + '_' + dir + '_' + subdir
                    new_dir_path = new_path + new_dir + '/'
                    if os.path.isdir(new_dir_path) is not True:
                        os.mkdir(new_dir_path)

                if os.path.isdir(path_sub_dir) and subdir != '.DS_Store':
                    list_frames = os.listdir(path_sub_dir)
                    for frame in list_frames:
                        # Copy the images in the dir into the path created above
                        copyfile(path_sub_dir + frame, new_dir_path + frame)
    dataset_log_head = 'angry,contempt,disgust,fear,happy,sad,surprise,missing,total\n'
    dataset_log = str(count_dict['angry']) + ',' + str(count_dict['contempt']) + ',' + str(
        count_dict['disgust']) + ',' + str(count_dict['fear']) + ',' + str(count_dict['happy']) + ',' + str(
        count_dict['sad']) + ',' + str(count_dict['surprise']) + ',' + str(missing_label) + ',' + str(total_elements)

    csv_log_dataset = log_path + 'logDataset.csv'
    with open(csv_log_dataset, 'w') as f:
        f.write(dataset_log_head)
        f.write(dataset_log)
    print(str(countTotal))
    print('Total: ' + str(total_elements))
    print('Missing: ' + str(missing_label))


def contemp_to_neutral(path_dataset='C:/CKpreprocessed_balanced/sequences/'):
    sequences = os.listdir(path_dataset)
    count_contempt = 0
    for sequence in sequences:
        dir_name = sequence.split('_')
        label = dir_name[0]
        dir_name[0] = 'neutral'
        if label == 'contempt':
            if count_contempt >= 15:
                #os.remove(path_dataset + sequence)
                break
            count_contempt += 1
            name = dir_name[0]
            for word in dir_name[1:]:
                name = name + '_' + word
            os.rename(path_dataset + sequence, path_dataset + name)
            frames = os.listdir(path_dataset + name + '/')
            for frame in frames[3:]:
                os.remove(path_dataset + name + '/' + frame)


def data_augmentation(path_dataset='C:/CKpreprocessed_balanced/sequences/'):
    sequences = os.listdir(path_dataset)
    for sequence in sequences:
        label = sequence.split('_')[0]
        if label == 'angry' or label == 'surprise' or label == 'neutral':
            # There enough data about angry, neutral and surprise emotion
            continue
        frames = os.listdir(path_dataset+sequence)
        new_dir = path_dataset + sequence + '_flipped'
        os.mkdir(new_dir)
        for frame in frames:
            if frame == '.DS_Store':
                os.remove(path_dataset + sequence + '/' + frame)
                continue
            image = cv2.imread(path_dataset + sequence + '/' + frame)
            image_flipped_hor = cv2.flip(image, 1)
            try:
                cv2.imwrite(new_dir + '/flipped_' + frame, image_flipped_hor)
            except():
                print('Exception!\n')
                print(new_dir + '/flipped_' + frame)
    sequences = os.listdir(path_dataset)
    for sequence in sequences:
        label = sequence.split('_')[0]
        if label == 'angry' or label == 'surprise' or label == 'neutral':
            # There enough data about angry, neutral and surprise emotion
            continue
        frames = os.listdir(path_dataset + sequence)
        new_dir = path_dataset + sequence + '_rotated'
        os.mkdir(new_dir)
        for frame in frames:
            image = cv2.imread(path_dataset + sequence + '/' + frame)
            image_rotated = ndimage.rotate(image, 5)
            cv2.imwrite(new_dir + '/rotated_' + frame, image_rotated)


def ck_rmn_classify(dataset_path='C:/CKpreprocessed_balanced/', contempt_as_neutral=False):
    '''
    This function evaluate the RMN on the sequences extracted from the CK+.
    :param dataset_path: Path where is located the dataset extracted from CK+.
    :param contempt_as_neutral: Boolean. This flag allow to ignore the sequences labelled as 'contempt'
    or to consider them as 'neutral'.
    :return: Classification report
    '''
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

    dataset_path_seq = dataset_path + 'sequences/'
    list_ck_dirs = os.listdir(dataset_path_seq)

    count = 0
    for dir in list_ck_dirs:
        if os.path.isdir(dataset_path_seq + dir) is False:
            print(dataset_path_seq+dir)
            continue
        frame_sequence = os.listdir(dataset_path_seq + dir)
        start = int(frame_sequence.__len__() / 2)
        frame_sequence = frame_sequence[start:]
        true_label = dir.split('_')[0]
        if true_label == 'contempt':
            if contempt_as_neutral is True:
                print('SONO QUI!')
                y_true.append('neutral')
            else:
                continue
        else:
            y_true.append(true_label)

        for frame_name in frame_sequence:
            frame = cv2.imread(dataset_path_seq + dir + '/' + frame_name)
            results = classifier_rmn.detect_emotion_for_single_frame(frame)
            label_rmn = results[0]['emo_label']
            votes_emotions[label_rmn] = votes_emotions[label_rmn] + 1
        emotion_voted = max(votes_emotions,
                            key=votes_emotions.get)  # NB: if there are more then 1 max, it takes the first encountered
        y_pred.append(emotion_voted)
        print(count)
        count += 1
        votes_emotions = {'angry': 0, 'neutral': 0, 'disgust': 0, 'fear': 0, 'happy': 0, 'sad': 0,
                          'surprise': 0}  # Reset
        count_emotions[emotion_voted] = count_emotions[emotion_voted] + 1

        '''if count == 5:
            break'''

    print(count)
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

    return classification_report


def ck_rmn_classify_b(dataset_path='C:/CKpreprocessed_balanced/'):
    '''
    This function evaluate the RMN on the sequences extracted from the CK+.
    :param dataset_path: Path where is located the dataset extracted from CK+.
    :return: Classification report
    '''
    dataset_log = dataset_path + 'logDataset.csv'
    results_file_name = 'results_balanced.txt'
    labels = ['angry', 'neutral', 'disgust', 'fear', 'happy', 'sad', 'surprise']
    y_true = []
    y_pred = []
    votes_emotions = {'angry': 0, 'neutral': 0, 'disgust': 0, 'fear': 0, 'happy': 0, 'sad': 0, 'surprise': 0}
    count_emotions = {'angry': 0, 'neutral': 0, 'disgust': 0, 'fear': 0, 'happy': 0, 'sad': 0, 'surprise': 0}
    classifier_rmn = RMN()

    start_time = time.time()

    dataset_path_seq = dataset_path + 'sequences/'
    list_ck_dirs = os.listdir(dataset_path_seq)

    count = 0
    for dir in list_ck_dirs:
        if os.path.isdir(dataset_path_seq + dir) is False:
            print(dataset_path_seq+dir)
            continue
        frame_sequence = os.listdir(dataset_path_seq + dir)
        start = int(frame_sequence.__len__() / 2)
        frame_sequence = frame_sequence[start:]
        true_label = dir.split('_')[0]

        y_true.append(true_label)

        for frame_name in frame_sequence:
            frame = cv2.imread(dataset_path_seq + dir + '/' + frame_name)
            results = classifier_rmn.detect_emotion_for_single_frame(frame)
            label_rmn = results[0]['emo_label']
            votes_emotions[label_rmn] = votes_emotions[label_rmn] + 1
        emotion_voted = max(votes_emotions,
                            key=votes_emotions.get)  # NB: if there are more then 1 max, it takes the first encountered
        y_pred.append(emotion_voted)
        print(count)
        count += 1
        votes_emotions = {'angry': 0, 'neutral': 0, 'disgust': 0, 'fear': 0, 'happy': 0, 'sad': 0,
                          'surprise': 0}  # Reset
        count_emotions[emotion_voted] = count_emotions[emotion_voted] + 1

        '''if count == 5:
            break'''

    print(count)
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

    return classification_report


def video_rmn_classify(dataset_path='C:/dataset_video/'):
    '''
    :param dataset_path:
    :return: Classification report
    '''
    results_file_name = 'results.txt'
    video_log = 'video_log.txt'
    labels = ['angry', 'neutral', 'disgust', 'fear', 'happy', 'sad', 'surprise']
    y_true = []
    y_pred = []
    votes_emotions = {'angry': 0, 'neutral': 0, 'disgust': 0, 'fear': 0, 'happy': 0, 'sad': 0, 'surprise': 0}
    classifier_rmn = RMN()
    stop = 5

    start_time = time.time()

    list_ck_dirs = os.listdir(dataset_path)

    count = 0
    count_video = 1
    with open(dataset_path + video_log, 'w') as f:
        f.write('filename, true, pred\n')
        for dir in list_ck_dirs:
            if os.path.isdir(dataset_path + dir) is False:
                print(dataset_path)
                continue

            videos = os.listdir(dataset_path + dir)

            for video in videos:
                vid = cv2.VideoCapture(dataset_path + dir + '/' + video)
                if not vid.isOpened():
                    print("Cannot open camera")
                    exit()
                while vid.isOpened():
                    success, frame = vid.read()

                    if success is False:
                        break
                    results = classifier_rmn.detect_emotion_for_single_frame(frame)
                    if results.__len__() < 1:
                        continue
                    label_rmn = results[0]['emo_label']
                    votes_emotions[label_rmn] = votes_emotions[label_rmn] + 1
                vid.release()
                emotion_voted = max(votes_emotions,
                                    key=votes_emotions.get)  # NB: if there are more then 1 max, it takes the first encountered
                y_true.append(dir.lower())
                y_pred.append(emotion_voted)
                print(count)
                print(y_true)
                print(y_pred)
                f.write(video + ',' + y_true[count] + ',' + y_pred[count] + '\n')
                votes_emotions = {'angry': 0, 'neutral': 0, 'disgust': 0, 'fear': 0, 'happy': 0, 'sad': 0,
                                  'surprise': 0}  # Reset
                print('count video: ' + str(count_video))
                count += 1
                count_video += 1


    print(count)
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

    return classification_report


def mead_rmn_classify(dataset_path='D:Download/video/'):
    '''
    :param dataset_path:
    :return: Classification report
    '''
    results_file_name = 'D:Download/report/results.txt'
    video_log = 'D:Download/report/video_log.txt'
    pred_filelist = 'D:Download/report/prevision_list'
    labels = ['angry', 'neutral', 'disgust', 'fear', 'happy', 'sad', 'surprise']
    y_true = []
    y_pred = []
    votes_emotions = {'angry': 0, 'neutral': 0, 'disgust': 0, 'fear': 0, 'happy': 0, 'sad': 0, 'surprise': 0}
    classifier_rmn = RMN()
    stop = 1

    start_time = time.time()

    list_direction = os.listdir(dataset_path)

    count = 0
    count_video = 0

    with open(video_log, 'w') as f:
        f.write('filename, true, pred\n')

        for direction in list_direction:
            list_dir_emo = os.listdir(dataset_path + direction)
            for dir in list_dir_emo:
                if os.path.isdir(dataset_path + direction + '/' + dir) is False:
                    print(dataset_path)
                    continue
                if dir.lower() == 'contempt':
                    continue
                levels = os.listdir(dataset_path + direction + '/' + dir)
                for level in levels:
                    videos = os.listdir(dataset_path + direction + '/' + dir + '/' + level)
                    for video in videos:
                        '''
                        if count_video > stop:
                            count_video = 0
                            break'''
                        vid = cv2.VideoCapture(dataset_path + direction + '/' + dir + '/' + level + '/' + video)
                        if not vid.isOpened():
                            print("Cannot open camera")
                            print(dataset_path + direction + '/' + dir + '/' + video)
                            continue
                        while vid.isOpened():
                            success, frame = vid.read()

                            if success is False:
                                break
                            results = classifier_rmn.detect_emotion_for_single_frame(frame)
                            if results.__len__() < 1:
                                continue
                            label_rmn = results[0]['emo_label']
                            votes_emotions[label_rmn] = votes_emotions[label_rmn] + 1
                        vid.release()
                        emotion_voted = max(votes_emotions,
                                            key=votes_emotions.get)  # NB: if there are more then 1 max, it takes the first encountered
                        if dir.lower() == 'disgusted':
                            y_true.append('disgust')
                        elif dir.lower() == 'surprised':
                            y_true.append('surprise')
                        else:
                            y_true.append(dir.lower())
                        y_pred.append(emotion_voted)
                        print(count)
                        print(y_true)
                        print(y_pred)
                        f.write(video + ',' + y_true[count] + ',' + y_pred[count] + '\n')
                        votes_emotions = {'angry': 0, 'neutral': 0, 'disgust': 0, 'fear': 0, 'happy': 0, 'sad': 0,
                                          'surprise': 0}  # Reset
                        print('count video: ' + str(count_video))
                        count += 1
                        count_video += 1
            classification_report_direction = metrics.classification_report(y_true, y_pred, labels=labels)
            print(classification_report_direction)
            with open('D:Download/report/report_' + direction + '.txt', 'w') as fp:
                fp.write('Classification Report\n')
                fp.write(classification_report_direction)
                fp.write('\n\n')
            with open('D:Download/report/true_pred_list_' + direction + '.txt', 'w') as p:
                p.write(str(y_true) + ',')
                p.write(str(y_pred))


    print(count)
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

    with open(results_file_name, 'w') as f:
        f.write('Classification Report\n')
        f.write(classification_report)
        f.write('\n\n')

        for name, value in metrics_dict.items():
            print(f'{name:10} = {value:.2f}')
            print('\n')
            f.write(f'{name:10} = {value:.2f}')
            f.write('\n')
        f.write('\n\nExecution time: ' + str(int(execution_time)) + ' seconds')

        with open(pred_filelist, 'w') as pred:
            pred.write(str(y_true) + ',')
            pred.write(str(y_pred))


    return classification_report


if __name__ == '__main__':
    #ck_preprocessor()
    #data_augmentation()
    #contemp_to_neutral()
    #ck_rmn_classify_b()
    #video_rmn_classify()
    mead_rmn_classify()
