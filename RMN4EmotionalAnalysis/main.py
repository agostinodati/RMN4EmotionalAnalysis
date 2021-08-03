import cv2
from shutil import copyfile
from rmn import RMN
import os
import csv
import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix
import time


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

    list_dir = os.listdir(path_dataset)
    for dir in list_dir:
        path_dir = path_dataset + dir + '/'
        if os.path.isdir(path_dir):
            list_sub_dir = os.listdir(path_dir)
            for subdir in list_sub_dir:
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

    print('Total: ' + str(total_elements))
    print('Missing: ' + str(missing_label))


def ck_rmn_classify(dataset_path='C:/CKpreprocessed/', contempt_as_neutral=False):
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
        occurrences = body[:-2]
        occurrences = [int(i) for i in occurrences]  # Convert the values from string to int

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


if __name__ == '__main__':
    ck_preprocessor()
    ck_rmn_classify()
