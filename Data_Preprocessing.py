from modules import *
from folder_setup import *

classes = {label:num for num, label in enumerate(actions)}
print(classes)
sequences, labels = [], []
for action in actions:
    for sequence in range(number_sequences):

        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
            window.append(res)
        sequences.append(window)
        labels.append(classes[action])

print(np.array(sequences).shape)
# X and y variables
X = np.array(sequences)
#np.random.shuffle(X)
y= to_categorical(labels).astype(int)
#np.random.shuffle(y)
# train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)
print(X_train.shape)
print(X_test.shape)

'''
from Code.modules import *
from Code.folder_setup import *
from Code.folder_setuptest import *
import os

classes = {label:num for num, label in enumerate(actions)}

sequences, labels = [], []
for action in actions:
    for sequence in range(number_sequences):
        window = []
        vpath_dir = 'C:/Users/user/PycharmProjects/Korean-Signlanguage-use-mediapipe/Code/Feature_Extraction/'+action+'/'+str(sequence)
        file_list = os.listdir(vpath_dir)

        num = 0
        for frame_num in range(sequence_length):
            if num == len(file_list):
                num = 0
                break
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
            window.append(res)
            num += 1
        sequences.append(window)
        labels.append(classes[action])

# X and y variables
X = np.array(sequences)
y= to_categorical(labels).astype(int)

# train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)
print(X_train.shape)
'''

