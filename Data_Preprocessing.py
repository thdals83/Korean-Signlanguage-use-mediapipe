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
X = np.array(sequences)
y= to_categorical(labels).astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y)
print(X_train.shape)
print(X_test.shape)