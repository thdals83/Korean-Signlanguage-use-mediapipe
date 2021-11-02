from folder_setup import actions
from modules import *
from Data_Preprocessing import X_train, y_train, X_test, y_test
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping
import itertools
from sklearn.metrics import confusion_matrix
import tensorflow as tf
import matplotlib.pyplot as plt

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    # Confusion matrix를 그리기 위한 코드
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, size=40)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90, fontsize=30)
    plt.yticks(tick_marks, classes, fontsize=30)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black", fontsize=25)

    plt.tight_layout()
    plt.ylabel('True label', fontsize=25)
    plt.xlabel('Predicted label', fontsize=25)

new_model = tf.keras.models.load_model('lstm_model.h5')
xhat = X_test
yhat = new_model.predict(xhat)

cfm = confusion_matrix(np.argmax(y_test,axis=1), np.argmax(yhat, axis=1))

np.set_printoptions(precision=2)
plt.figure(figsize=(20,10))
plt.rc('font', family='Malgun Gothic')

plot_confusion_matrix(cfm, classes=actions, title='Confusion Matrix on Test data')
plt.show()