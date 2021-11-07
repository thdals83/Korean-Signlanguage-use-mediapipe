from modules import *
from Data_Preprocessing import X_train, y_train, X_test, y_test
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping
import os
checkpoint = ModelCheckpoint('lstm_model2.h5',             # file명을 지정합니다
                             monitor='val_loss',   # val_loss 값이 개선되었을때 호출됩니다
                             verbose=1,            # 로그를 출력합니다
                             save_best_only=True,  # 가장 best 값만 저장합니다
                             mode='auto'           # auto는 알아서 best를 찾습니다. min/max
                            )
earlystopping = EarlyStopping(monitor='val_loss',  # 모니터 기준 설정 (val loss)
                              patience=50,         # 10회 Epoch동안 개선되지 않는다면 종료
                             )
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(50,258)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(3, activation='softmax'))

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
#history = model.fit(X_train, y_train, epochs=700,batch_size=32,validation_data=(X_test,y_test),callbacks=[checkpoint, earlystopping])
history = model.fit(X_train, y_train, epochs=500,validation_data=(X_test,y_test),callbacks=[checkpoint, earlystopping])

plt.plot(history.history['categorical_accuracy'])
plt.plot(history.history['val_categorical_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()


model.save('lstm_model.h5')
model.summary()
# 모델 좀 eally