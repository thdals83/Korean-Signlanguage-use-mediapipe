# Korean Signlanguage use mediapipe
미디어파이프를 이용한 한국형 수화 영상 번역 시스템


### 기술 스택
<p align='center'>
<img src="https://img.shields.io/badge/Python-3776AB?stylestyle=for-the-badge&logo=Python&logoColor=white">
  <img src="https://img.shields.io/badge/C++-00599C?stylestyle=for-the-badge&logo=C++&logoColor=white">
  <img src="https://img.shields.io/badge/Keras-D00000?stylestyle=for-the-badge&logo=Keras&logoColor=white">
  <img src="https://img.shields.io/badge/TensorFlow-FF6F00?stylestyle=for-the-badge&logo=TensorFlow&logoColor=white">
  <img src="https://img.shields.io/badge/OpenCV-5C3EE8?stylestyle=for-the-badge&logo=OpenCV&logoColor=white">
   matplotlib
   mediapipe
</p>

### 배경
  - 일반인은 수화를 이해하지 못해, 수화 사용자와의 대화에 불편함이 존재
  - 컴퓨터 비전과 인공지능을 이용해 해결하는 것을 목표
  - AI HUB의 수화 데이터 + 직접 촬영한 수화 데이터
  - MediaPipe를 통한 영상 인식
  - RNN의 LSTM 딥러닝 기술을 이용한 단어 예측을 통해 한국 수화 동작을 예측

### 사용 데이터
  - AI HUB의 수화 데이터
  - 부족한 영상은 직접 촬영

### 기능구현
  - 총 15개의 단어에 대해서 단어당 200개의 영상 데이터를 가지고 해당 수화의 분류
  - lstm 모델의 경우 옵티마이저(Optimezer)- 아담(Adam), Loss 함수 - ategorical_crossentropy, 3계층으로 하여 사용
  - 프레임당 lstm 모델로 학습시켜 비교 예측

### 모델 예측 결과
<img src="https://user-images.githubusercontent.com/59475849/147439474-f35aa642-8bbd-4770-9b31-27cf00b072b3.png">
<img src="https://user-images.githubusercontent.com/59475849/147439489-45d88970-a66a-48cc-b471-9a60bf98198d.png">

