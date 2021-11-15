from modules import *
'''
DATA_PATH = os.path.join('Feature_Extraction')

actions = np.array(['도와주세요','구해주세요','119','심장마비','경찰','불나다','기절','배고프다','가렵다','감금','귀','머리','아기','물','낯선사람'])
#actions = np.array(['낯선사람'])
#actions = np.array(['도와주세요','구해주세요'])
number_sequences = 200
sequence_length = 30

for action in actions:
    for sequence in range(number_sequences):
        try:
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
        except: pass
'''
DATA_PATH = os.path.join('Feature_Extraction2')

actions = np.array(['사람이 갑자기 쓰러졌어요','119 구조대를 불러주세요','112에 신고해주세요'])

number_sequences = 200
sequence_length = 50

for action in actions:
    for sequence in range(number_sequences):
        try:
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
        except: pass