# modeluse.py

from pymongo import MongoClient
import tensorflow as tf
import numpy as np
from konlpy.tag import Okt
from pprint import pprint

selected_words = []
model = None
okt = Okt()
all_count = 0
pos_count = 0

# MongoDB 데이터 불러오기
client = MongoClient('127.0.0.1', 27017)
db = client['local']
collection = db.get_collection('movie')
reply_list = []

def mongo_select_all():
    for one in collection.find({'movieNm': {'$eq':'야구소녀'}}, {'_id':0, 'movieNm':1, 'content':1, 'score':1}):  # 제목, 내용
        reply_list.append([one['movieNm'], one['content'], one['score']]) # dict에서 Value와 Score
    return reply_list

mongo_select_all()
#print(len(reply_list))
all_count = len(reply_list)

def read_data(filename):
    words_data = []
    with open(filename, 'r', encoding='UTF-8') as f:
        while True:
            line = f.readline()[:-1]
            if not line: break
            words_data.append(line)
    return  words_data

selected_words = read_data('selectword.txt')
model = tf.keras.models.load_model('my_model.h5')

# 예측할 데이터의 전처리를 진행할 메서드
def tokenize(doc):
    # norm은 정규화, stem은 근어로 표시하기를 나타냄
    return ['/'.join(t) for t in okt.pos(doc, norm=True, stem=True)]

# 예측할 데이터의 벡터화를 진행할 메서드(임베딩)
def term_frequency(doc):
    return [doc.count(word) for word in selected_words]

# 모델로 예측하는 메서드 구현
def predict_pos_neg(review):
    token = tokenize(review)
    tf = term_frequency(token)
    data = np.expand_dims(np.asarray(tf).astype('float32'), axis=0)
    score = float(model.predict(data))
    global pos_count
    if(score > 0.5):
        pos_count += 1
        print("[{}]sms {:.2f}% 확률로 긍정 리뷰이지 않을까 추측해봅니다^^\n".format(review, score*100))
    else:
        print("[{}]는 {:.2f}% 확률로 부정 리뷰이지 않을까 추측해봅니다^^\n".format(review, (1-score)*100))

# 예측시작
def predict():
    for one in reply_list:
        predict_pos_neg(one[1])

    aCount = all_count
    pCount = pos_count
    pos_pct = (pCount*100)/aCount
    neg_pct = 100 - pos_pct
    #print(aCount, pCount, pos_pct, neg_pct
    print('■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■')
    print('■({}) 댓글 {}개를 감성분석한 결과'.format(reply_list[0][0], aCount))
    print('■긍정적인 의견{:.2f}% / 부정적인 의견{:.2f}%'.format(pos_pct, neg_pct))
    print('■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■')

predict()