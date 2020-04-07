---
layout: post
title:  "Translate bAbI to Korean"
date:   2020-04-07 13:00:00
author: Taehee Hong
categories: 2020-1SME
tags:	2020-1SME korQA bAbI
---

먼저 data를 받았을 때 DataFrame으로 만들기 전의 처리를 한다.
```python
import re

def data_sequencing(path):
    data=[]
    with open(path,'r') as f:
        for line in f.readlines():
            line=line.strip()
            index,context=line.split(' ', 1)
            # 질문이 있는 행의 경우
            if '\t' in line:
                query,answer,supporting=context.split('\t')
                data.append([index,query,answer,supporting])
            # 질문이 없는 행의 경우
            else:
                data.append([index,context,'',''])
    return data
```

이제 `train_data`와 `test_data`를 가져오고 이를 DataFrame으로 나타내자.
```python
train_data=data_sequencing('../bAbI/tasks_1-20_v1-2/en/qa12_conjunction_train.txt')
test_data=data_sequencing('../bAbI/tasks_1-20_v1-2/en/qa12_conjunction_test.txt')

import pandas as pd

df_train=pd.DataFrame(train_data,columns=['Index','Query','Answer','Supporting'])
df_test=pd.DataFrame(test_data,columns=['Index','Query','Answer','Supporting'])

df_test[:10]
```

||Index|Query|Answer|Supporting|
|---|---|---|---|---|
|0|1|John and Mary travelled to the hallway.|||
|1|2|Sandra and Mary journeyed to the bedroom.|||
|2|3|Where is Mary?|bedroom|2|
|3|4|Mary and Daniel travelled to the bathroom.|||
|4|5|Daniel and Sandra journeyed to the office.|||
|5|6|Where is Mary?|bathroom|4|
|6|7|Daniel and Mary went to the bedroom.|||
|7|8|Daniel and Sandra travelled to the hallway.|||
|8|9|Where is Sandra?|hallway|8|
|9|10|Mary and Sandra journeyed to the garden.|||

이제 `keras`의 `Tokenizer`를 이용해 영어 단어들을 토큰화해준다.
```python
from keras.preprocessing.text import Tokenizer

tokenizer=Tokenizer(filters='!?"#$%&()*+,-/:;<=>@[\\]^_`{|}~\t\n')
tokenizer.fit_on_texts(df_train['Query'])

tokenizer.word_index
```

    {'and': 1,
     'to': 2,
     'the': 3,
     'daniel': 4,
     'john': 5,
     'sandra': 6,
     'mary': 7,
     'where': 8,
     'is': 9,
     'went': 10,
     'journeyed': 11,
     'moved': 12,
     'back': 13,
     'travelled': 14,
     'hallway.': 15,
     'kitchen.': 16,
     'garden.': 17,
     'bathroom.': 18,
     'bedroom.': 19,
     'office.': 20}


qa12 데이터의 특징
1. 질문이 아닌 query에서는 사람이 둘씩, 질문인 query에서는 하나씩 나옴.
  : 일단 다 하나의 list에 받고, 질문일 때(`tokenized[0] == 'where'`)와 아닐 때 나눠서 문장을 재구성.
2. `back`은 `went back`의 형태로밖에 나타나지 않음  
  : `went`를 봤을 시 뒷 단어를 보고 결정


```python
voca = {
    'name': {
        'daniel': '동수',
        'john'  : '준석',
        'sandra': '수아',
        'mary'  : '민경'
    },
    'verb': {
        'journeyed': '여행했다.',
        'moved'    : '이동했다.',
        'went'     : '갔다.',
        'travelled': '여행했다.'
    },
    'place': {
        'hallway' : '복도',
        'kitchen' : '부엌',
        'garden'  : '정원',
        'bathroom': '욕실',
        'bedroom' : '침실',
        'office'  : '사무실'
    }
}

from keras.preprocessing.text import text_to_word_sequence
text_to_word_sequence(df_test['Query'][0])
```
['john', 'and', 'mary', 'travelled', 'to', 'the', 'hallway']

단어를 이어붙일 때 앞 단어의 종성에 따라 조사가 바뀐다.
구글에서 [관련된 내용](https://github.com/myevan/pyjosa/blob/master/pyjosa.py)을 찾아서 참고함.
```python
# 조사 변경
# https://github.com/myevan/pyjosa/blob/master/pyjosa.py 참조
def josa(text, input):
    if input == '은는':
        if (ord(text[-1])- 0xac00)%28 != 0: # 종성이 있을 때
                output = '은 '
        else:
                output = '는 '
    elif input == '와과':
        if (ord(text[-1])- 0xac00)%28 != 0: # 종성이 있을 때
                output = '과 '
        else:
                output = '와 '
    elif input == '으로':
        if (ord(text[-1])- 0xac00)%28 in [0,8]: # 종성이 없거나 ㄹ일 때
                output = '로 '
        else:
                output = '으로 '
    return(text+output)
```

이제 DataFrame을 받아서 번역된 DataFrame을 돌려주는 함수를 만들자.
```python
def data_translation(data):

    query_tr=[]
    for query in data['Query']:
        tokenized=text_to_word_sequence(query)
        name, verb, place = [], '', ''
        for word in tokenized:
            if word in voca['name'].keys():
                name.append(voca['name'][word])
            elif word in voca['verb'].keys():
                if word == 'went' and tokenized[tokenized.index(word)+1] == 'back':
                    verb = '돌아왔다.'
                else:
                    verb = voca['verb'][word]
            elif word in voca['place'].keys():
                place = voca['place'][word]

        # assemble
        if tokenized[0] == 'where':
            if place == '':
                place = '어디'
            query_tr.append(josa(name[0],'은는') + place + '에 있습니까?')
        else:
            query_tr.append(josa(name[0],'와과')+josa(name[1],'은는')  +josa(place,'으로')+verb)
                

    answer_tr=[]
    for answer in data['Answer']:
        if answer:
            answer_tr.append(voca['place'][answer])
        else:
            answer_tr.append(answer)
       
    return query_tr,answer_tr
    
train_tr=data_translation(df_train)
test_tr=data_translation(df_test)
```

이제 query와 answer의 번역은 끝났다. 원 데이터와 비교해서 index와 supporting을 채우고 csv로 저장해주자.
```python
def data_reconstruction(original_data,translated_data):
    
    data=[]
    for i in range(len(original_data)):
        index,supporting=original_data[i][0],original_data[i][3]
        query,answer=translated_data[0][i],translated_data[1][i]
                                                      
        data.append([index,query,answer,supporting])
                                                      
    return data

df_train_tr=pd.DataFrame(data_reconstruction(train_data,train_tr),
                         columns=['Index','Query','Answer','Supporting'])
df_test_tr=pd.DataFrame(data_reconstruction(test_data,test_tr),
                        columns=['Index','Query','Answer','Supporting'])

df_train_tr.to_csv('./qa12_conjunction_train_kr.csv',index=False, encoding = 'utf-8-sig')
df_test_tr.to_csv('./qa12_conjunction_test_kr.csv',index=False, encoding = 'utf-8-sig')
```