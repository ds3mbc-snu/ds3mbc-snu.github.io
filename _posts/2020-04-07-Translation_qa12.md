---
layout: post
title:  "Translate bAbI to Korean"
date:   2020-04-07 13:00:00
author: Taehee Hong
categories: 2020-1SME
tags:	2020-1SME korQA bAbI
---

```python
import re

def data_sequencing(path):
    data=[]
    with open(path,'r') as f:
        for line in f.readlines():
            line=line.strip()
            index,context=line.split(' ', 1)
            if '\t' in line:
                query,answer,supporting=context.split('\t')
                data.append([index,query,answer,supporting])
            else:
                data.append([index,context,'',''])
    return data
```


```python
train_data=data_sequencing('../bAbI/tasks_1-20_v1-2/en/qa12_conjunction_train.txt')
test_data=data_sequencing('../bAbI/tasks_1-20_v1-2/en/qa12_conjunction_test.txt')
```


```python
import pandas as pd

df_train=pd.DataFrame(train_data,columns=['Index','Query','Answer','Supporting'])
df_test=pd.DataFrame(test_data,columns=['Index','Query','Answer','Supporting'])

df_test[:10]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Index</th>
      <th>Query</th>
      <th>Answer</th>
      <th>Supporting</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>John and Mary travelled to the hallway.</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Sandra and Mary journeyed to the bedroom.</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Where is Mary?</td>
      <td>bedroom</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Mary and Daniel travelled to the bathroom.</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Daniel and Sandra journeyed to the office.</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>Where is Mary?</td>
      <td>bathroom</td>
      <td>4</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>Daniel and Mary went to the bedroom.</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>7</th>
      <td>8</td>
      <td>Daniel and Sandra travelled to the hallway.</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>8</th>
      <td>9</td>
      <td>Where is Sandra?</td>
      <td>hallway</td>
      <td>8</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10</td>
      <td>Mary and Sandra journeyed to the garden.</td>
      <td></td>
      <td></td>
    </tr>
  </tbody>
</table>
</div>




```python
from keras.preprocessing.text import Tokenizer

tokenizer=Tokenizer(filters='!?"#$%&()*+,-/:;<=>@[\\]^_`{|}~\t\n')
tokenizer.fit_on_texts(df_train['Query'])
```

    Using TensorFlow backend.
    C:\WinPython37F\python-3.7.2.amd64\lib\site-packages\tensorflow\python\framework\dtypes.py:516: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
      _np_qint8 = np.dtype([("qint8", np.int8, 1)])
    C:\WinPython37F\python-3.7.2.amd64\lib\site-packages\tensorflow\python\framework\dtypes.py:517: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
      _np_quint8 = np.dtype([("quint8", np.uint8, 1)])
    C:\WinPython37F\python-3.7.2.amd64\lib\site-packages\tensorflow\python\framework\dtypes.py:518: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
      _np_qint16 = np.dtype([("qint16", np.int16, 1)])
    C:\WinPython37F\python-3.7.2.amd64\lib\site-packages\tensorflow\python\framework\dtypes.py:519: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
      _np_quint16 = np.dtype([("quint16", np.uint16, 1)])
    C:\WinPython37F\python-3.7.2.amd64\lib\site-packages\tensorflow\python\framework\dtypes.py:520: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
      _np_qint32 = np.dtype([("qint32", np.int32, 1)])
    C:\WinPython37F\python-3.7.2.amd64\lib\site-packages\tensorflow\python\framework\dtypes.py:525: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
      np_resource = np.dtype([("resource", np.ubyte, 1)])
    C:\WinPython37F\python-3.7.2.amd64\lib\site-packages\tensorboard\compat\tensorflow_stub\dtypes.py:541: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
      _np_qint8 = np.dtype([("qint8", np.int8, 1)])
    C:\WinPython37F\python-3.7.2.amd64\lib\site-packages\tensorboard\compat\tensorflow_stub\dtypes.py:542: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
      _np_quint8 = np.dtype([("quint8", np.uint8, 1)])
    C:\WinPython37F\python-3.7.2.amd64\lib\site-packages\tensorboard\compat\tensorflow_stub\dtypes.py:543: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
      _np_qint16 = np.dtype([("qint16", np.int16, 1)])
    C:\WinPython37F\python-3.7.2.amd64\lib\site-packages\tensorboard\compat\tensorflow_stub\dtypes.py:544: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
      _np_quint16 = np.dtype([("quint16", np.uint16, 1)])
    C:\WinPython37F\python-3.7.2.amd64\lib\site-packages\tensorboard\compat\tensorflow_stub\dtypes.py:545: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
      _np_qint32 = np.dtype([("qint32", np.int32, 1)])
    C:\WinPython37F\python-3.7.2.amd64\lib\site-packages\tensorboard\compat\tensorflow_stub\dtypes.py:550: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
      np_resource = np.dtype([("resource", np.ubyte, 1)])
    


```python
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



데이터의 특징 / QA12
1. 한 query에 사람이 두 명 씩 나옴.  
    : 문장 번역시 `name`을 길이가 2인 list로
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
```


```python
from keras.preprocessing.text import text_to_word_sequence
text_to_word_sequence(df_test['Query'][0])
```




    ['john', 'and', 'mary', 'travelled', 'to', 'the', 'hallway']




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
```


```python
train_tr=data_translation(df_train)
test_tr=data_translation(df_test)
```


```python
def data_reconstruction(original_data,translated_data):
    
    data=[]
    for i in range(len(original_data)):
        index,supporting=original_data[i][0],original_data[i][3]
        query,answer=translated_data[0][i],translated_data[1][i]
                                                      
        data.append([index,query,answer,supporting])
                                                      
    return data
```


```python
df_train_tr=pd.DataFrame(data_reconstruction(train_data,train_tr),
                         columns=['Index','Query','Answer','Supporting'])
df_test_tr=pd.DataFrame(data_reconstruction(test_data,test_tr),
                        columns=['Index','Query','Answer','Supporting'])
```


```python
df_train_tr.to_csv('./qa12_conjunction_train_kr.csv',index=False, encoding = 'utf-8-sig')
df_test_tr.to_csv('./qa12_conjunction_test_kr.csv',index=False, encoding = 'utf-8-sig')
```


```python

```
