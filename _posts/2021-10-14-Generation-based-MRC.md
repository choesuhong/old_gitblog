---
layout: post
title:  "Generation-based MRC"
date:   2021-10-14
author: choesuhong
categories: nlp-Pstage
tags: ['nlp','Pstage','MRC']
use_math: true
---

# Extraction-based MRC

---------------------

## 목표

### 실제 추출기반 기계독해를 어떻게 풀까?

- #### 학습 전 준비해야 할 단계

- #### 모델 학습 단계

- #### 추출기반으로 답을 얻어냄

- #### 얻은 답을 원하는 텍스트 형태로 변형

 

## 1. Extraction-based MRC

-----------

quesion의 answer이 항상 주어진 context내에 span으로 존제![스크린샷 2021-10-19 오후 2.11.28](https://raw.githubusercontent.com/choesuhong/save-image-repo/image/uPic/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202021-10-19%20%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE%202.11.28.png)



### 평가 방법

- Exact Match (EM) Score
- F1-Score 
  - $Precision = \frac{num(same\_token)}{num(pred\_koekns)}$
  - $Recall = \frac{num(same\_token)}{num(ground\_tokens)}$
  - $F 1=\frac{2 \times \text { Precision } \times \text { Recall }}{\text { Precision }+\text { Recall }}$

![스크린샷 2021-10-19 오후 2.30.27](https://raw.githubusercontent.com/choesuhong/save-image-repo/image/uPic/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202021-10-19%20%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE%202.30.27.png)

​			Precision과 Recall 을 계산을 해서 evaluation이 가장 높은 갚을 산출함



### Extraction-based MRC Overview

![스크린샷 2021-10-19 오후 2.31.56](https://raw.githubusercontent.com/choesuhong/save-image-repo/image/uPic/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202021-10-19%20%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE%202.31.56.png)





## 2. Pre-Processing

---------

### 입력 형태

![스크린샷 2021-10-19 오후 2.44.52](https://raw.githubusercontent.com/choesuhong/save-image-repo/image/uPic/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202021-10-19%20%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE%202.44.52.png)



### Tokenization

텍스트를 작은 단위(Token)로 나누는 것

- 띄어쓰기 기준, 형태소, subword 여러 단위 토큰 기준이 사용
- 최근엔 Out-Of-Vocabulary(OOV) 문제를 해결해주고 정보학적으로 이점을 가진 Byte Pair Encoding(BPE)을 주로 사용



WordPiece Tokenzier 사용 예시

![스크린샷 2021-10-19 오후 2.48.17](https://raw.githubusercontent.com/choesuhong/save-image-repo/image/uPic/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202021-10-19%20%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE%202.48.17.png)

직관적으로 표현한다면 => 자주 나오는 단어는 합쳐서 표현, 덜 나오는 단어들은 나누어서 표현



### Special Tokens

![스크린샷 2021-10-19 오후 2.52.15](https://raw.githubusercontent.com/choesuhong/save-image-repo/image/uPic/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202021-10-19%20%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE%202.52.15.png)

<center>before tokenizer</center>

![스크린샷 2021-10-19 오후 2.55.31](https://raw.githubusercontent.com/choesuhong/save-image-repo/image/uPic/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202021-10-19%20%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE%202.55.31.png)

<center>after tokenizer</center>



### Attention Mask

- 입력 시퀸스 중에서 attention 을 연산할 때 무시할 토큰을 표시
- 0은 무시, 1은 연산에 표함
- 보통 [PAD]와 같은 의미가 없는 특수토큰을 무시하기 위해 사용

![스크린샷 2021-10-19 오후 2.58.24](https://raw.githubusercontent.com/choesuhong/save-image-repo/image/uPic/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202021-10-19%20%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE%202.58.24.png)

모델 입장에서 무시할 것들을 알려주는 masking 작업



### Token Type IDs

입력이 2개이상의 시퀸스일 떄 (예: 질문 & 지문), 각각에게 ID를 부여하여 모델이 구분해서 해석하도록 유도

![스크린샷 2021-10-19 오후 3.00.27](https://raw.githubusercontent.com/choesuhong/save-image-repo/image/uPic/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202021-10-19%20%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE%203.00.27.png)



### 모델 출력값

- 정답은 문서내 존재하는 연속된 단어토큰 (span)이므로, span의 시작과 끝 위치를 알면 정답을 맞출 수 있음
- Extraction-based에선 답안을 생성하기 보다, 시작위치와 끝위치를 예측하도록 학습함. 즉 Token Classification문제로 치환

![스크린샷 2021-10-19 오후 3.04.44](https://raw.githubusercontent.com/choesuhong/save-image-repo/image/uPic/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202021-10-19%20%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE%203.04.44.png)

답안을 생성하게 만든다면 국+육군+부+참모+총장이 될수 있는 경우가 있기 때문에 위치를 예측하도록 하여 분류문제로 치환하였다.



## 3. Fine-tuning

-----------

### Fine-tuning BERT



![스크린샷 2021-10-19 오후 3.08.00](https://raw.githubusercontent.com/choesuhong/save-image-repo/image/uPic/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202021-10-19%20%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE%203.08.00.png)





## 4. Post-processing

--------------

### 불가능한 답 제거하기

다음과 같은 경우 candidate list에서 제거

- End position 이 start position보다 앞에 있는 경우 (start=90, end=80)
- 예측한 위치가 context를 벗어난 경우 (question 위치쪽에 답이 나온 경우)
- 미리 설정한 max_answer_length 보다 길이가 더 긴 경우



### 최적의 답안 찾기

1. Start/end position prediction에서 score (logits)가 가장 높은 N개를 각각 찾는다.
2. 불가능한 start/end 조합을 제거한다.
3. 가능한 조합들을 score의 합이 큰 순서대로 정렬한다.
4. score가 가장 큰 조합을 최종 예측으로 선정한다.
5. Top-k 가 필요한 경우 차례대로 내보낸다.




