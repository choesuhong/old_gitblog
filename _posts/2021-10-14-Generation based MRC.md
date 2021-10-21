---
layout: post
title:  "Generation based MRC"
date:   2021-10-14
author: choesuhong
categories: nlp-Pstage
tags: ['nlp','Pstage','MRC']
use_math: true
published: true
---



# Generation-based MRC

--------------------

## 1. Genertation-based MRC

### Generation-based MRC 문제 정의

MRC 문제를 푸는 방법

1) Extraction-based mrc : context 내 답의 위치를 예측 => 분류 문제 (classification)

![스크린샷 2021-10-20 오전 9.39.37](https://raw.githubusercontent.com/choesuhong/save-image-repo/image/uPic/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202021-10-20%20%E1%84%8B%E1%85%A9%E1%84%8C%E1%85%A5%E1%86%AB%209.39.37.png)

2. Generation-based mrc : 주어진 context와 question을 보고, 답변을 생성 => 생성 문제(generation)

![스크린샷 2021-10-20 오전 9.43.09](https://raw.githubusercontent.com/choesuhong/save-image-repo/image/uPic/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202021-10-20%20%E1%84%8B%E1%85%A9%E1%84%8C%E1%85%A5%E1%86%AB%209.43.09.png)

모든 mrc 문제는 generation-based mrc로 치환이 가능함 => 지문내에 정답이 존재해도 생성하면 그만이기 떄문



### Generation-based MRC 평가 방법

extraction MRC와 동일한 평가 한 방법을 사용

1. Exact Match (EM) Score
2. F1 Score
3. 루지나 blue스코어를 사용하기도 함



### Generation-based MRC Overview

![스크린샷 2021-10-20 오전 9.47.23](https://raw.githubusercontent.com/choesuhong/save-image-repo/image/uPic/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202021-10-20%20%E1%84%8B%E1%85%A9%E1%84%8C%E1%85%A5%E1%86%AB%209.47.23.png)



### Generation-based MRC & Extraction-based MRC 비교

1. MRC 모델 구조

   Seq-to-seq PLM 구조 (generation) vs. PLM + Classifier 구조 (extraction)

2. Loss 계산을 위한 답의 형태 / Prediction의 형태

   Free-form text 형태 (generation) vs. 지문 내 답의 위치 (extraction)

   => Extraction-based MRC: F1 계산을 위해 text로의 별도 변환 과정이 필요

![스크린샷 2021-10-20 오전 9.57.36](https://raw.githubusercontent.com/choesuhong/save-image-repo/image/uPic/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202021-10-20%20%E1%84%8B%E1%85%A9%E1%84%8C%E1%85%A5%E1%86%AB%209.57.36.png)



## 2. Pre-processing

### 입력 표현 - 데이터 예시

![스크린샷 2021-10-20 오전 10.00.10](https://raw.githubusercontent.com/choesuhong/save-image-repo/image/uPic/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202021-10-20%20%E1%84%8B%E1%85%A9%E1%84%8C%E1%85%A5%E1%86%AB%2010.00.10.png)

정답 그대로 넘기는 형태



### 입력 표현 - 토큰화

Tokenization : 텍스트를 의미를 가진 작은 단위로 나눈 것 (형태소)

- Extraction-based MRC 와 같이 WordPiece Tokenizer 를 사용함
  - WordPiece Tokenizer 사전학습 단계에서 먼저 학습에 사용한 전체 데이터 집합(Corpus)에 대해서 구축되어 있어야함
  - 구축 과정에서 미리 각 단어 토큰들에 대해 순서대로 번호(인덱스)를 부여해둠
- Tokenizer 은 입력 텍스트를 토큰화한 뒤, 각 토큰을 미리 만들어둔 단어 사전에 따라 인덱스로 변환함



WordPiece Tokenizer 사용 예시

![스크린샷 2021-10-20 오전 10.04.39](https://raw.githubusercontent.com/choesuhong/save-image-repo/image/uPic/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202021-10-20%20%E1%84%8B%E1%85%A9%E1%84%8C%E1%85%A5%E1%86%AB%2010.04.39.png)



=> 인덱스로 바뀐 질문을 보통 input_ids(또는 input_token_ids)로 부름

=> 모델의 기본 입력은 Input_ids만 필요하나, 그 외 추가적인 정보가 필요함



### 입력표현 - Special Token

학습 시에만 사용되며 단어 자체의 의미는 가지지 않는 특별한 ㅋ토큰

- SOS(start Of Sentence), EOS(End Of Sentence), CLS, SEP, PAD, UNK .. 등등

  => Extraction-based MRC 에선 CLS, SEP, PAD 토큰을 사용

  => Genration-based MRC 에서도 PAD 토큰은 사용됨. CLS, SEP 토큰 또한 사용할 수 있으나, 대신 자연어를 이용하여 정해진 텍스트 포맷으로 데이터를 생성

  ![스크린샷 2021-10-20 오전 10.08.08](https://raw.githubusercontent.com/choesuhong/save-image-repo/image/uPic/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202021-10-20%20%E1%84%8B%E1%85%A9%E1%84%8C%E1%85%A5%E1%86%AB%2010.08.08.png)



### 입력 표현 - additional information

Attention mask

- Extraction-based MRC 와 똑같이 어텐션 연산을 수행할 지 결정하는 어넽션 마스크 존재



Token type ids

- BERT 와 달리 BART 에서는 입력시퀸스에 대한 구분이 없어 token_type_ids 가 존재하지 않음
- 따라서 Extraction-based MRC 와 달리 입력에 token_type_ids 가 들어가지 않음

![스크린샷 2021-10-20 오전 10.13.41](https://raw.githubusercontent.com/choesuhong/save-image-repo/image/uPic/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202021-10-20%20%E1%84%8B%E1%85%A9%E1%84%8C%E1%85%A5%E1%86%AB%2010.13.41.png)

why? token_type_ids로 구분을 해주려고 했으나 실질적으로 영향이 없어 뺸것으로 보임



### 출력 표현 - 정답 출력

Sequence of token ids

- Extraction-based MRC에선 텍스트를 생성해내는 대신 시작/끝 토큰의 위치를 출력하는 것이 모델의 최종 목표였음
- Generation-based MRC는 그보다 조금 더 어려운 실제 텍스트를 생성하는 과제를 수행
- 전체 시퀸스의 각 위치 마다 모델이 아는 모든 단어들 중 하나의 단어를 맞추는 classification 문제

![스크린샷 2021-10-20 오전 10.17.26](https://raw.githubusercontent.com/choesuhong/save-image-repo/image/uPic/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202021-10-20%20%E1%84%8B%E1%85%A9%E1%84%8C%E1%85%A5%E1%86%AB%2010.17.26.png)



도식화 그림

![스크린샷 2021-10-20 오전 10.18.12](https://raw.githubusercontent.com/choesuhong/save-image-repo/image/uPic/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202021-10-20%20%E1%84%8B%E1%85%A9%E1%84%8C%E1%85%A5%E1%86%AB%2010.18.12.png)



## 3. Model

### BART

기계 독해, 기계 번역, 요약, 대화 등 sequence to sequence 문제의 pre-training을 위한 denoising autoencoder

![스크린샷 2021-10-20 오전 11.43.21](https://raw.githubusercontent.com/choesuhong/save-image-repo/image/uPic/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202021-10-20%20%E1%84%8B%E1%85%A9%E1%84%8C%E1%85%A5%E1%86%AB%2011.43.21.png)



### BART Encoder & Decoder

- BART의 인코더는 BERT처럼 bi-directional
- BART의 디코더는 GPT처럼 uni-directional(autoregressive)

![스크린샷 2021-10-20 오전 11.45.58](https://raw.githubusercontent.com/choesuhong/save-image-repo/image/uPic/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202021-10-20%20%E1%84%8B%E1%85%A9%E1%84%8C%E1%85%A5%E1%86%AB%2011.45.58.png)



### Pre-training BART

BART는 텍스트에 노이즈를 주고 원래 텍스트를 복구하는 문제를 푸는 것으로 pre-training함

![스크린샷 2021-10-20 오전 11.47.37](https://raw.githubusercontent.com/choesuhong/save-image-repo/image/uPic/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202021-10-20%20%E1%84%8B%E1%85%A9%E1%84%8C%E1%85%A5%E1%86%AB%2011.47.37.png)

![스크린샷 2021-10-20 오전 11.48.42](https://raw.githubusercontent.com/choesuhong/save-image-repo/image/uPic/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202021-10-20%20%E1%84%8B%E1%85%A9%E1%84%8C%E1%85%A5%E1%86%AB%2011.48.42.png)



## 4. Post-processing

### Searching

![스크린샷 2021-10-20 오전 11.50.14](https://raw.githubusercontent.com/choesuhong/save-image-repo/image/uPic/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202021-10-20%20%E1%84%8B%E1%85%A9%E1%84%8C%E1%85%A5%E1%86%AB%2011.50.14.png)




