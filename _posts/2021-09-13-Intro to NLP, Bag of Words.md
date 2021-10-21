---
layout: post
title:  "Intro to NLP, Bag-of-Words"
date:   2021-09-13
author: choesuhong
categories: nlp-Ustage
tags: ['nlp','UstageW']
use_math: true
published: true
---

# 목표

- [ ] Bag-of-Words
- [ ] Naive Bayes Classifier
- [ ] 단어를 벡터로 표현하는 방법, 문서를 벡터로 표현하는 방법



# NLP task

---------------------------

- low_level parsing
  - Tokenization, stemming
- Word and phrase level
  - Named entity recougnition(NER), part-of-speech(POS) tagging, noun-phrase chunking, dependency parsing, coreference resolution
- Sentence level
  - Sentiment analysis, machine translation
- Multi-sentence and paragraph level
  - Entailment prediction, question answering, dialog systems, summarization



# Trends of NLP

---------------

- word embedding
- RNN -> LSTM or GRUs -> Transformer



# Bag-of-Words

---------------------------------------

- 단어 및 문서를 숫자형태로 표현
- unique word를 모아 Vocabulary를 구축
- Encoding to one-hot vectors
- 안녕하세요 $LaTeX$ 입니다.

- sum of one-hot vectors

![image-20210906145954273](https://raw.githubusercontent.com/choesuhong/save-image-repo/image/img/image-20210906150108098.png)



# NaiveBayes Classifier for Document Classification

-------------------

![image-20210906150108098](https://raw.githubusercontent.com/choesuhong/save-image-repo/image/img/image-20210906145954273.png)

영상에 나오는 NavieBayes 계산해보기
