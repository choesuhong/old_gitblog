---
layout: post
title:  "MRC Intro and Python Basics"
date:   2021-10-12
author: choesuhong
categories: nlp-Pstage
tags: ['nlp','Pstage','MRC']
use_math: true
published: true
---

# MRC(기계독해) Intro & Python Basics

---------------

## 목차

### (MRC)기계독해

기계독해란 무엇일까?

기계독해는 어떠한 종류가 있을까?

그럼 평가는 어떻게 해야할까?



### 자연어 처리에 필요한 개념

unicode

tokenization



### KorQuAD

실제 기계독해 데이터의 구성



### MRC(Machine Reading Comprehension)란?

말 그대로 주어진 context를 이해하고, 주어진 (Query/Question)의 답변을 추론하는 문제 

![Screen Shot 2021-10-15 at 11.51.15 AM](https://raw.githubusercontent.com/choesuhong/save-image-repo/image/uPic/Screen%20Shot%202021-10-15%20at%2011.51.15%20AM.png)



지문이 어떤 지문과 관계가 있는지 알기가 어렵기 때문에 질문에 대해 관련된 지문을 찾고 그 지문에서 질문에 대한 답을 내놓음

![Screen Shot 2021-10-15 at 11.56.49 AM](https://raw.githubusercontent.com/choesuhong/save-image-repo/image/uPic/Screen%20Shot%202021-10-15%20at%2011.56.49%20AM.png)



MRC의 종류

1. Extractive Answer Datasets - question에 대한 답이 항상 주어진 context의 segment(or span)으로 존재![Screen Shot 2021-10-15 at 12.00.09 PM](https://raw.githubusercontent.com/choesuhong/save-image-repo/image/uPic/Screen%20Shot%202021-10-15%20at%2012.00.09%20PM.png)



2. Descriptive/Narrative Answer Datasets - context 내에서 추출한 span이 아닌 question을 보고 생성된 sentence (or free-form)의 형태

![Screen Shot 2021-10-15 at 12.01.08 PM](https://raw.githubusercontent.com/choesuhong/save-image-repo/image/uPic/Screen%20Shot%202021-10-15%20at%2012.01.08%20PM.png)



3. Multiple-choice Datasets - question에 대한 답을 여러 개의 answer candidates 중 하나로 고르는 형태

![Screen Shot 2021-10-15 at 12.04.09 PM](https://raw.githubusercontent.com/choesuhong/save-image-repo/image/uPic/Screen%20Shot%202021-10-15%20at%2012.04.09%20PM.png)

