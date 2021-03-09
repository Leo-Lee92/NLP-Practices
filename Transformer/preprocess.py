# %%
import collections
import logging
import os
import pathlib
import re
import string
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
import tensorflow_text as text
import tensorflow as tf
# %%
# (아마 텐서플로로 작업을 수행할 때 발생하는 경고를 무시하도록 하는 명령어 인듯)
logging.getLogger('tensorflow').setLevel(logging.ERROR)  # suppress warnings 

# %%
# 데이터 로드
## TED Talks Open Translation Project의 포르투갈-영어 번역 데이터 세트
## 50k 개의 교육 예제, 1.1k개의 검증 예쩨 및 2k개의 테스트 예제를 포함함
examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en', with_info=True, as_supervised=True)
train_examples, val_examples = examples['train'], examples['validation']

# 데이터 확인
for pt_examples, en_examples in train_examples.batch(3).take(1):
    for pt in pt_examples.numpy():
        print(pt.decode('utf-8'))

    print()

    for en in en_examples.numpy():
        print(en.decode('utf-8'))

# %%
