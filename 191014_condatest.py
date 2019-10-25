import nltk  # 通常用在英文
# nltk.download() 超大天荒地老
import numpy as np
import pandas as pd
# standfordNLP
import stanfordnlp
# stanfordnlp.download('en')
# stanfordnlp.download('zh')

import os
from pathlib import Path
from ckiptagger import data_utils

path = os.path.join(str(Path.home()), 'ckip/')
if not os.path.exists(path): os.mkdir(path)
data_utils.download_data_gdown(path)  # gdrive-ckip2GB

from stanfordnlp.utils.resources import DEFAULT_MODEL_DIR

chinese_sentence = '中華郵政未來智慧物流服務，將取之大眾智慧，帶給民眾更好的便利生活'

zh_pipeline = stanfordnlp.Pipeline(processors="tokenize",
                                   models_dir=DEFAULT_MODEL_DIR,
                                   lang="zh",
                                   use_gpu=False)
zh_doc = zh_pipeline(chinese_sentence)

for i,sentence in enumerate(zh_doc.sentences):
    print("sentence {}:".format(i))
    print("index\ttxt")
    for word in sentence.words:
        print("{}\t{}".format(word.index,word.text))


print("iihdsidihdsio")