#!/usr/bin/env python
# -*- coding: utf-8 -*-

import jieba
import pandas as pd
import re
import numpy as np
from keras.callbacks import Callback
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.utils import to_categorical
from sklearn.externals import joblib
import xgboost as xgb
from gensim.models.word2vec import Word2Vec

stop_words = []
sigmoid_threshold = 0.3  # sigmoid to label的阈值,大于此阈值label=1，否则=0


def load_data_from_csv(file_name, header=0, encoding="utf-8"):
    data_df = pd.read_csv(file_name, header=header, encoding=encoding)
    return data_df


def seg_words(contents):
    # —————————————————— 载入停用词 ——————————————————
    # with open('stopWords.txt', 'r') as f:
    #     content = f.read()
    #     stop_words = content.split(' ')
    # ——————————————————————————————————————————————
    contents_segs = list()
    for content in contents:
        rcontent = content.replace("\r\n", " ").replace("\n", " ")
        rcontent = re.sub(r'[\s+\.\!\/_,$%^*(+\"\')]+|[+——()?【】“”！，。？、~@#￥%……&*（）]+', '', rcontent)  # 去标点
        segs = [word for word in jieba.cut(rcontent) if word not in stop_words]
        contents_segs.append(segs)
    return contents_segs

def label_transform(labels, map_dict, reverse = False):
    # 将标量label按字典进行映射以满足模型的要求
    app_dict = map_dict
    if reverse:                     # 按value->key的形式反向映射
        app_dict = dict((v,k) for k,v in map_dict.items())
    transformed_labels = [app_dict[i] for i in labels]
    return np.asarray(transformed_labels)

def sample_balance_weights(labels):
    labels = labels.tolist()
    classes = list(set(labels))
    nb_samples = []
    for i in range(len(classes)):
        nb_samples.append(labels.count(classes[i]))   # 统计每类label的样本数
    total_nb = sum(nb_samples)
    class_weights = dict(zip(classes, [1/(nb/total_nb) for nb in nb_samples]))
    sample_weights = np.asarray([class_weights[c] for c in labels])
    return sample_weights

def label2onehot(labels):
    labels = np.asarray(labels)
    onehot_labels = to_categorical(labels)
    return onehot_labels

def onehot2label(onehot):
    onehot = np.asarray(onehot)
    scalar_labels = np.argmax(onehot, axis=1)
    return scalar_labels

def docTokens2tf_idf(train_tokens, validation_tokens, test_tokens, min_count=10):
    # 将字符串文档转换成tf-idf特征文档，原始文档每个词以空格隔开
    train_docs = []
    validation_docs = []
    test_docs = []
    for d in train_tokens:
        train_docs.append(' '.join(map(str,d.tolist())))    # np array转化为每个元素以空格间隔的字符串
    for d in validation_tokens:
        validation_docs.append(' '.join(map(str,d.tolist())))
    for d in test_tokens:
        test_docs.append(' '.join(map(str,d.tolist())))
    vectorizer = TfidfVectorizer(min_df=min_count)    # 词频低于min_count的词语将被忽略
    transformer = vectorizer.fit(train_docs+validation_docs+test_docs)
    train_tf_idf = transformer.transform(train_docs)
    validation_tf_idf = transformer.transform(validation_docs)
    test_tf_idf = transformer.transform(test_docs)
    return train_tf_idf, validation_tf_idf, test_tf_idf

def build_xgboost_dataformat(train_data_format, validation_data_format, test_data_format, paths):
    train, validation, test = docTokens2tf_idf(train_data_format, validation_data_format, test_data_format)
    xgb_train = xgb.DMatrix(train)
    xgb_validation = xgb.DMatrix(validation)
    xgb_test = xgb.DMatrix(test)

    xgb_train.save_binary(paths[0])
    xgb_validation.save_binary(paths[1])
    xgb_test.save_binary(paths[2])

class F1_Metrics(Callback):                  # 计算F1 score的callback函数
    def __init__(self, val_X, val_y):
        self.val_X = val_X
        self.val_y = val_y

    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []

    def on_epoch_end(self, epoch, logs={}):
        output = self.model.predict(self.val_X)
        if output.shape[1]>1:
            aver_mode = 'macro'
            val_predict = onehot2label(output)
        else:
            aver_mode = 'binary'
            val_predict = np.asarray([(0 if y[0]<sigmoid_threshold else 1) for y in output])

        val_targ = self.val_y
        _val_f1 = f1_score(val_targ, val_predict, average=aver_mode)
        _val_recall = recall_score(val_targ, val_predict, average=aver_mode)
        _val_precision = precision_score(val_targ, val_predict, average=aver_mode)
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        print('-val_f1: %.4f --val_precision: %.4f --val_recall: %.4f' % (_val_f1, _val_precision, _val_recall))
        return

def gensim_word2vec_train():
    corpus1 = joblib.load('tempData/seg_trainset')
    corpus2 = joblib.load('tempData/seg_validateset')
    corpus3 = joblib.load('tempData/seg_testset')
    corpus = np.concatenate((corpus1,corpus2,corpus3))

    train_corpus = []
    for sentence in corpus:
        sentence = sentence.tolist()
        while sentence[0]==0:
            sentence.pop(0)
        train_corpus.append([str(x) for x in sentence])
    model = Word2Vec(train_corpus, size=350, window=7, min_count=5)
    model.save('tempData/gensim_word2vec')
    return model

