#!/usr/bin/env python
# -*- coding: utf-8 -*-

import config
import logging
import os
import argparse
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.externals import joblib
from util import *
from model import *

parser = argparse.ArgumentParser()
parser.add_argument('-bd', '--build_dataset', type=int, nargs='?',
                    default=0)
parser.add_argument('-bxd', '--build_xgb_dataset', type=int, nargs='?',
                    default=0)
parser.add_argument('-mn', '--model_No', type=int, nargs='?',
                    default=0)
parser.add_argument('-vn', '--validation_No', type=int, nargs='?',
                    default=0)
args = parser.parse_args()

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] <%(processName)s> (%(threadName)s) %(message)s')
logger = logging.getLogger(__name__)

# hyperParameters
model_name = 'AwsomeTextCNN'
train_model = AwsomeTextCNN
num_class = 4

tokenizer_path = 'tempData/words_tokenizer'
seg_trainset_path = 'tempData/seg_trainset'
seg_validateset_path = 'tempData/seg_validateset'
seg_testset_path = 'tempData/seg_testsetb'
xgb_trainset_path = 'tempData/xgb_train'
xgb_validateset_path = 'tempData/xgb_validateset'
xgb_testset_path = 'tempData/xgb_testset'
F1score_path = 'tempData/F1score.csv'
max_len = 300
embedding_dim = 350
batch_size = 512
epochs = 50
end_col = None


def PrepareData():
    #----------------------------load train data---------------------------------
    if args.build_dataset == 1:
        logger.info("start load data")
        train_data_df = load_data_from_csv(config.train_data_path)
        content_train = train_data_df.iloc[:, 1]
        validate_data_df = load_data_from_csv(config.validate_data_path)
        content_validate = validate_data_df.iloc[:, 1]
        test_data_df = load_data_from_csv(config.test_data_path)
        content_test = test_data_df.iloc[:, 1]

        logger.info("start seg train data")
        content_train = seg_words(content_train)
        logger.info("complete seg train data")
        logger.info("start seg validate data")
        content_validate = seg_words(content_validate)
        logger.info("complete seg validate data")
        logger.info("start seg test data")
        content_test = seg_words(content_test)
        logger.info("complete seg test data")

        logger.info("prepare data format")
        words = []
        for x in content_train+content_validate+content_test:
            for w in x:
                words.append(w)
        max_words = len(set(words))
        # max_words = 40000
        tokenizer = Tokenizer(num_words=max_words)
        tokenizer.fit_on_texts(content_train+content_validate+content_test)
        data_train = tokenizer.texts_to_sequences(content_train)
        train_data_format = pad_sequences(data_train, maxlen=max_len)
        joblib.dump(train_data_format, seg_trainset_path)  # 保存分词、转化为索引后的训练集
        data_val = tokenizer.texts_to_sequences(content_validate)
        validate_data_format = pad_sequences(data_val, maxlen=max_len)
        joblib.dump(validate_data_format, seg_validateset_path)  # 保存分词、转化为索引后的验证集
        data_test = tokenizer.texts_to_sequences(content_test)
        test_data_format = pad_sequences(data_test, maxlen=max_len)
        joblib.dump(test_data_format, seg_testset_path)  # 保存分词、转化为索引后的测试集
        joblib.dump(tokenizer, tokenizer_path)      # 保存索引词典
        logger.info("complete formate data")

    elif args.build_dataset == 2:
        logger.info("start load data")
        test_data_df = load_data_from_csv(config.test_data_path)
        content_test = test_data_df.iloc[:, 1]

        logger.info("start seg test data")
        content_test = seg_words(content_test)
        logger.info("complete seg test data")

        tokenizer = joblib.load(tokenizer_path)
        data_test = tokenizer.texts_to_sequences(content_test)
        test_data_format = pad_sequences(data_test, maxlen=max_len)
        joblib.dump(test_data_format, seg_testset_path)  # 保存分词、转化为索引后的测试集


def BuildXgbData():
    train_data_format = joblib.load(seg_trainset_path)
    validate_data_format = joblib.load(seg_validateset_path)
    test_data_format = joblib.load(seg_testset_path)
    build_xgboost_dataformat(train_data_format, validate_data_format, test_data_format,\
                             (xgb_trainset_path, xgb_validateset_path, xgb_testset_path))

def TrainModel(model_No):
    # ----------------------------model train---------------------------------
    train_data_df = load_data_from_csv(config.train_data_path)
    validate_data_df = load_data_from_csv(config.validate_data_path)
    train_data_format = joblib.load(seg_trainset_path)
    validate_data_format = joblib.load(seg_validateset_path)
    columns = train_data_df.columns.values.tolist()
    column = columns[model_No+1]                        # 训练1~20号模型，1号模型对应column=2的label

    logger.info("start train %s model" % column)

    if not os.path.exists(config.model_path):
        os.makedirs(config.model_path)

    if train_model == XGboost:
        xgb_train_format = xgb.DMatrix(xgb_trainset_path)
        train_label = label_transform(train_data_df[column])
        xgb_train_format.set_label(train_label)

        xgb_validate_format = xgb.DMatrix(xgb_validateset_path)
        validate_label = label_transform(validate_data_df[column])
        xgb_validate_format.set_label(validate_label)

        xgb_model = train_model()
        xgb_model.fit(xgb_train_format, xgb_validate_format)
        xgb_model.save_model(config.model_path+model_name+'_%s'%column)

        preds = label_transform(xgb_model.predict(xgb_validate_format), reverse=True)
        val_f1 = f1_score(validate_label, preds, average='macro')
        val_precision = precision_score(validate_label, preds, average='macro')
        val_recall = recall_score(validate_label, preds, average='macro')
        print('-val_f1: %.4f --val_precision: %.4f --val_recall: %.4f' % (val_f1, val_precision, val_recall))

    elif train_model == HasSentimentBinary:
        tokenizer = joblib.load(tokenizer_path)
        max_words = len(tokenizer.word_index)
        train_label = np.asarray([(0 if y==-2 else 1) for y in train_data_df[column]])
        validate_label = np.asarray([(0 if y == -2 else 1) for y in validate_data_df[column]])

        early_stopping = EarlyStopping(monitor='val_loss', patience=5)      # 过拟合防控
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=3)       # 学习率衰减
        checkPoint = ModelCheckpoint(filepath=config.model_path+model_name+'_%s'%column, monitor='val_loss', save_best_only=True)
        f1_metrics = F1_Metrics(validate_data_format, validate_label)   # 计算在验证集上的F1
        callbacks = [early_stopping, reduce_lr, checkPoint, f1_metrics]

        textCNN_model = train_model(max_words+1, embedding_dim=embedding_dim, maxlen=max_len)
        textCNN_model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['acc'])
        textCNN_model.fit(train_data_format, train_label, epochs=epochs, batch_size=batch_size,\
                                 shuffle=True, validation_data=(validate_data_format, validate_label),callbacks=callbacks)

    else:
        tokenizer = joblib.load(tokenizer_path)
        max_words = len(tokenizer.word_index)

        if num_class == 3:
            train_index = np.asarray(train_data_df[column][train_data_df[column]!=-2].index)  # 只使用标签不等于-2的样本训练
            train_data_format = np.asarray([train_data_format[i] for i in train_index])
            train_label = np.asarray([train_data_df[column][i] for i in train_index])
            train_label = label_transform(train_label, map_dict={0:0,1:1,-1:2})

            validate_index = np.asarray(validate_data_df[column][validate_data_df[column]!=-2].index)  # 只使用标签不等于-2的样本训练
            validate_data_format = np.asarray([validate_data_format[i] for i in validate_index])
            validate_label = np.asarray([validate_data_df[column][i] for i in validate_index])
            validate_label = label_transform(validate_label, map_dict={0:0,1:1,-1:2})
        elif num_class == 4:
            train_label = label_transform(train_data_df[column], map_dict={0:0,1:1,-1:2,-2:3})
            validate_label = label_transform(validate_data_df[column], map_dict={0:0,1:1,-1:2,-2:3})

        sample_weights = sample_balance_weights(train_label)
        train_label = label2onehot(train_label)   # 标量label转成onehot
        validate_label = label2onehot(validate_label)

        early_stopping = EarlyStopping(monitor='val_loss', patience=5)      # 过拟合防控
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=3)       # 学习率衰减
        checkPoint = ModelCheckpoint(filepath=config.model_path+model_name+'_%s'%column, monitor='val_loss', save_best_only=True)
        f1_metrics = F1_Metrics(validate_data_format, onehot2label(validate_label))   # 计算在验证集上的F1
        callbacks = [early_stopping, reduce_lr, checkPoint, f1_metrics]

        textCNN_model = train_model(max_words+1, embedding_dim=embedding_dim, maxlen=max_len)
        textCNN_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
        textCNN_model.fit(train_data_format, train_label, epochs=epochs, batch_size=batch_size, sample_weight=sample_weights,\
                                 shuffle=True, validation_data=(validate_data_format, validate_label),callbacks=callbacks)

    logger.info("complete train %s model" % column)

    # logger.info("start save %s model"%column)
    # model_path = config.model_path
    # if not os.path.exists(model_path):
    #     os.makedirs(model_path)
    # textCNN_model.save(model_path+model_name+'_%s'%column)    # 保存模型
    # logger.info("complete save %s model"%column)


def compute_F1score(validation_No):
    # ----------------------------validation---------------------------------
    validate_data_df = load_data_from_csv(config.validate_data_path)
    validate_data_format = joblib.load(seg_validateset_path)
    columns = validate_data_df.columns.values.tolist()

    logger.info("start compute f1 score for %s model" % columns[validation_No+1])

    column = columns[validation_No+1]
    classifier = load_model(config.model_path+model_name+'_%s'%column)

    # pred_label = []
    # valid_batch = 10000          # 预测时限制每轮样本数
    # for i in range(int((len(validate_data_format)-1)/valid_batch)+1):
    #     tail = (i+1)*valid_batch
    #     if tail >= len(validate_data_format):                    # 最后一批验证样本
    #         tail = None
    #     label = classifier.predict(validate_data_format[i*valid_batch:tail])
    #     if label.shape[1]>1:
    #         pred_label+=onehot2label(label).tolist()
    #     else:
    #         pred_label+=[(0 if y[0]<sigmoid_threshold else 1) for y in label]

    true_label = np.asarray(validate_data_df[column])

    if train_model == HasSentimentBinary:
        label = classifier.predict(validate_data_format)
        pred_label=[(0 if y[0]<sigmoid_threshold else 1) for y in label]
        true_label=np.asarray([(0 if y==-2 else 1) for y in true_label])
        aver_mode = 'binary'
    elif num_class == 3:
        validate_index = np.asarray(validate_data_df[column][validate_data_df[column] != -2].index)  # 只使用标签不等于-2的样本训练
        validate_data_format = np.asarray([validate_data_format[i] for i in validate_index])
        pred_label= onehot2label(classifier.predict(validate_data_format))
        pred_label = label_transform(pred_label, map_dict={0:0,1:1,-1:2}, reverse=True)
        true_label = np.asarray([true_label[i] for i in validate_index])
        aver_mode = 'macro'
    elif num_class == 4:
        pred_label = onehot2label(classifier.predict(validate_data_format))
        pred_label = label_transform(pred_label, map_dict={0:0,1:1,-1:2,-2:3}, reverse=True)
        aver_mode = 'macro'

    pred_label = np.asarray(pred_label)
    f1 = f1_score(true_label, pred_label, average = aver_mode)
    precision = precision_score(true_label, pred_label, average = aver_mode)
    recall = recall_score(true_label, pred_label, average = aver_mode)
    if validation_No == 1:
        with open(F1score_path, 'w') as f:    # 重写score文件
            f.write(column+',%s,%s,%s'%(str(f1),str(precision),str(recall))+'\n')
    else:
        with open(F1score_path, 'a+') as f:   # 追加F1score
            f.write(column + ',%s,%s,%s' % (str(f1), str(precision), str(recall)) + '\n')
    if validation_No == 20:                        # 计算完最后模型的F1score,输出
        f1_score_dict = dict()
        precision_score_dict = dict()
        recall_score_dict = dict()
        scores = pd.read_csv(F1score_path, header=None)
        s_f1 = scores[1].values
        s_p = scores[2].values
        s_r = scores[3].values
        for i in range(20):
            f1_score_dict[columns[i+2]] = s_f1[i]
            precision_score_dict[columns[i+2]] = s_p[i]
            recall_score_dict[columns[i+2]] = s_r[i]

        f1 = np.mean(list(f1_score_dict.values()))
        precision = np.mean(list(precision_score_dict.values()))
        recall = np.mean(list(recall_score_dict.values()))
        with open(F1score_path, 'a+') as f:   # 写入平均F1score
            f.write('\n'+'F1_score,%s,%s,%s'%(str(f1),str(precision),str(recall))+'\n')
        str_score = "\n"
        for column in columns[2:end_col]:
            str_score += column + ":" + str(f1_score_dict[column]) + "\n"

        logger.info("f1_scores: %s\n" % str_score)
        logger.info("f1_score: %s" % f1)
        logger.info("complete compute f1 score for validate model")

if __name__ == '__main__':
    if args.build_dataset != 0:
        PrepareData()
    if args.build_xgb_dataset == 1:
        BuildXgbData()
    if args.model_No>=1 and args.model_No<=20:
        TrainModel(args.model_No)
    if args.validation_No>=1 and args.validation_No<=20:
        compute_F1score(args.validation_No)



