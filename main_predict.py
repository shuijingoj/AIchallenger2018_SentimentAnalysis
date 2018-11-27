#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import config
import logging
from sklearn.externals import joblib
from keras.models import load_model
from util import *

parser = argparse.ArgumentParser()
parser.add_argument('-mn', '--model_No', type=int, nargs='?',
                    default=0)
args = parser.parse_args()

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] <%(processName)s> (%(threadName)s) %(message)s')
logger = logging.getLogger(__name__)

model_name = 'AwsomeTextCNN'
seg_testset_path = 'tempData/seg_testsetb'
binary_result_path = 'result/result_binary.csv'
num_class = 4

if __name__ == '__main__':
    validate_data_df = load_data_from_csv(config.validate_data_path)
    columns = validate_data_df.columns.values.tolist()                  # 从验证集中获取列名
    test_data_format = joblib.load(seg_testset_path)

    column = columns[args.model_No+1]

    # load model
    logger.info("start load %s model" % column)
    classifier = load_model(config.model_path+model_name+'_%s'%column)

    # model predict
    logger.info("start predict %s model" % column)
    if num_class == 3:
        binary_label = load_data_from_csv(binary_result_path)[column]
        test_index = np.asarray(binary_label[binary_label==1].index)            # 仅取label=1的样本进行情感分类
        test_data_format_3classes = np.asarray([test_data_format[i] for i in test_index])
        output = classifier.predict(test_data_format_3classes)
        sentiment_predictions = label_transform(onehot2label(output), map_dict={0: 0, 1: 1, -1: 2}, reverse=True).tolist()

        predictions = []
        for i in range(len(test_data_format)):
            predictions.append(sentiment_predictions.pop(0) if i in test_index else -2)
        predictions = np.asarray(predictions)
    elif num_class == 4:
        output = classifier.predict(test_data_format)
        predictions = label_transform(onehot2label(output), map_dict={0:0,1:1,-1:2,-2:3}, reverse=True)
    else:
        output = classifier.predict(test_data_format)
        predictions = np.asarray([(0 if y[0] < sigmoid_threshold else 1) for y in output])

    if args.model_No == 1:              # 创建预测结果csv文件
        result_df = pd.DataFrame({columns[0]:[x for x in range(len(test_data_format))]})
        result_df[columns[1]] = ''       # 省略content
    else:
        result_df = load_data_from_csv('result/result.csv')
    result_df[column] = predictions
    result_df.to_csv('result/result.csv', index=False, encoding="utf-8")

    logger.info("complete %s predict" % column)

    if args.model_No == 20:
        logger.info("complete predict test data")
