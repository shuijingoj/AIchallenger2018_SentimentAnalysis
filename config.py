#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

data_path = os.path.abspath('..') + "/Dataset"
model_path = '/media/ouchiye/文档/UbuntuFile/AwsomeTextCNN/'#data_path + "/model/"
train_data_path = data_path + "/ai_challenger_sentiment_analysis_trainingset_20180816/sentiment_analysis_trainingset.csv"
validate_data_path = data_path + "/ai_challenger_sentiment_analysis_validationset_20180816/sentiment_analysis_validationset.csv"
test_data_path = data_path + "/ai_challenger_sentimetn_analysis_testb_20180816/sentiment_analysis_testb.csv"
test_data_predict_output_path = data_path + "/predict/fastText_testa.csv"
