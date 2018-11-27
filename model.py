#-*-coding:utf-8-*-
from keras.models import Sequential, Model, load_model
from keras.layers import Input, Embedding, Dropout, Dense, concatenate, Activation,Add
from keras.layers import Conv1D, MaxPooling1D, AveragePooling1D, GlobalMaxPooling1D, GlobalAveragePooling1D, Flatten
from keras.layers import GRU, Lambda
from keras.layers.normalization import BatchNormalization
from keras import regularizers
import xgboost as xgb
import tensorflow as tf
from gensim.models import Word2Vec

def HasSentimentBinary(max_words=1000, embedding_dim=350, maxlen=300):
	# 判断是否存在情感倾向的二分类模型，先行剔除不存在相关情感评论的样本
	inputs = Input(shape=(maxlen, ))
	x = Embedding(max_words, embedding_dim, input_length=maxlen)(inputs)
	conv1 = Conv1D(100, 5, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)            # 用三种不同size的卷积核提取特征
	conv2 = Conv1D(100, 7, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
	conv3 = Conv1D(100, 9, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
	conv_out = concatenate([conv1, conv2, conv3], axis=1)   # 三个卷积层的输出拼接
	conv_out = BatchNormalization()(conv_out)

	x = GlobalMaxPooling1D()(conv_out)
	x = Dropout(0.5)(x)
	x = Dense(1, activation='sigmoid')(x)
	model = Model(input=inputs, outputs=x)
	return model

def FastText(max_words=1000, embedding_dim=350, maxlen=300):
	model = Sequential()
	model.add(Embedding(max_words, embedding_dim, input_length=maxlen))
	model.add(GlobalAveragePooling1D())
	model.add(Dense(64))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(Dense(4, activation='softmax'))
	return model

def GensimTextCNN(max_words=1000, embedding_dim=350, maxlen=300):
	inputs = Input(shape=(maxlen, ))
	wv_model = Word2Vec.load('tempData/gensim_word2vec')
	x = wv_model.wv.get_keras_embedding(train_embeddings=False)(inputs)
	conv1 = Conv1D(100, 5, kernel_regularizer=regularizers.l2(0.01))(x)            # 用三种不同size的卷积核提取特征
	conv2 = Conv1D(100, 7, kernel_regularizer=regularizers.l2(0.01))(x)
	conv3 = Conv1D(100, 9, kernel_regularizer=regularizers.l2(0.01))(x)
	conv_out = concatenate([conv1, conv2, conv3], axis=1)   # 三个卷积层的输出拼接
	conv_out = BatchNormalization()(conv_out)
	conv_out = Activation('relu')(conv_out)

	x = GlobalMaxPooling1D()(conv_out)
	x = Dropout(0.5)(x)
	x = Dense(64)(x)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)
	x = Dense(4, activation='softmax')(x)
	model = Model(input=inputs, outputs=x)
	return model

def KMaxPooling(inputs, k):
	return Lambda(lambda x: tf.reshape(tf.nn.top_k(x,k=k)[0], shape=(-1,k*inputs.shape[1])))(inputs)

def AwsomeTextCNN(max_words=1000, embedding_dim=350, maxlen=300):
	inputs = Input(shape=(maxlen, ))
	x = Embedding(max_words, embedding_dim, input_length=maxlen, embeddings_regularizer=regularizers.l2(0.01))(inputs)
	conv1 = Conv1D(100, 5, kernel_regularizer=regularizers.l2(0.01))(x)            # 用三种不同size的卷积核提取特征
	conv2 = Conv1D(100, 7, kernel_regularizer=regularizers.l2(0.01))(x)
	conv3 = Conv1D(100, 9, kernel_regularizer=regularizers.l2(0.01))(x)
	conv_out = concatenate([conv1, conv2, conv3], axis=1)   # 三个卷积层的输出拼接
	conv_out = BatchNormalization()(conv_out)
	conv_out = Activation('relu')(conv_out)

	# x = KMaxPooling(conv_out, 3)
	x = GlobalMaxPooling1D()(conv_out)
	x = Dropout(0.5)(x)
	x = Dense(64)(x)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)
	x = Dense(4, activation='softmax')(x)
	model = Model(input=inputs, outputs=x)
	return model

def TextRNN(max_words=1000, embedding_dim=350, maxlen=300):
	inputs = Input(shape=(maxlen, ))
	x = Embedding(max_words, embedding_dim, input_length=maxlen, embeddings_regularizer=regularizers.l2(0.01),\
	              mask_zero=True)(inputs)

	GRUout = GRU(128,implementation=2)(x)
	reversed_GRUout = GRU(128, go_backwards=True,implementation=2)(x)
	y = concatenate([GRUout, reversed_GRUout])
	y = Dense(64)(y)
	y = BatchNormalization()(y)
	y = Activation('relu')(y)
	y = Dense(4, activation='softmax')(y)
	model = Model(input=inputs, outputs=y)
	return model

def TextCNN_4classes(max_words=1000, embedding_dim=350, maxlen=300):
	inputs = Input(shape=(maxlen, ))
	x = Embedding(max_words, embedding_dim, input_length=maxlen)(inputs)
	conv1 = Conv1D(100, 5, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)            # 用三种不同size的卷积核提取特征
	conv2 = Conv1D(100, 7, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
	conv3 = Conv1D(100, 9, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
	conv_out = concatenate([conv1, conv2, conv3], axis=1)   # 三个卷积层的输出拼接
	conv_out = BatchNormalization()(conv_out)

	x = GlobalMaxPooling1D()(conv_out)
	x = Dropout(0.5)(x)
	x = Dense(4, activation='softmax')(x)
	model = Model(input=inputs, outputs=x)
	return model

def TextCNN_3classes(max_words=1000, embedding_dim=350, maxlen=300):
	inputs = Input(shape=(maxlen, ))
	x = Embedding(max_words, embedding_dim, input_length=maxlen)(inputs)
	conv1 = Conv1D(100, 5, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)            # 用三种不同size的卷积核提取特征
	conv2 = Conv1D(100, 7, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
	conv3 = Conv1D(100, 9, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
	conv_out = concatenate([conv1, conv2, conv3], axis=1)   # 三个卷积层的输出拼接
	conv_out = BatchNormalization()(conv_out)

	x = GlobalMaxPooling1D()(conv_out)
	x = Dropout(0.5)(x)
	x = Dense(3, activation='softmax')(x)
	model = Model(input=inputs, outputs=x)
	return model

def DPCNN(max_words=1000, embedding_dim=350, maxlen=300):

	channel_size = 250
	kernel_size = 3

	def block(x, channel_size, kernel_size):
		x = MaxPooling1D(pool_size=3 , strides=2)(x)
		x_shortcut = x
		x = Conv1D(channel_size, kernel_size, padding='same', kernel_regularizer=regularizers.l2(0.01))(x)
		x = BatchNormalization()(x)
		x = Activation('relu')(x)
		x = Conv1D(channel_size, kernel_size, padding='same', kernel_regularizer=regularizers.l2(0.01))(x)
		x = BatchNormalization()(x)
		x = Activation('relu')(x)
		x = Add()([x,x_shortcut])       # 卷积前后的结果相加作为下一层的输入
		return x

	inputs = Input(shape=(maxlen, ))
	x = Embedding(max_words, embedding_dim, input_length=maxlen)(inputs)

	x_shortcut = Conv1D(channel_size, 1)(x)    # 改变维度的卷积
	x = Conv1D(channel_size, kernel_size, padding='same', kernel_regularizer=regularizers.l2(0.01))(x)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)
	x = Conv1D(channel_size, kernel_size, padding='same', kernel_regularizer=regularizers.l2(0.01))(x)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)
	x = Add()([x,x_shortcut])           # 卷积前后的结果相加作为下一层的输入

	block_num = 2              # 添加任何数量的block
	for i in range(block_num):
		x = block(x, channel_size,kernel_size)

	x = GlobalMaxPooling1D(x)
	x = Dropout(0.5)(x)
	y = Dense(4, activation='softmax')(x)

	model = Model(input=inputs, outputs=y)
	return model

class XGboost():
	def __init__(self, load_path = None):
		self.params = {'booster': 'gbtree',
		         'objective': 'multi:softmax',
		         'num_class': 4,                # 类数，与 multisoftmax 并用
		         'max_depth': 60,
		         'eta': 0.4,
				 'eval_metric': 'merror',
				 'silent': 1}
		if load_path == None:
			self.model = None
		else:
			self.model = xgb.Booster(model_file=load_path)

	def fit(self,train_set, validate_set):
		evallist = [(train_set, 'train'), (validate_set, 'validation')]
		num_round = 200 # 开始训练
		print("开始训练xgboost！")
		self.model = xgb.train(self.params, train_set, num_round, evallist, early_stopping_rounds=10)

	def save_model(self, path):
		if self.model != None:
			self.model.save_model(path)
		else:
			raise TypeError('xgboost model does not exist!')

	def load_model(self, path):
		self.model = xgb.Booster(model_file=path)

	def predict(self, data_set):
		result = self.model.predict(data_set)
		return result

a = AwsomeTextCNN()
a.summary()