
import config
from sklearn.externals import joblib
import numpy as np
import tensorflow as tf
from keras.utils import to_categorical
from keras.layers import Input, Lambda, Flatten, Layer
from keras.models import Model

def KMaxPooling(inputs, k):
	return Lambda(lambda x: tf.reshape(tf.nn.top_k(x,k=k)[0], shape=(-1,k*data.shape[1])))(inputs)

l = [[1,2,3,4,5,6],[11,22,33,44,55,66]]
data = np.reshape(np.asarray(l), (len(l),2,3))
print(data)

input = Input(shape=(2,3), dtype='int32')
la = KMaxPooling(input, 2)
model = Model(inputs=input, outputs=la)
pre = model.predict(data)
print(pre)

