
# coding: utf-8

# # kerasに慣れよう2(Mnist認識)
# Mnist：28×28=784次元(各次元$\in\{0,1\}$)の正解ラベル付き手書き数字画像  
# よく機械学習のチュートリアルで使われる  
# 参考:[無から始めるKeras 第5回][1]
# [1]:https://qiita.com/Ishotihadus/items/b171272b954147976bfc

#  <div style="text-align: center;">
#   Mnist：手書き数字画像(100枚分)
# </div>
# ![MNIST](http://europa:37564/tree/keras_zemi/mnist.png)

# ## ライブラリなどの準備

# In[1]:


from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout

#GPUの仕様に関する設定
import tensorflow as tf
from keras import backend as K
config = tf.ConfigProto(
    gpu_options = tf.GPUOptions(
        visible_device_list = "0",
        per_process_gpu_memory_fraction = 0.3))
sess = tf.Session(config=config)
K.set_session(sess)


# ## Mnist読み込み

# In[2]:


(x_train, y_train), (x_test, y_test) = mnist.load_data()
#{0,255}なので{0,1}に正規化&データの構造変更
print("変更前")
print(x_train.shape)
x_train = x_train.reshape(60000, 784)/255.0
print("変更後")
print(x_train.shape)
x_test = x_test.reshape(10000, 784)/255.0
#ラベルのonehot化
print("読み込んだラベル")
print(y_train[:3])
y_train = np_utils.to_categorical(y_train, num_classes = 10)
print("onehot化したラベル")
print(y_train[:3])


# ## モデルの作成

# In[3]:


model = Sequential([
    Dense(1300, input_dim = 784, activation='relu'),
    Dropout(0.4),
    Dense(10, activation='softmax')
])

model.summary()


# ## 学習

# In[4]:


model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, batch_size = 100, validation_split=0.2, epochs=50)


# ## 識別

# In[5]:


predict = model.predict_classes(x_test)
print(sum(predict == y_test)/ 10000.0)

