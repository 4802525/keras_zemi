
# coding: utf-8

# # kerasに慣れよう1
# 入力ベクトル$(x_{1},x_{2},x_{3},x_{4},x_{5})$ ($x_i \in [0,1]$)の各要素の加算結果が2.5以上で1,未満で0を出すモデル  
# $f(x) = if\ \sum^5_{i=1}x_i \ \geq\ 2.5 \ then\  1\  else\  0$  
# 参考:[無から始めるKeras 第1回][1]
# [1]:https://qiita.com/Ishotihadus/items/c2f864c0cde3d17b7efb

# ## ライブラリなどの準備

# In[1]:


#ライブラリ
import numpy as np
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation

#GPUの仕様に関する設定
import tensorflow as tf
from keras import backend as K
config = tf.ConfigProto(
    gpu_options = tf.GPUOptions(
        visible_device_list = "0",
        allow_growth = True,
        per_process_gpu_memory_fraction = 0.1))
sess = tf.Session(config=config)
K.set_session(sess)


# ## データセットの生成

# In[2]:


#学習データセット
data = np.random.rand(10000,5) #1000個の5次元ベクトル
labels = (np.sum(data, axis=1) > 2.5) * 1 #ラベル(0,1)
labels = np_utils.to_categorical(labels)  #ラベル(onehot)
#onehotラベルはこんな感じ
#学習データ
print("学習データ")
print(data[:3])
print("正解(onehot)")
print(labels[:3])


# ## ネットワークの構築
# kerasでは層を積み上げていくイメージ  
# model.add以外にもこんな書き方もできる
# ```
# model = Sequential([
#     Dense(20, input_dim=5, activation='relu'),
#     Dense(2,activation='softmax')
# ])
# ```
# Functional APIを使った記法  
# 複雑なモデルを作るときに使用，最初はこんなのあるんだ見たいな気持ちでOK
# ```
# from keras.layers import Input, Dense, Activation
# from keras.models import Model
# inputs = Input(shape=(5,))
# x = Dense(20)(inputs)
# x = Activation('relu')(x)
# outputs = Dense(2,activation='softmax')(x)
# model = Model(inputs=inputs, outputs = outputs)
# ```
# ### 活性化関数
# - sigmoid
# - relu
# - tanh  
# 
# ### バッチ正規化層
# ミニバッチ内での直前の出力を正規化する
# - 学習を早く進行できる
# - 初期値に依存しない
# - 過学習抑制
# ```
# from keras.layers.normalization import BatchNormalization
# ```

# In[3]:


#Sequential:層を積み上げる単純なモデル
model = Sequential()
model.add(Dense(20, input_dim=5))
model.add(Activation('relu'))
model.add(Dense(2, activation='softmax'))

#ネットワーク構造の出力
model.summary()


# ## 学習
# ### 最適化関数(optimizer)  
# - SGD(確率的勾配降下法):  
# 傾きに比例して移動する  
# - Momentam:  
# ボールが転がるイメージ  
# - AdaGrad:  
# 学習係数を下げていく(最初は大きく動かし，終盤は微調整)
# - Adam:  
# MomentumとAdaGradの融合    
# 
# 例(optimizerを設定する場合
# ```
# from keras import optimizers
# adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999,epsilon=None, decay=0.0, amsgrad=False)
# model.compile(optimizer = adam, 'categorical_crossentropy', metrics=['accuracy'])
# ```
# ### 損失関数(loss)
# - mean_squared_errer:二乗平均誤差
# - mean_absolute_error:絶対平均誤差
# - hinge:マイナスのところは0,プラスのところはその値
# - categorical_crossentropy:複数クラスの交差エントロピー

# In[4]:


#compli:最適化関数,損失関数,評価指標
model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])
#学習
#fit;入力，出力，バッチサイズ(一度の更新に使用するデータ数), エポック数(学習回数),検証セットの割合(ランダムではないので注意)
model.fit(data, labels, batch_size = 100, epochs=150, validation_split=0.2)


# ## 未知データの予測

# In[5]:


#評価データセット
test_data = np.random.rand(1000,5)
test_label = (np.sum(test_data, axis=1) > 2.5) * 1 


# In[6]:


print("入力")
print(test_data[0])
print("正解")
print(test_label[0])
print("予測結果の確率(モデルの出力)")
#出力結果の確率
print(model.predict(test_data[0:1]))
print("確率が最大の次元")
#最も大きい次元を出力とする
print(np.argmax(model.predict(test_data[0:1])))


# In[7]:


predict = np.argmax(model.predict(test_data), axis=1)
#predictの中身
print("各データに対する予測")
print(predict[:3])
print("正解と予測の比較")
print((predict==test_label)[:3])
#sumを取るとtrueの数を集計する
print("認識率")
print(sum(predict == test_label) /1000)


# ## モデルの中身(重みとバイアスを取得)

# In[8]:


weight = model.get_weights()
print("hidden weight")
print(weight[0]) #隠れ層重み
print("hidden bias")
print(weight[1]) #隠れ層バイアス
print("output weight")
print(weight[2]) #出力層重み
print("output bias")
print(weight[3]) #出力層バイアス

