
# coding: utf-8

# # kerasを使って話者識別モデルを作ろう
# データベース：  
# 科警研DB 話者99人  
# 発話内容 ATR音素バランス文A50文  
#   
# 実験：  
# フレームレベルの認識  
# 発話内容open  
#  A01-A05の5文で学習，A06-A10の5文で検証，A11-A50の40文で評価  
# 特徴量  
#  対数MFB40bin×7フレーム

# ## ライブラリなどの準備

# In[1]:


#ライブラリ群
import argparse
import numpy as np
from numpy.random import *
from tqdm import tqdm
import keras
from keras.models import Sequential,load_model
from keras.initializers import TruncatedNormal
from keras.layers import Activation, Dense, Dropout
from keras.utils import np_utils
import tensorflow as tf
from keras import backend as K
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')

#プログラムの引数
parser = argparse.ArgumentParser(description="Speaker Varification")

parser.add_argument( '--gpu', '-g', default = "0", type = str,
                    help = 'GPU ID')
parser.add_argument( '--gpu_use', '-gu', default = 0.3, type = float,
                    help = 'GPU use rate')
parser.add_argument( '--list_directory', '-ld', default="/net/venus/research3/B4system2019/keras_zemi/kakeiken/list",
                    help = 'list directory')
parser.add_argument( '--learn_input', '-li', default="mfb40_learn.lst",
                    help = 'learn set')
parser.add_argument( '--valid_input', '-vi', default="mfb40_valid.lst",
                    help = 'valid set')
parser.add_argument( '--test_input', '-ti', default="mfb40_test.lst",
                    help = 'test set')
parser.add_argument( '--input_node', '-in', default=40, type=int,
                    help = 'number of input node')
parser.add_argument( '--num_frame', '-nf', default=7, type=int,
                    help = 'number of input frame')
parser.add_argument( '--output_node', '-on', default=99, type=int,
                    help = 'number of output node')
parser.add_argument( '--batch', '-b', default=1000, type=int,
                    help = 'learning minibatch size')
parser.add_argument( '--epoch', '-e', default=100, type=int,
                    help = 'number of epochs to learn')
parser.add_argument( '--savemodel', '-sm', default='none',
                    help = 'model file')
parser.add_argument( '--loadmodel', '-lm', default='none',
                    help = 'model file')
args = parser.parse_args()

#parserの代わり
#class Argument:
#  gpu = "0"
#  gpu_use=0.3
#  list_directory = "/net/venus/research3/B4system2019/keras_zemi/kakeiken/list"
#  learn_input = "mfb40_learn.lst"
#  valid_input = "mfb40_valid.lst"
#  test_input = "mfb40_test.lst"
#  input_node = 40
#  num_frame = 7
#  output_node = 99
#  batch = 1000
#  epoch = 100
#  savemodel = 'none'
#  loadmodel = 'none'

#args = Argument

config = tf.ConfigProto(
  gpu_options = tf.GPUOptions(
    visible_device_list=args.gpu,
    per_process_gpu_memory_fraction=args.gpu_use
  )
)
sess = tf.Session(config=config)
K.set_session(sess)

batchsize = args.batch
n_input = args.input_node
n_frame = args.num_frame
n_output = args.output_node
n_epoch = args.epoch


# ## 関数の準備
# ### file_read:リストファイルを読み込んでラベルと特徴量を返す
# labels:1フレームごとのone-hotラベル  
# features:1フレームごとの特徴量  
# ### nomalization:正規化処理を行う
# 各次元で学習データの平均分散を計算し，それを用いて正規化(標準化)を行う．

# In[2]:


def file_read(lstfile):
  features_lst=[]
  labels_lst=[]
    
  with open(lstfile) as fp:
    for data in tqdm(fp):
      tmp = np.load(data.rstrip())
      #前後無音が入っているのでその部分は除去
      x = tmp['feature'][10:-40]
      y = tmp['label'][10:-40]
      i = 0
        
      #n_frame結合し，リストに格納
      for i in range(len(x) - n_frame):
        x_ = x[i:i+n_frame].reshape(n_input*n_frame)
        features_lst.append(x_)
        labels_lst.append(y[0])

  labels = np.array(np_utils.to_categorical(labels_lst), dtype = np.int32)
  features = np.array(features_lst, dtype = np.float32)
  return labels, features

def nomalization(features, mean, std):
  features=(features-mean)/std
  return features


# ## データ読み込み

# In[3]:


#学習データ
print("learn data loding")
lstfile = args.list_directory + "/" + args.learn_input
learn_labels, learn_features = file_read(lstfile)
mean = np.mean(learn_features,axis=0)
std = np.std(learn_features,axis=0)
learn_features = nomalization(learn_features, mean, std)

#検証データ
print("valid data loding")
lstfile = args.list_directory + "/" + args.valid_input
valid_labels, valid_features = file_read(lstfile)
valid_features = nomalization(valid_features, mean, std)

#評価データ
print("test data loding")
lstfile = args.list_directory + "/" + args.test_input
test_labels, test_features = file_read(lstfile)
test_features = nomalization(test_features, mean, std)

print(np.shape(learn_labels),np.shape(learn_features))


# ## モデルの構築

# In[4]:


TN = TruncatedNormal(seed=0) #切断正規分布

#自分で作ってみよう(input_dim=n_input*n_frame)
if args.loadmodel != 'none':
  model = load_model(args.loadmodel)
else:
  model = Sequential([
    Dense(700, input_dim=n_input*n_frame, kernel_initializer=TN),
    Activation('relu'),
    Dropout(0.2),
    Dense(400, kernel_initializer=TN),
    Activation('relu'),
    Dropout(0.2),
    Dense(100, kernel_initializer=TN),
    Activation('relu'),
    Dropout(0.2),
    Dense(n_output, activation='softmax')
  ])
model.summary()


# ## 学習

# In[5]:


if args.loadmodel == 'none':
  model.compile(loss='categorical_crossentropy', 
                optimizer='adam', metrics=['accuracy'])
  hist = model.fit(x=learn_features, y=learn_labels,
            batch_size=batchsize, epochs=n_epoch,
            validation_data=(valid_features,valid_labels))


# ## モデルの保存と学習曲線の表示

# In[6]:


#モデルの保存
if args.savemodel != 'none':
  model.save(args.savemodel)

#if args.loadmodel == 'none':
#  loss
#  loss = hist.history["loss"]
#  val_loss = hist.history["val_loss"]
#  fig = plt.figure()
#  plt.plot(range(len(loss)), loss, "bo", color="r", label="Training loss")
#  plt.plot(range(len(val_loss)), val_loss, "bo", color="b", label="validing loss")
#  plt.xlabel("epochs")
#  plt.title("loss")
#
#  acc
#  acc = hist.history["acc"]
#  val_acc = hist.history["val_acc"]
#  fig = plt.figure()
#  plt.ylim(0,1)
#  plt.plot(range(len(acc)), acc, "bo", color="r", label="Training acc")
#  plt.plot(range(len(val_acc)), val_acc, "bo", color="b", label="validing acc")
#  plt.xlabel("epochs")
#  plt.title("acc")


# ## フレームレベルの評価

# In[7]:


#学習データ
predict = model.predict_classes(learn_features)
print("trainset error rate")
print(sum(predict != np.argmax(learn_labels, axis=1))/ len(learn_labels)*100)

predict = model.predict_classes(valid_features)
print("validset error rate")
print(sum(predict != np.argmax(valid_labels, axis=1))/ len(valid_labels)*100)

predict = model.predict_classes(test_features)
print("testset error rate")
print(sum(predict != np.argmax(test_labels, axis=1))/ len(test_labels)*100)

