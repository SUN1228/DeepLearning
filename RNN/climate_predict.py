'''
进阶循环神经网络用法
1.Recurrent dropout
2.Stacking recurrent layers
3.Bidirectional recurrent layers 
'''

import os
import numpy as np
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop

fname='E:/DataSet/jena_climate/jena_climate_2009_2016.csv'
f=open(fname)
data=f.read()
f.close()

lines=data.split('\n')     #数据行数
header=lines[0].split(',') #属性名

#Parsing the data
float_data=np.zeros((len(lines),len(header)-1))
for i,line in enumerate(lines): #用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标
    values=[float(x) for x in line.split(',')[1:]]
    float_data[i,:]=values

#Plotting the temperature timeseries
#temp=float_data[:,1]
#plt.plot(range(len(temp),temp))

#Plotting the first 10 days of the temperature timeseries
#plt.plot(range(1440),temp[:1440]) #data recorded every 10 mins

#Normalizing the data
mean=float_data[:200000].mean(axis=0)
float_data-=mean
std=float_data[:200000].std(axis=0)
float_data/=std

#Generator yielding timeseries sample and their target
def generator(data,lookback,delay,min_index,max_index,shuffle=False,batch_size=128,step=6):
    '''
    data 原始浮点数组
    lookback 输入数据查看历史数据的长度
    delay 预测将来数据的长度
    min_index|max_index 数组的索引
    shuffle 是否打乱顺序
    batch_size 批量大小
    step 对数据进行采样的时间段
    '''
    if max_index is None:
        max_index=len(data)-delay-1
    i=min_index+lookback
    while 1:
        if shuffle:
            rows=np.random.randint(min_index+lookback,max_index,size=batch_size)
        else:
            if i+batch_size>=max_index:
                i=min_index+lookback
            rows=np.arange(i,min(i+batch_size,max_index))
            i+=len(rows)
        samples=np.zeros((len(rows),lookback//step,data.shape[-1]))
        targets=np.zeros((len(rows),))
        for j,row in enumerate(rows):
            indices=range(rows[j]-lookback,rows[j],step)
            samples[j]=data[indices]
            targets[j]=data[rows[j]+delay][1]
        yield samples,targets #把函数变成generator
    
#Preparing the training validation and test generator
lookback=1440 #回看前10天
step=6        #间隔一小时
delay=144     #预测一天后
batch_size=128

train_gen=generator(float_data,
                    lookback=lookback,
                    delay=delay,
                    min_index=0,
                    max_index=200000,
                    shuffle=True,
                    step=step,
                    batch_size=batch_size)
val_gen=generator(float_data,
                lookback=lookback,
                delay=delay,
                min_index=200001,
                max_index=300000,
                step=step,
                batch_size=batch_size)
test_gen=generator(float_data,
                lookback=lookback,
                delay=delay,
                min_index=300001,
                max_index=None,
                step=step,
                batch_size=batch_size)

val_steps=(300000-200001-lookback)
test_steps=(len(float_data)-300001-lookback)

#使用GRU网络层
model=Sequential()
model.add(layers.GRU(32,input_shape=(None,float_data.shape[-1])))
model.add(layers.Dense(1))

model.compile(optimizer=RMSprop(),loss='mae')
history=model.fit_generator(train_gen,
                            steps_per_epoch=500,
                            epochs=20,
                            validation_data=val_gen,
                            validation_steps=val_steps)

'''
Dropout
dropout针对输入
recurrent_dropout针对循环状态的线性转换
'''
model1=Sequential()
model1.add(layers.GRU(32,dropout=0.2,recurrent_dropout=0.2,input_shape=(None,float_data.shape[-1])))
model1.add(layers.Dense(1))

model1.compile(optimizer=RMSprop(),loss='mae')
history1=model1.fit_generator(train_gen,
                            steps_per_epoch=500,
                            epochs=40,
                            validation_data=val_gen,
                            validation_steps=val_steps)


'''Stacking recurrent layers'''
model2=Sequential()
model2.add(layers.GRU(32,dropout=0.2,recurrent_dropout=0.2,return_sequences=True,input_shape=(None,float_data.shape[-1])))
model2.add(layers.GRU(64,activation='relu',dropout=0.1,recurrent_dropout=0.5))
model2.add(layers.Dense(1))

model2.compile(optimizer=RMSprop(),loss='mae')
history2=model2.fit_generator(train_gen,
                            steps_per_epoch=500,
                            epochs=40,
                            validation_data=val_gen,
                            validation_steps=val_steps)

'''bidirectional GRU'''
model3=Sequential()
model3.add(layers.Bidirectional(layers.GRU(32),input_shape=(None,float_data.shape[-1])))
model3.add(layers.Dense(1))

model3.compile(optimizer=RMSprop(),loss='mae')
history3=model3.fit_generator(train_gen,
                            steps_per_epoch=500,
                            epochs=40,
                            validation_data=val_gen,
                            validation_steps=val_steps)
