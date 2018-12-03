'''简单Word2Vec'''

import collections
import math
import os
import sys
import argparse
import random
from tempfile import gettempdir
import zipfile

import numpy as np 
from six.moves import urllib
from six.moves import xrange
import tensorflow as tf 

from tensorflow.contrib.tensorboard.plugins import projector

#设置tensorboard保存路径
current_path=os.path.dirname(os.path.realpath(sys.argv[0]))

parser=argparse.ArgumentParser()
parser.add_argument(
    '--log_dir',
    type=str,
    default=os.path.join(current_path,'log'),
    help='The log directory for tensorboard summaries.')
FLAGS,unparsed=parser.parse_known_args()

#为tensorboard变量创建文件夹
if not os.path.exists(FLAGS.log_dir):
    os.makedirs(FLAGS.log_dir)

#Step 1 下载数据
url='http://mattmahoney.net/dc/'

def maybe_download(filename,expected_bytes):
    local_filename=os.path.join(gettempdir(),filename)
    if not os.path.exists(local_filename):
        local_filename,_=urllib.request.urlretrieve(url+filename,local_filename)
    
    statinfo=os.stat(local_filename)
    if statinfo.st_size==expected_bytes:
        print('found and verified',filename)
    else:
        print(statinfo.st_size)
        raise Exception('failed to verify'+local_filename+'.can you get to it with browser')
    return local_filename

filename=maybe_download('text8.zip',31344016)

#把数据读取成一列字符串
def read_data(filename):
    '''extract the first file enclosed in a zip file as a list of words'''
    with zipfile.ZipFile(filename) as f:
        data=tf.compat.as_str(f.read(f.namelist()[0])).split()
    return data

vocabulary=read_data(filename)
print('data size=',len(vocabulary))

#Step 2 建造字典和将罕见单词替换成<UNK>
vocabulary_size=50000

def build_dataset(words,n_words):
    '''process raw input into dataset'''
    count=[['UNK',-1]]
    count.extend(collections.Counter(words).most_common(n_words-1)) #输出word和出现的次数
    dictionary=dict() #字典 生成word的索引index
    for word,_ in count:
        dictionary[word]=len(dictionary) 
    data=list()
    unk_count=0
    for word in words:
        index==dictionary.get(word,0) #函数返回指定键的值，如果值不在字典中返回默认值 这里默认值是0
        if index==0:
            unk_count+=1
        data.append(index) #tokenizer
    count[0][1]=unk_count
    reversed_dictionary=dict(zip(dictionary.values(),dictionary.keys()))
    return data,count,dictionary,reversed_dictionary
    #data 经过tokenizer处理的text
    #count 每个word的出现次数统计
    #dictionary words --> index
    #reversed_dictionary index-->words

data,count,dictionary,reversed_dictionary=build_dataset(vocabulary,vocabulary_size)
del vocabulary #为了节省内存
print('most common words(+UNK):',count[:5])
print('sample data',data[:10],[reversed_dictionary[i] for i in data[:10]])

data_index=0

#Step 3 为skip-gram模型生成训练批次的函数
def generate_batch(batch_size,num_skips,skip_window):
    '''
    batch_size: 序列长度
    num_skips: 跳接选择背景词的次数
    skip_window: 窗口大小
    '''
    global data_index #想要对全局变量作修改
    assert batch_size % num_skips==0 #断言
    assert num_skips <= 2*skip_window
    batch=np.ndarray(shape=(batch_size),dtype=np.int32)
    labels=np.ndarray(shape=(batch_size,1),dtype=np.int32)
    span=2*skip_window+1 #[skip_window target skip_window]
    buffer=collections.deque(maxlen=span)
    if data_index + span >len(data): #中心词到达结尾
        data_index=0 
    buffer.extend(data[data_index,data_index+span])
    data_index+=span
    for i in range(batch_size//num_skips):
        context_words=[w for w in range(span) if w!=skip_window]
        words_to_use=random.sample(context_words,num_skips) #从指定序列中随机获取指定长度的片断
        for j,context_word in enumerate(words_to_use): #what???
            batch[i*num_skips+j]=buffer[skip_window] #中心词
            labels[i*num_skips+j,0]=buffer[context_word] #背景词
        if data_index==len(data):
            buffer.extend(data[0:span])
            data_index=span
        else:
            buffer.append(data[data_index])
            data_index+=1
    #Backtrack a little bit to avoid skipping words in the end of a batch
    data_index=(data_index+len(data)-span)%len(data)
    return batch,labels

batch,labels=generate_batch(batch_size=8,num_skips=2,skip_window=1)

#Step 4 创建和训练skip-gram模型
batch_size=128
embedding_size=128
skip_window=1
num_skips=2 #how many times to reuse an input to generate a label
num_sampled=64 #number of negative example to sample 

valid_size=16 #random set of words to evaluate similarity
valid_window=100 #only pick dev samples in the head of the distribution
valid_examples=np.random.choice(valid_window,valid_size,replace=True)

graph=tf.Graph()

with graph.as_default():
    #Input data
    with tf.name_scope('inputs'):
        train_inputs=tf.placeholder(tf.int32,shape=[batch_size]) 
        train_labels=tf.placeholder(tf.int32,shape=[batch_size,1]) 
        valid_dataset=tf.constant(valid_examples,dtype=tf.int32)
    
    #ops and variables pinned to the cpu
    with tf.device('/cpu:0'):
        #look up embedding for input
        with tf.name_scope('embeddings'):
            embeddings=tf.Variable(tf.random_uniform([vocabulary_size,embedding_size],-1.0,1.0)) #初始化变量
            embed=tf.nn.embedding_lookup(embeddings,train_inputs) 

        #construct the variables for the NCE loss/噪声对比估计损失函数
        with tf.name_scope('weights'):
            nce_weights=tf.Variable(tf.truncated_normal(
                [vocabulary_size,embedding_size],
                stddev=1.0/math.sqrt(embedding_size)))
        with tf.name_scope('biases'):
            nce_biases=tf.Variable(tf.zeros[vocabulary_size])
    
    with tf.name_scope('loss'):
        loss=tf.reduce_mean(
            tf.nn.nce_loss(
                weights=nce_weights,
                biases=nce_biases,
                labels=train_labels,
                inputs=embed,
                num_sampled=num_sampled,
                num_classes=vocabulary_size))
    
    #add the loss value as a scalar to summary
    tf.summary.scalar('loss',loss)

    #SGD optimizer with learning rate of 1.0
    with tf.name_scope('optimizer'):
        optimizer=tf.train.GradientDescentOptimizer(1.0).minimize(loss)

    #compute the cosine similarity 衡量两个词汇的相似度
    norm=tf.sqrt(tf.reduce_sum(tf.square(embeddings),1,keepdims=True))
    normalized_embeddings=embeddings/norm
    valid_embeddings=tf.nn.embedding_lookup(normalized_embeddings,valid_dataset)
    similarity=tf.matmul(valid_embeddings,normalized_embeddings,transpose_b=True)
    
    #merge all summary
    merged=tf.summary.merge_all()

    #add variable initializer
    init=tf.global_variables_initializer()

    #create a saver
    saver=tf.train.Saver()

#Step 5 开始训练
num_steps=100001

with tf.Session(graph=graph) as session: 
    #open a writer to write summary
    writer=tf.summary.FileWriter(FLAGS.log_dir,session.graph)

    #initialize all variables
    init.run()

    average_loss=0
    for step in xrange(num_steps):
        batch_inputs,batch_labels=generate_batch(batch_size,num_skips,skip_window)
        feed_dict={train_inputs:batch_inputs,train_labels:batch_labels}

        #define metadata variable
        run_metadata=tf.RunMetadata()

        _,summary,loss_val=session.run(
            [optimizer,merged,loss],
            feed_dict=feed_dict,
            run_metadata=run_metadata)
        average_loss+=loss_val

        #add returned summary to writer in each step
        writer.add_summary(summary,step)
        #add metadata to visualize the graph for the last run
        if step==(num_step-1):
            writer.add_run_metadata(run_metadata,'step%d'%step)
        
        if step%2000==0:
            if step>0:
                average_loss/=2000
                # The average loss is an estimate of the loss over the last 2000 batches.
                print('Average loss at step ', step, ': ', average_loss)
            average_loss=0
        
        if step%10000==0:
            sim=similarity.eval()
            for i in range(valid_size):
                valid_word = reverse_dictionary[valid_examples[i]]
                top_k = 8  # number of nearest neighbors
                nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                log_str = 'Nearest to %s:' % valid_word
                for k in range(top_k):
                    close_word = reverse_dictionary[nearest[k]]
                    log_str = '%s %s,' % (log_str, close_word)
                print(log_str)
    final_embeddings=normalized_embeddings.eval()

    with open(FLAGS.log_dir+'/metadata.tsv','w') as f:
        for i in xrange(vocabulary_size):
            f.write(reversed_dictionary[i]+'\n')
    
    #save the model for checkpoint
    saver.save(session,os.path.join(FLAGS.log_dir,'model.ckpt'))

    #create a configuration for visualizing embeddings with the labels in TensorBoard
    config=projector.ProjectorConfig()
    embedding_conf=config.embeddings.add()
    embedding_conf.tensor_name=embeddings.name
    embedding_conf.metadata_path=os.path.join(FLAGS.log_dir,'metadata.tsv')
    projector.visualize_embeddings(writer,config)

writer.close()

    
