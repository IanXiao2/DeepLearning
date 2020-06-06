# -*- coding: utf-8 -*-
# @Time    : 2020/6/4 4:59 下午
# @Author  : Ian
# @File    : data_process.py
# @Project : class01

from config import config
import numpy as np
import re
import jieba
import pandas as pd
from multiprocessing import cpu_count, Pool
from gensim.models.word2vec import Word2Vec, LineSentence
import torch
from sklearn.model_selection import train_test_split


def clean_sentence(sentence):
    '''
    特殊符号去除
    :param sentence: 待处理字符串
    :return: 过滤特殊字符后的字符串
    '''
    if isinstance(sentence, str):
        return re.sub(
            r'[\s+\-\!\/\|\[\]\{\}_,.$%^*(+\"\')]+|[:：+——()?【】“”！，。？、~@#￥%……&*（）]',
            '', sentence)
    else:
        return ' '

def load_stop_words(stop_words_path):
    '''
    加载停用词
    :param stop_words_path: 停用词路径
    :return: 停用词列表
    '''
    with open(stop_words_path, 'r', encoding='utf-8') as file:
        stop_words = [line.strip() for line in file.readlines()]

    return stop_words

def filter_stopwords(words):
    '''
    过滤停用词
    :param words: 切好词的列表
    :return: 过滤后的停用词
    '''
    stop_words = load_stop_words(config.stop_words_path)

    return [word for word in words if word not in stop_words]

def sentence_proc(sentence):
    '''
    预处理模块
    :param sentence: 待处理字符串
    :return: 处理后的字符串
    '''
    sentence = clean_sentence(sentence)
    # 切词，默认精确模式
    words = jieba.cut(sentence)
    # 过滤停用词
    words = filter_stopwords(words)
    # 拼接成一个字符串，按空格分隔
    return ' '.join(words)

def sentences_proc(df):
    '''
    数据集批量处理的方法
    :param df: 数据集
    :return: 处理好的数据集
    '''
    if 'review' in df.columns:
        df['review'] = df['review'].apply(sentence_proc)

    return df

def parallelize(df, func):
    '''
    多核并行处理模块
    :param df: DataFrame数据
    :param func: 预处理函数
    :return: 处理后的数据
    '''
    cores = cpu_count()
    partitions = cores
    # 数据切分
    data_split = np.array_split(df, partitions)
    # 进程池
    pool = Pool(cores)
    # 数据分发 合并
    data = pd.concat(pool.map(func, data_split))
    # 关闭线程池
    pool.close()
    # 执行完close后不会有新的进程加入到pool,join函数等待所有子进程结束
    pool.join()
    return data

def get_max_len(data):
    '''
    获得合适的最大长度值
    :param data: 待统计的数据
    :return: 合适的最大长度值
    '''
    max_lens = data.apply(lambda x: x.count(' ')) + 1
    return int(np.mean(max_lens) + 2 * np.std(max_lens))


def pad_proc(sentence, max_len, vocab):

    words = sentence.strip().split(' ')
    # 截取规定长度的词数
    words = words[:max_len]
    # 遍历判断词是否在vocab中，不在填充 <unk>
    sentence = [word if word in vocab else '<UNK>' for word in words]
    # 根据最大长度，填充 <PAD>
    sentence = sentence + ['<PAD>'] * (max_len - len(words))
    return ' '.join(sentence)

def transform_data(sentence, vocab):
    '''
    word 2 index
    :param sentence: [word1,word2,word3, ...] ---> [index1,index2,index3 ......]
    :param vocab: 词表
    :return: 转换后的序列
    '''
    words = sentence.split(' ')
    idxs = [vocab[word] for word in words]
    return idxs


def process_data():
    # 1. 加载数据
    data_df = pd.read_csv(config.data_path, encoding='utf-8')
    print('data size {}'.format(len(data_df)))
    # 检查是否存在空值
    #print(data_df[data_df.isnull().values == True])

    # 2. 多进程, 批量数据处理 (清理无用字符，分词，过滤停用词，拼接)
    data_df = parallelize(data_df, sentences_proc)
    data_df['review'].to_csv(config.data_review, index=False, header=False)

    # 3. 训练词向量
    print('start build w2v model')
    wv_model = Word2Vec(LineSentence(config.data_review), size=config.embedding_dim,
                        negative=5, workers=8, iter=config.wv_train_epochs,
                        window=3, min_count=5)

    # 4. 未知词填充，长度填充

    vocab = wv_model.wv.vocab
    # 获取适当的最大长度
    max_len = get_max_len(data_df['review'])

    data_df['review'] = data_df['review'].apply(lambda x: pad_proc(x, max_len, vocab))
    data_df['review'].to_csv(config.data_review_pad, index=False, header=False)

    # 5. 词向量再次训练
    print('start retrain w2v model')
    wv_model.build_vocab(LineSentence(config.data_review_pad), update=True)
    wv_model.train(LineSentence(config.data_review_pad),
                   epochs=config.wv_train_epochs,
                   total_examples=wv_model.corpus_count)

    # 6. 保存词向量模型
    wv_model.save(config.wv_model_path)
    print('finish retrain w2v model')
    print('final w2v_model has vocabulary of ', len(wv_model.wv.vocab))

    # 7. 更新vocab
    vocab = {word: index for index, word in enumerate(wv_model.wv.index2word)}
    reverse_vocab = {index: word for index, word in enumerate(wv_model.wv.index2word)}
    torch.save(vocab, config.vocab)
    torch.save(reverse_vocab,config.reverse_vocab)

    # 8. 训练集、验证集划分
    train_df, val_df = train_test_split(data_df,
                                        test_size=0.2,
                                        shuffle=True,
                                        stratify=data_df['label'],
                                        random_state=33)
    print("train data size {}, validation data size {}".format(len(train_df), len(val_df)))


    # 9. 将词转换成索引
    train_X = train_df['review'].apply(lambda x: transform_data(x, vocab))
    train_Y = train_df['label']
    val_X = val_df['review'].apply(lambda x: transform_data(x, vocab))
    val_Y = val_df['label']

    train_X = np.array(train_X.tolist())
    val_X = np.array(val_X.tolist())
    train_Y = np.array(train_Y.tolist())
    val_Y = np.array(val_Y.tolist())

    np.savetxt(config.train_X, train_X, fmt='%0.8f')
    np.savetxt(config.train_Y, train_Y, fmt='%0.8f')
    np.savetxt(config.val_X, val_X, fmt='%0.8f')
    np.savetxt(config.val_Y, val_Y, fmt='%0.8f')




if __name__ == '__main__':

    process_data()
    wv_model = Word2Vec.load(config.wv_model_path)
    embedding_matrix = wv_model.wv.vectors
    print(embedding_matrix.shape)
    print('---------------')