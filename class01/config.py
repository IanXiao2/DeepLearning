
import os
import pathlib

class Config:

    root = pathlib.Path(os.path.abspath(__file__)).parent

    data_path = os.path.join(root, 'data', 'weibo_senti_100k.csv')
    data_review = os.path.join(root, 'data', 'data_review.csv')
    data_review_pad = os.path.join(root, 'data', 'data_review_pad.csv')

    train_X = os.path.join(root, 'data', 'train_X.txt')
    train_Y = os.path.join(root, 'data', 'train_Y.txt')
    val_X = os.path.join(root, 'data', 'val_X.txt')
    val_Y = os.path.join(root, 'data', 'val_Y.txt')

    stop_words_path = os.path.join(root, 'data', 'stopwords/哈工大停用词表.txt')

    wv_model_path = os.path.join(root, 'data','wv', 'word2vec.model')
    vocab = os.path.join(root, 'data', 'wv', 'vocab.pth')
    reverse_vocab = os.path.join(root, 'data', 'wv', 'reverse_vocab.pth')
    embedding_dim = 300

    ckpt = 'ckpt'

    # 词向量训练轮数
    wv_train_epochs = 5

    model_name = 'TextCNN'

    batch_size = 64
    max_epoch = 256

    lr_decay = 0.1
    lr = 1e-3
    #在第几个epoch进行到下一个stage，调整lr
    stage_epoch = [32, 64, 128]

config = Config()