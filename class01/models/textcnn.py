
import torch.nn as nn
import torch
from gensim.models.word2vec import Word2Vec
from config import config

import torch.nn.functional as F
from dataset import TextDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TextCNN(nn.Module):
    def __init__(self):
        super(TextCNN, self).__init__()
        wv_model = Word2Vec.load(config.wv_model_path)
        embedding_matrix = wv_model.wv.vectors
        embedding_matrix = torch.tensor(embedding_matrix, dtype=torch.float32).to(device)
        self.vocab_size = embedding_matrix.shape[0]
        self.embedding_dim = embedding_matrix.shape[1]

        self.filter_size = (2, 3, 4)
        self.num_filters = 256
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim).from_pretrained(embedding_matrix)
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, self.num_filters, (k, self.embedding_dim)) for k in self.filter_size])
        self.dropout = nn.Dropout(.5)

        self.fc = nn.Linear(self.num_filters * len(self.filter_size), 1)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        out = self.embedding(x)
        out = out.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.dropout(out)
        out = self.fc(out)
        return out

if __name__ == "__main__":

    d = TextDataset(train=True)
    model = TextCNN()
    sample = d[0][0].unsqueeze(0)
    out = model(sample)

    print('------------')