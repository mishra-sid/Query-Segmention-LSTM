import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

class LSTMSegmenter(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, target_size):
        super(LSTMSegmenter, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        
        self.hidden2boundaries = nn.Linear(hidden_dim, target_size)

        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.zeros(1, 1, self.hidden_dim),
                torch.zeros(1, 1, self.hidden_dim))

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, self.hidden = self.lstm(
                embeds.view(len(sentence), 1, -1), self.hidden)
        boundaries = self.hidden2boundaries(lstm_out.view(len(sentence), -1))
        boundary_labels = F.log_softmax(boundaries, dim=1)
        return boundary_labels

class QuerySegmentation:
    def __init__(self, training_file, label_file, n_epochs=2000, embedding_dim=15, hidden_dim=5):
        self.training_X = []
        self.training_Y_all_labels = []
        self.training_Y = []
        self.word_to_ix = {}
        self.n_epochs = n_epochs
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim


        with open(training_file, 'r') as f:
            line = f.readline()
            sentence = line.rstrip().split("\t")
            while line:
                self.training_X.append(sentence[1])
                line = f.readline()
                sentence = line.rstrip().split("\t")

        with open(label_file, 'r') as f:
            line = f.readline()
            sentence = line.rstrip().split("\t")
            while line:
                labels = eval(sentence[1])
                self.training_Y_all_labels.append(labels)
                self.training_Y.append( list(filter(lambda y : y[0] == max(x[0] for x in labels), labels))[0])
                line = f.readline()
                sentence = line.rstrip().split("\t")

        print(self.training_X)
        print(self.training_Y_all_labels)
        print(self.training_Y)

        self.preprocess_vocab()
        self.model = LSTMSegmenter(self.embedding_dim, self.hidden_dim, len(self.word_to_ix), 2)
        self.loss_function = nn.NLLLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.05)

    def preprocess_vocab(self):
        for s in self.training_X:
            for w in s.split(" "):
                if w not in self.word_to_ix:
                    self.word_to_ix[w] = len(self.word_to_ix)

    def encode_sequence(self, sentence):
        idxs = [self.word_to_ix[w] for w in sentence.split(" ")]
        idxs += [0] * (self.embedding_dim - len(idxs))
        return torch.tensor(idxs, dtype=torch.long)

    def train(self):
        for epoch in range(self.n_epochs):
            average_loss = 0.0
            for idx, sentence in enumerate(self.training_X):
                self.model.zero_grad()
                self.model.hidden = self.model.init_hidden()

                sentence_in = self.encode_sequence(sentence)
                prob_vector = []
                target_labels = self.training_Y_all_labels[idx]
                bar_prob = {}
                id = 0
                for word in sentence.split(" "):
                    bar_prob[id] = 0.0
                    id += 1

                for label in target_labels:
                    id = 0
                    for word in label[1].split(" "):
                        lo_words = word.split("|")
                        for w in lo_words[ :-1]:
                            bar_prob[id] += label[0]
                            id += 1
                        id += 1

                for key in bar_prob:
                    bar_prob[key] /= 10.0

                for i in range(len(sentence.split(" "))):
                    prob_vector.append(int(bar_prob[i] > 0.5))

                prob_vector += [0] * (self.embedding_dim - len(prob_vector))

                label_vector = torch.tensor(prob_vector, dtype=torch.long)

                scores = self.model(sentence_in)
                loss = self.loss_function(scores, label_vector)
                average_loss += loss.item()
                print("Loss for epoch : " + str(epoch) + " sentence" + str(idx)  + "is : " + str(loss))
                loss.backward()
                self.optimizer.step()
            print("Average loss for epoch : " + str(epoch) + " is : " + str(average_loss/len(self.training_X)))


if __name__=="__main__":
    qs = QuerySegmentation('webis-qsec-10-training-set-queries.txt', 'webis-qsec-10-training-set-segmentations-crowdsourced.txt')
    qs.train()
