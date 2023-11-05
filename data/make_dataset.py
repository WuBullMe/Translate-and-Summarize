import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np

SOS_token = 0
EOS_token = 1
PAD_token = 2

class Vocabulary:
    """
        Vocabulary of words:
            * Initially filled by 3 tokens
                * <sos> -> start of sequence
                * <eos> -> end of sequence
                * <pad> -> fill to MAX_length
    """
    def __init__(self, name):
        self.name = name
        self.word2index = {"<sos>": 0, "<eos>": 1, "<pad>": 2}
        self.word2count = {}
        self.index2word = {0: "<sos>", 1: "<eos>", 2 : "<pad>"}
        self.n_words = 3

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


# Split sentence
def getList(sentence):
    return sentence.split(' ')


# Filter pair(toxic text, or translated text) by number of words < MAX_LENGTH
def filterPair(p, MAX_LENGTH):
    return len(getList(p[0])) < MAX_LENGTH - 2 and \
        len(getList(p[1])) < MAX_LENGTH - 2 # EOS


# Filter every pair(toxic text, or translated text) by number of words < MAX_LENGTH
def filter(norm_ref, norm_trs, MAX_LENGTH):
    filter_ref = []
    filter_trs = []
    for pair in zip(norm_ref, norm_trs):
        if filterPair(pair, MAX_LENGTH):
            filter_ref.append(pair[0])
            filter_trs.append(pair[1])
    return filter_ref, filter_trs


# Create vocabulary for toxic text and translated, also pair of them
def prepareData(data, MAX_LENGTH):
    # Normalize every data and filter
    norm_ref = [row for row in data['en']]
    norm_trs = [row for row in data['ru']]
    
    norm_ref, norm_trs = filter(norm_ref, norm_trs, MAX_LENGTH)
    # Make Vocabulary instances
    vocab_en = Vocabulary('en-vocab')
    vocab_ru = Vocabulary('ru-vocab')
    pairs = []
    for row in zip(norm_ref, norm_trs):
        pairs.append(row)

    for row in norm_ref:
        vocab_en.addSentence(row)

    for row in norm_trs:
        vocab_ru.addSentence(row)

    print("Counted words:")
    print(vocab_en.name, vocab_en.n_words)
    print(vocab_ru.name, vocab_ru.n_words)

    return vocab_en, vocab_ru, pairs

# Convert every word in sentence to indexes of vocabulary
def indexesFromSentence(vocab, sentence):
    return [vocab.word2index[word] for word in getList(sentence)]

# Convert every word in sentence to indexes of vocabulary in tensor format
def tensorFromSentence(vocab, sentence, device="cpu"):
    indexes = [SOS_token] + indexesFromSentence(vocab, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(1, -1)

# Convert every word in pair sentences to indexes of vocabulary in tensor format
def tensorsFromPair(pair, device="cpu"):
    input_tensor = tensorFromSentence(vocab_tox, pair[0], device)
    target_tensor = tensorFromSentence(vocab_detox, pair[1], device)
    return (input_tensor, target_tensor)


def get_dataloader(batch_size, vocab_en, vocab_ru, pairs, MAX_LENGTH, device="cpu", train_size=0.9):
    """
        Return dataloaders of data pairs by given parameters:
            :param batch_size: dataloader of batch_size
            :param vocab_tox: vocabulary for toxic text
            :param vocab_detox: vocabulary for translated text
            :param pairs: data to create dataloader
            :param train_size: proportion for train part
    """
    n = len(pairs)
    input_ids = np.zeros((n, MAX_LENGTH), dtype=np.int32)
    target_ids = np.zeros((n, MAX_LENGTH), dtype=np.int32)

    for idx, (inp, tgt) in enumerate(pairs):
        inp_ids = [SOS_token] + indexesFromSentence(vocab_en, inp)
        tgt_ids = [EOS_token] + indexesFromSentence(vocab_ru, tgt)
        inp_ids.append(EOS_token)
        tgt_ids.append(EOS_token)
        while len(inp_ids) < MAX_LENGTH:
            inp_ids.append(PAD_token)
        
        while len(tgt_ids) < MAX_LENGTH:
            tgt_ids.append(PAD_token)
        
        input_ids[idx] = inp_ids
        target_ids[idx] = tgt_ids

    idx = [i for i in range(n)]
    train_idx, val_idx = train_test_split(idx, train_size=train_size, random_state=420)
    train_data = TensorDataset(torch.LongTensor(input_ids[train_idx]).to(device),
                               torch.LongTensor(target_ids[train_idx]).to(device))
    val_data = TensorDataset(torch.LongTensor(input_ids[val_idx]).to(device),
                               torch.LongTensor(target_ids[val_idx]).to(device))


    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    return train_dataloader, val_dataloader
