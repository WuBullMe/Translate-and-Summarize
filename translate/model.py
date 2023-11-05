import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForward(nn.Module):
    """
        Linear layer after non-linearity
    """
    def __init__(self, n_embd):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(n_embd, n_embd * 4),
            nn.ReLU(),
            nn.Linear(n_embd * 4, n_embd),
            nn.Dropout(0.2)
        )
    
    def forward(self, data):
        return self.model(data)


class AttentionHead(nn.Module):
    """
        Attention head of heads
    """
    def __init__(self, embed_size, head_size, batch_size, max_length, mask=False, dropout=0.2):
        super().__init__()
        self.mask = mask
        
        self.query = nn.Linear(embed_size, head_size, bias=False)
        self.key = nn.Linear(embed_size, head_size, bias=False)
        self.value = nn.Linear(embed_size, head_size, bias=False)
        if mask:
            self.register_buffer('tril', torch.tril(torch.ones(max_length, max_length)))
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, data, crossData=None):
        """
            :param crossData: 
                * crossData = None -> It's Self Attention
                * else             -> It's Cross Attention
                
                
            B -> batch size
            T -> series length
            C -> size of embedding
        """
        B,T,C = data.shape
        query = self.query(data)
        if crossData is not None:
            key = self.key(crossData)
            v = self.value(crossData)
        else:
            key = self.key(data)
            v = self.value(data)
        
        wei = query @ key.transpose(1, 2) * (C**-0.5) # # (B, 1, C) @ (B, C, T) -> (B, 1, T)
        if self.mask:
            wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        
        out = wei @ v # (B, 1, T) @ (B, T, C) -> (B, 1, C)
        return out
        
        
class MultipleAttention(nn.Module):
    """
        Multiple head for Attention
    """
    def __init__(self, embed_size, n_head, head_size, batch_size, max_length, mask=False, dropout=0.2):
        super().__init__()
        self.blockHead = nn.ModuleList([AttentionHead(embed_size, head_size, batch_size, max_length, mask, dropout) for _ in range(n_head)])
        self.fc = nn.Linear(head_size * n_head, embed_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, data, crossData=None):
        heads = torch.cat([layer(data, crossData=crossData) for layer in self.blockHead], dim=-1)
        
        out = self.dropout(self.fc(heads))
        return out


class Attention(nn.Module):
    """
        Attention Block
    """
    def __init__(self, embed_size, n_head, batch_size, max_length, mask=False, feedforward=False, dropout=0.2):
        super().__init__()
        head_size = embed_size // n_head
        self.feedforward = feedforward
        self.sa = MultipleAttention(embed_size, n_head, head_size, batch_size, max_length, mask, dropout)
        self.ffwd = FeedForward(embed_size)
        self.ln1 = nn.LayerNorm(embed_size)
        self.ln2 = nn.LayerNorm(embed_size)
        
    def forward(self, data, crossData=None):
        data = data + self.sa(self.ln1(data), crossData=crossData)
        if self.feedforward:
            data = data + self.ffwd(self.ln2(data))
        return data


class Transformer(nn.Module):
    """
        Transformer model
    """
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        
    def forward(self, data, target=None, t_f=None): # t_f is only because of train code
        B, T = data.shape # batch, time length
        encoder_outputs = self.encoder(data)
        out = self.decoder(encoder_outputs, target)
        
        return out

    def generate_max(self, query, max_length=10):
        """
            Predict sentence with geedy search (In each time take with maximum probability)
        """
        with torch.no_grad():
            encoder_ouputs = self.encoder(query)
            targets = torch.empty(1, 1, dtype=torch.long, device=self.encoder.device).fill_(self.decoder.vocab.word2index['<sos>'])
            targets = [self.decoder.vocab.word2index['<sos>']]
            for i in range(1, max_length):
                out = self.decoder(encoder_ouputs, torch.tensor(targets, device=self.encoder.device).unsqueeze(0)) # (1, i, C)
                targets.append(out.argmax(-1)[0][-1].detach().cpu().item())
                if targets[-1] == self.decoder.vocab.word2index['<eos>']:
                    break
        
            return targets
    
    def generate_beam(self, query, max_length=10, beam_width=4):
        """
            Implementation of Beam search algorithm
            (for more information check: https://www.width.ai/post/what-is-beam-search)
            
                :param query: the sentence to be detoxified
                :param max_length: the upper limit length of predicted sentence
                :param beam_width: hyperparameter of beam_search algorithm
        """
        with torch.no_grad():
            encoder_ouputs = self.encoder(query)
            targets = torch.empty(1, 1, dtype=torch.long, device=self.encoder.device).fill_(self.decoder.vocab.word2index['<sos>'])
            
            values = [torch.log(torch.tensor(1.0))]
            for i in range(1, max_length):
                out = self.decoder(encoder_ouputs, targets) # (B, C, T)
                out = out[:, -1, :] # (B, T)
                out = F.softmax(out, dim=-1)
                val, idx = out.topk(beam_width) # (B, beam_width)
                B, T = idx.shape
                assert T == beam_width
                cand = []
                for i1 in range(B):
                    for i2 in range(T):
                        cand.append((i1, i2, torch.log(val[i1][i2])))
                cand.sort(key=lambda tup: tup[2] + values[tup[0]], reverse=True)
                i1 = cand[0][0]
                i2 = cand[0][1]
                n_idx = torch.empty(1, 1, dtype=torch.long, device=self.encoder.device).fill_(idx[i1][i2])
                n_targets = torch.cat((targets[i1:i1+1, :], n_idx), dim=1)
                n_values = [values[i1] + cand[0][2]]
                for j1 in range(T - 1):
                    i1 = cand[j1][0]
                    i2 = cand[j1][1]
                    j_idx = torch.empty(1, 1, dtype=torch.long, device=self.encoder.device).fill_(idx[i1][i2])
                    j_targets = torch.cat((targets[i1:i1+1, :], j_idx), dim=1)
                    j_value = values[i1] + cand[j1][2]
                    n_targets = torch.cat((n_targets, j_targets), dim=0)
                    n_values.append(j_value)
                
                targets = n_targets
                values = n_values
                if targets[0][-1].item() == self.decoder.vocab.word2index['<eos>']:
                    break
            
            return targets[0:1, :]


class Encoder(nn.Module):
    """
        Encoder part
    """
    def __init__(self, input_size, n_layer, n_head, batch_size, embed_size, hidden_size, vocab, device=torch.device("cpu"), max_length=0, dropout=0.2):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.vocab = vocab
        self.device = device
        self.max_length = max_length
        
        self.word_embedding = nn.Embedding(input_size, embed_size, padding_idx=self.vocab.word2index['<pad>'])
        self.pos_embedding = nn.Embedding(max_length, embed_size, padding_idx=self.vocab.word2index['<pad>'])
        self.self_attn = nn.Sequential(*[Attention(embed_size, n_head, batch_size, max_length, dropout=dropout, feedforward=True) for _ in range(n_layer)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        B, T = data.shape # Batch size, Length of sequence
        embed = self.word_embedding(data) # (B, T, C)
        pos = self.pos_embedding(torch.arange(0, T).unsqueeze(0).repeat(B, 1).to(self.device))
        
        out = embed + pos # (B, T, C)
        out = self.self_attn(out) # (B, T, C)
        return out


class Decoder(nn.Module):
    """
        Decoder part
    """
    def __init__(self, n_layer, n_head, batch_size, embed_size, hidden_size, output_size, vocab, device=torch.device("cpu"), max_length = 0, dropout=0.2):
        super(Decoder, self).__init__()
        self.vocab = vocab
        self.device = device
        self.max_length = max_length
        
        self.word_embedding = nn.Embedding(output_size, embed_size, padding_idx=self.vocab.word2index['<pad>'])
        self.pos_embedding = nn.Embedding(max_length, embed_size, padding_idx=self.vocab.word2index['<pad>'])
        self.self_attn = nn.Sequential(*[Attention(embed_size, n_head, batch_size, max_length, mask=True, dropout=dropout) for _ in range(n_layer)])
        self.cross_attn = nn.ModuleList([Attention(embed_size, n_head, batch_size, max_length, dropout=dropout, feedforward=True) for _ in range(n_layer)])
        self.out = nn.Linear(hidden_size, output_size)

    
    def forward(self, encoder_outputs, target_tensor):
        B, T = target_tensor.shape # Batch size, Length of sequence
        embed = self.word_embedding(target_tensor)
        pos = self.pos_embedding(torch.arange(0, T).unsqueeze(0).repeat(B, 1).to(self.device))
        
        out = embed + pos
        out = self.self_attn(out)
        
        for layer in self.cross_attn:
            out = layer(out, encoder_outputs)
        
        output = self.out(out)
        return output
