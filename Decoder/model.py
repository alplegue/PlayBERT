import math
import torch


class Head(torch.nn.Module):

    def __init__(self, block_size, d_model, d_v, d_k):
        super().__init__()
        self.d_k = d_k
        self.W_q = torch.nn.Linear(d_model, d_k)
        self.W_k = torch.nn.Linear(d_model, d_k)
        self.W_v = torch.nn.Linear(d_model, d_v)
        self.register_buffer("causal_mask", self.ins_causal_mask(block_size))


    def forward(self, X_e):
        B, T, d_model = X_e.shape
        Q = self.W_q(X_e) #(B,T,d_k)
        K = self.W_k(X_e) #(B,T,d_k)
        V = self.W_v(X_e) #(B,T,d_v)
        QK = Q @ K.transpose(1,2)
        QK_scale = QK/math.sqrt(self.d_k) #(B,T,T)
        QK_scale += self.causal_mask[:T, :T]
        QV_softmax = torch.nn.functional.softmax(QK_scale, dim = 2) #(B,T,T)
        added_weights = QV_softmax @ V #(B, T, d_v)
        return added_weights


    @staticmethod
    def ins_causal_mask(block_size):
        mask = torch.triu(torch.ones(block_size, block_size), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask


class MLP(torch.nn.Module):

    def __init__(self, d_model):
            
        super().__init__()
        self.W_ff1 = torch.nn.Linear(d_model, 4*d_model)
        self.W_ff2 = torch.nn.Linear(4*d_model, d_model)

    def forward(self, X):

        linear_1 = self.W_ff1(X)
        non_linearity = torch.nn.functional.gelu(linear_1)
        linear_2 = self.W_ff2(non_linearity)
        return linear_2

        

class MultiHead(torch.nn.Module):
    
    def __init__(self, block_size, d_model, n_heads, d_k):
        super().__init__()
        assert d_model % n_heads == 0
        d_v = d_model // n_heads
        d_k = d_k if d_k is not None else d_v
        self.heads = torch.nn.ModuleList([Head(block_size, d_model, d_v, d_k) for _ in range(n_heads)])
        self.W_o = torch.nn.Linear(d_model, d_model)

    def forward(self, X_e):

        added_weights = [head(X_e) for head in self.heads]
        concat = torch.cat(added_weights, dim = -1)
        return self.W_o(concat)

class Block(torch.nn.Module):
     
    def __init__(self, block_size, d_model, n_heads, d_k):
        super().__init__()
        self.att = MultiHead(block_size, d_model, n_heads, d_k)
        self.ln_1 = torch.nn.LayerNorm(d_model)
        self.mlp = MLP(d_model)
        self.ln_2 = torch.nn.LayerNorm(d_model)

    def forward(self, X):

        X = X + self.att(self.ln_1(X))
        X = X + self.mlp(self.ln_2(X))
        return X


class Decoder_Only_Transformer(torch.nn.Module):

    def __init__(self, vocab_size, block_size, d_model, n_heads = 1, d_k = None, n_layers = 1):

        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, d_model) 
        self.blocks = torch.nn.ModuleList([Block(block_size, d_model, n_heads, d_k) for _ in range(n_layers)])
        self.ln_f = torch.nn.LayerNorm(d_model)
        self.linear = torch.nn.Linear(d_model, vocab_size)
        self.register_buffer("pos_encoding", self.sinusoidal_pe(block_size, d_model))
        self.loss = torch.nn.CrossEntropyLoss()

    
    @staticmethod
    def sinusoidal_pe(block_size, d_model):
        pe = torch.zeros(block_size, d_model)
        for pos in range(block_size):
            for i in range(0, d_model, 2):
                denom = 10000 ** (i / d_model)
                pe[pos, i]     = math.sin(pos / denom)
                pe[pos, i + 1] = math.cos(pos / denom)
        return pe

    
    
    def forward(self, X, Y = None):

        B, T = X.shape
        X_e = self.embedding(X) + self.pos_encoding[:T] #(B,T,d_model)
        for block in self.blocks:
            X_e = block(X_e)
        X_e = self.ln_f(X_e) #(B,T,d_model)
        logits = self.linear(X_e) #(B,T,vocab_size)
        loss = None
        if Y is not None:
            loss = self.loss(logits.reshape(B*T,-1), Y.reshape(B*T,))
        
        return logits, loss



@torch.no_grad()
def generate_stream(modelo, idx, max_new_tokens, temperature=1.0, top_k=None):
    modelo.eval()
    block_size = modelo.pos_encoding.shape[0]
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:]
        logits, _ = modelo(idx_cond)
        logits = logits[:, -1, :] / temperature
        if top_k is not None:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = float('-inf')
        probs = torch.softmax(logits, dim=-1)
        next_tok = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_tok], dim=1)
        yield next_tok.item()                 
