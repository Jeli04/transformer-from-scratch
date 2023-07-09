import torch
import torch.nn as nn
from torch.nn import functional as F

batch_size = 64 # number of sequences per batch
block_size = 256 # number of tokens/characters per sequence
max_iters = 6000
eval_interval = 500
learning_rate = 3e-4    # lower learning rate for bigger neural networks
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embed = 384
n_layer = 6
n_head = 6  # every head is 64 dim (384/6)
dropout = 0.2

print(f"Is CUDA supported by this system?{torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
 
# Storing ID of current CUDA device
cuda_id = torch.cuda.current_device()
print(f"ID of current CUDA device:{torch.cuda.current_device()}")
       
print(f"Name of current CUDA device:{torch.cuda.get_device_name(cuda_id)}")

torch.manual_seed(1337)

# open the text file as a string
# python -m wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding="utf-8") as f:
  text = f.read()

chars = sorted(list(set(text))) # sets gets one of each value
vocab_size = len(chars)
# create a simple mapping between text and int based on python convertion between int and char
stoi = {ch:i for i, ch in enumerate(chars)} # map for string to int
itos = {i:ch for i, ch in enumerate(chars)} # map for int to stirng
encode = lambda s : [stoi[c] for c in s]  # encodes the string into ints
decode = lambda l : ''.join(itos[i] for i in l)  # ''.join() concatanates a list into a string

data = torch.tensor(encode(text), dtype=torch.long)

# train and test splits
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
  # generate a small random batch of data x inputs and y targets
  data = train_data if split == "train" else val_data
  ix = torch.randint(len(data) - block_size, (batch_size,)) # generates n numbrs based on btach_size that represent an index from data, randomally choosing four places to parallel train
  # torch.stack stacks each 1D tensors of length 8 (block_size) 4 times (batch_size)
  x = torch.stack([data[i:i+block_size] for i in ix]) # inputs
  y = torch.stack([data[i+1:i+block_size+1] for i in ix]) # targets

  x, y = x.to(device), y.to(device) # moves to GPU if available

  return x, y

@torch.no_grad()  # tells pytorch that everything inside this function wont have back propogation
def estimate_loss():
  out = {}
  model.eval()
  for split in ["train", "val"]:
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
       X, Y = get_batch(split)
       logits, loss = model(X, Y)
       losses[k] = loss.item()
    out[split] = losses.mean()
  model.train()
  return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)

        self.key = nn.Linear(n_embed, n_embed, bias=False)  # 384, 384 changed from n_embed, head_size to n_embed, n_embed
        self.query = nn.Linear(n_embed, n_embed, bias=False)
        self.value = nn.Linear(n_embed, n_embed, bias=False)

        # the mask matrix
        self.register_buffer('tril', torch.tril(torch.ones((block_size, block_size)))) # a buffer is a tensor in a PyTorch module that isn't a model parameter but still used in the module

        self.num_heads = num_heads
        self.head_size = head_size  # head_size = n_embed // n_head

    def forward(self, x):
        B, T, C = x.shape   
  
        k,q,v = self.key(x), self.query(x), self.value(x)

        # have to creaste a 4D tensor with num_heads is the number of batches for parallel processing
        k = k.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)  # (B, num_heads, T, head_size)
        q = q.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)  # (B, num_heads, T, head_size)
        v = v.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)  # (B, num_heads, T, head_size)

        wei = q @ k.transpose(-2, -1) * C**-0.5  # C is head size 
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # mask out the upper triangle (B, T, T)
        wei = F.softmax(wei, dim=-1) # normalize

        out = wei @ v

        out = out.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side after spliting into batches (97-99)
        out = self.dropout(self.proj(out))  # projection layer back to the residual path
        return out


class FeedForward(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, n_embed * 4),    # * 4 based on the transformer paper
            nn.ReLU(),
            nn.Linear(n_embed * 4, n_embed),  # projection layer back to the residual path  
            nn.Dropout(dropout),
        )

    def forward(self, x):
       return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embed, n_head):
        super().__init__()
        head_size = n_embed // n_head   # 32 // 4 = 8
        self.sa_heads = MultiHeadAttention(n_head, head_size)  # self attention heads (4 heads each with a dim of 8)
        self.ffwd = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        # the x + is the residual connection/skip layers
        x = x + self.sa_heads(self.ln1(x))  # send to layer norm than the self attention heads
        x = x + self.ffwd(self.ln2(x))   # send to layer norm than the feed forward network
        return x
         

class BigramLanguageModel(nn.Module):
  def __init__(self):
    super().__init__()  # calls the constructor of the parent class (nn.Module)
    self.token_embedding_table = nn.Embedding(vocab_size, n_embed) # paramters are num_embeddings (size of dictionary), embedding_dim (dim of embeddign vec)
    self.position_embedding_table = nn.Embedding(block_size, n_embed) # embedding for token location
    self.blocks = nn.Sequential(*[Block(n_embed, n_head=n_head) for _ in range(n_layer)]) # shortened way for multiple blocks in a sequential model
    self.ln_f = nn.LayerNorm(n_embed)    # final layer norm

    # final linear layer that decodes the output of the transformer into the vocabulary
    self.lm_head = nn.Linear(n_embed, vocab_size) # paramters are in_features, out_features

  def forward(self, idx, targets=None):
    B, T = idx.shape

    # idx (inputs) and targets are bo th (B, T) tensors
    # based on each input int a row from the embedding corresponding with the index (input) that will be an embedding vector
    tok_emb = self.token_embedding_table(idx)  # (B, T, C) (Batch, Time, Channels) (4, 8, vocab_size)
    pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T, C) (8, vocab_size) gets the possible next 8 characters
    x = tok_emb + pos_emb  # (B,T,C)  x holds the token identities and their positions
    x = self.blocks(x)  # (B, T, C) 
    x = self.ln_f(x)  # (B, T, C)
    logits = self.lm_head(x)  # (B, T, vocab_size) (4, 8, vocab_size)

    if targets == None:
      loss = None
    else:
      # pytorch wants a (B, C, T) tensor for cross_entropy so we need some reshaping
      B, T, C = logits.shape
      logits = logits.view(B*T, C)
      targets = targets.view(B*T)
      loss = F.cross_entropy(logits, targets)

    return logits, loss

  # the job of generate is to extend idx to be B by T+1, B by T+2 ....
  def generate(self, idx, max_new_tokens):
    # idx is (B, T) array of indicies in the current context
    for _ in range(max_new_tokens):
      # crop the context to the last block_size tokens since the position embedding_table is only block_size
      idx_cond = idx[:, -block_size:]
      # gets the predictions
      logits, loss, = self(idx_cond) # call forward
      # focus only on the last time step
      logits = logits[: , -1, :]  # becomes (B, C)
      # apply softmax to normalize and get probabilities
      probs = F.softmax(logits, dim=-1) # dim are (B, C)
      # sample from distribution to get a single prediction for what char comes next
      idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
      # append sampled index to the running sequence
      idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
    return idx
  
  
def train_model(m):
  # print the number of parameters in the model
  print(sum(p.numel() for p in m.parameters())/1e6, "M paramters")

  # create a PyTorch optimizer
  optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

  for iter in range(max_iters):
    # once awhile evaluate the loss on train and val sets
    if iter % eval_interval == 0:
      losses = estimate_loss()  # estimate loss averages the losses of multiple batches 
      print(f"Step: {iter} | Train loss: {losses['train']:.4f} | Val loss: {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = m.forward(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

  # generation from the model 
  context = torch.zeros((1,1), dtype = torch.long, device = device)
  print(decode(m.generate(context, max_new_tokens=1000)[0].tolist()))   # we will get random 100 results at first since its not trained yet

  # save the model
  torch.save(m.state_dict(), "model2.pth")


# loading the model and genearting text
model = BigramLanguageModel()
m = model.to(device)  # moves all the calcualtions on the GPU if available
train_model(m)

#m.load_state_dict(torch.load("model2.pth"))
#print(sum(p.numel() for p in m.parameters())/1e6, "M paramters")

context = torch.zeros((1,1), dtype = torch.long, device = device)
print(decode(m.generate(context, max_new_tokens=1000)[0].tolist()))   # we will get random 100 results at first since its not trained yet
#open('output.txt', 'w').write(decode(m.generate(context, max_new_tokens=10000)[0].tolist()))