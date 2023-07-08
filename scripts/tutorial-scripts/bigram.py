import torch
import torch.nn as nn
from torch.nn import functional as F

batch_size = 32 # number of sequences per batch
block_size = 8 # number of tokens/characters per sequence
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200

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

class BigramLanguageModel(nn.Module):
  def __init__(self, vocab_size):
    super().__init__()  # calls the constructor of the parent class (nn.Module)
    self.token_embedding_table = nn.Embedding(vocab_size, vocab_size) # paramters are num_embeddings (size of dictionary), embedding_dim (dim of embeddign vec)

  def forward(self, idx, targets=None):

    # idx (inputs) and targets are both (B, T) tensors
    # based on each input int a row from the embedding corresponding with the index (input) that will be an embedding vector
    logits = self.token_embedding_table(idx)  # (B, T, C) (Batch, Time, Channels) (4, 8, vocab_size)

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
      # gets the predictions
      logits, loss, = self(idx) # call forward
      # focus only on the last time step
      logits = logits[: , -1, :]  # becomes (B, C)
      # apply softmax to normalize and get probabilities
      probs = F.softmax(logits, dim=-1) # dim are (B, C)
      # sample from distribution to get a single prediction for what char comes next
      idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
      # append sampled index to the running sequence
      idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
    return idx
  
model = BigramLanguageModel(vocab_size)
m = model.to(device)  # moves all the calcualtions on the GPU if available

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

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
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))   # we will get random 100 results at first since its not trained yet