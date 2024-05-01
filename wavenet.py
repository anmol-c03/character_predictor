import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random


#declaring variables
block_size=8
n_emb=24
n_hidden=128
max_steps=200000
batch_size=32
n=batch_size
evaluation_iters=10000
lossi,lri=[],[]
g=torch.Generator().manual_seed(2147483647)


words=open('names.txt','r').read().splitlines()
char=sorted(set(''.join(words))) 
stoi={s:(i+1) for i,s in enumerate(char)}
stoi['.']=0
itos={i:s for s,i in stoi.items()}
vocab_size=len(itos)



def build_dataset(words):
    x,y=[],[]
    for w in words:
        context=[0]*block_size
        for ch in w+'.':
            ix=stoi[ch]
            x.append(context)
            y.append(ix)
            context=context[1:]+[ix]
    xs=torch.tensor(x)    
    ys=torch.tensor(y)
    return xs,ys

random.seed(42)
random.shuffle(words)
n1=int(0.8*len(words))
n2=int(0.9*len(words))

x_train,y_train=build_dataset(words[:n1])
x_dev,y_dev=build_dataset(words[n1:n2])
x_test,y_test=build_dataset(words[n2:])
#-------------------------------------------------------------------------------------------------------------
class Linear:
    def __init__(self,fan_in,fan_out,bias=True):
        self.weight=torch.randn((fan_in,fan_out),generator=g)/fan_in**0.5
        self.bias=torch.zeros(fan_out) if bias else None
        
    def __call__(self,x):
        self.out=x @ self.weight
        if self.bias is not None:
            self.out+=self.bias
        return self.out
        
    def parameters(self):
        return [self.weight]+([] if self.bias is None else [self.bias])
#--------------------------------------------------------------------------------------------------------------------------------------------
class BatchNorm1d:
  
  def __init__(self, dim, eps=1e-5, momentum=0.1):
    self.eps = eps
    self.momentum = momentum
    self.training = True
      
    self.gamma = torch.ones(dim)
    self.beta = torch.zeros(dim)

    self.running_mean = torch.zeros(dim)
    self.running_var = torch.ones(dim)
  
  def __call__(self, x):

    if self.training:
      if x.ndim==2:
          dim=0
      elif x.ndim==3:
          dim=(0,1)
      xmean = x.mean(dim, keepdim=True) 
      xvar = x.var(dim, keepdim=True) 
    else:
      xmean = self.running_mean
      xvar = self.running_var
    
    xhat = (x - xmean) / torch.sqrt(xvar + self.eps) 
      
    self.out = self.gamma * xhat + self.beta

    if self.training:
      with torch.no_grad():
        self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * xmean
        self.running_var = (1 - self.momentum) * self.running_var + self.momentum * xvar
    return self.out
  
  def parameters(self):
    return [self.gamma, self.beta]

#--------------------------------------------------------------------------------------------------------------------------------------------
class Tanh:
  def __call__(self, x):
    self.out = torch.tanh(x)
    return self.out
      
  def parameters(self):
    return []
#--------------------------------------------------------------------------------------------------------------------------------------------      
class Embedding:
    def __init__(self,num_emb,emb_dim):
        self.weight=torch.randn(num_emb,emb_dim,generator=g)

    def __call__(self,index):
        self.out=self.weight[index]
        return self.out

    def parameters(self):
        return [self.weight]

#--------------------------------------------------------------------------------------------------------------------------------------------
class FlattenC:
    def __init__(self,n):
        self.n=n
        
    def __call__(self,x):
        B,T,C=x.shape
        x=x.view(B,T//self.n,C*self.n)
        if x.shape[1]==1:
            x=x.squeeze(1)
        self.out=x
        return self.out

    def parameters(self):
        return []
#----------------------------------------------------------------------------------------------------------------------------------------------
class Sequential:
    def __init__(self,layers):
        self.layers=layers

    def __call__(self,x):
        for layer in self.layers:
            x=layer(x)
        self.out=x
        return self.out

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
#----------------------------------------------------------------------------------------------------------------------------------------------      



model=Sequential([
    Embedding(vocab_size,n_emb),
    FlattenC(2),
    Linear(2*n_emb,n_hidden,bias=False),BatchNorm1d(n_hidden),Tanh(),
    FlattenC(2),    
    Linear(2*n_hidden,n_hidden,bias=False),BatchNorm1d(n_hidden),Tanh(),
    FlattenC(2),    
    Linear(2*n_hidden,n_hidden,bias=False),BatchNorm1d(n_hidden),Tanh(),
    Linear(           n_hidden,vocab_size)
])

with torch.no_grad():
    model.layers[-1].weight*=0.1
    '''
    this commented out layer below is used to properly initialize the all the weights such that after 
    matrix operations, the indermediate logits will have gussian distribution and that distrb wii be 
    standard(it is ensured by sqrt(fan_in) during weights initialization in Linear class) 
    if not initailized weights by scaling with gain of tanh then the squashing results those 
    logits to be more concentrated toward zero
    
    simple example is ReLU activation it just neglects all the negative intermediate logits so
    to compensate that its gain is 2
    if applied batch norm , all dataset are normalized hence gain can be considered as 1 

    for layer in layers[:-1]:
        if isinstance(layer,Linear):
            layer.weight*=1 #5/3
'''

parameters=model.parameters()
print(sum(p.nelement() for p in parameters))
for p in parameters:
    p.requires_grad=True

#defining batches
def get_batch(self,split):
    data=x_train if split is 'train' else x_dev
    ix=torch.randint(0,data.shape[0],(batch_size,))
    xs,ys=x_train[ix],y_train[ix]
    return xs,ys

@torch.no_grad()
def estimate_loss():
    out={}
    model.eval()
    for split in ['train','dev']:
        lossi=torch.zeros(evaluation_iters)
        for i in range(evaluation_iters):
            xb,yb=get_batch(split)
            logits=model(xb,yb)
            loss=F.cross_entropy(logits,yb)
            lossi[i]=loss
        out[split]=lossi.mean()
    model.train()
    return out


def training(lr):
    xb,yb=get_batch('train')
            
    logits=model(xb)
    loss=F.cross_entropy(logits,yb)
            
    for p in parameters:
        p.grad=None
    loss.backward()

    for p in parameters:
        p.data+=-lr*p.grad

for i in range(max_steps):
    lr=0.1 if i<100000 else 0.01
    training(lr)
    if i % evaluation_iters ==0:
        losses=estimate_loss()
        print(f'iters{i}\t train_loss{losses['train']}\t val_loss{losses['dev']}')
    



for layer in model.layers:
    layer.training=False

# evaluation phase 
@torch.no_grad() # this decorator disables gradient tracking inside pytorch
def split_loss(split):
    xb,yb = {
        'train': (x_train, y_train),
        'val': (x_dev, y_dev),
        'test': (x_test, y_test),
      }[split]
    logits=model(xb)
    loss = F.cross_entropy(logits, yb)
    print(split, loss.item())

split_loss('train')
split_loss('val')

# prediction phase
block_size=8
for i in range(10):
    context=[0]*block_size
    out=[]
    while True:
        logits=model(torch.tensor([context]))
        prob=F.softmax(logits,dim=1)
        ix=torch.multinomial(prob,num_samples=1,).item()
        out.append(itos[ix])
        context=context[1:]+[ix]
        if ix==0:
            break
    print(''.join(out))
    