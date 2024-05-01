import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random
from api import Linear,BatchNorm1d,Tanh,Embedding,FlattenC,Sequential

#declaring variables
block_size=8
n_emb=24
n_hidden=128
max_steps=200000
batch_size=32
n=batch_size
evaluation_iters=10000
g=torch.Generator().manual_seed(2147483647)

# reading data
words=open('names.txt','r').read().splitlines()
char=sorted(set(''.join(words))) 

# encoding and decoding data
stoi={s:(i+1) for i,s in enumerate(char)}
stoi['.']=0
itos={i:s for s,i in stoi.items()}
vocab_size=len(itos)


#building dataset
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

#splitting in train,test,val
random.seed(42)
random.shuffle(words)
n1=int(0.8*len(words))
n2=int(0.9*len(words))

x_train,y_train=build_dataset(words[:n1])
x_dev,y_dev=build_dataset(words[n1:n2])
x_test,y_test=build_dataset(words[n2:])
#-------------------------------------------------------------------------------------------------------------

def get_batch(split):
    data=x_train if split == 'train' else x_dev
    ix=torch.randint(0,data.shape[0],(batch_size,))
    xs,ys=x_train[ix],y_train[ix]
    return xs,ys



class wavenet_model():
    def __init__(self):
        self.model=Sequential([
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
            self.model.layers[-1].weight*=0.1
        self.parameters=self.model.parameters()
        for p in self.parameters:
            p.requires_grad=True
        

    def estimate_loss(self):
        out={}
        self.model.eval()
        for split in ['train','dev']:
            lossi=torch.zeros(10000)
            for i in range(10000):
                xb,yb=get_batch(split)
                logits=self.model(xb)
                loss=F.cross_entropy(logits,yb)
                lossi[i]=loss
            out[split]=lossi.mean()
        self.model.train()
        return out
        
    def forward(self,xb,yb,lr):
        logits=self.model(xb)
        loss=F.cross_entropy(logits,yb)
            
        for p in self.parameters:
            p.grad=None
        loss.backward()

        for p in self.parameters:
            p.data+=-lr*p.grad
    
    @torch.no_grad() # this decorator disables gradient tracking inside pytorch
    def evaluate(self,split):
        xb,yb = {
            'train': (x_train, y_train),
            'dev': (x_dev, y_dev),
            'test': (x_test, y_test),
        }[split]
        self.model.eval()
        logits=self.model(xb)
        loss = F.cross_entropy(logits, yb)
        print(split, loss.item())

    def generate(self):
        for i in range(10):
            context=[0]*block_size
            out=[]
            while True:
                logits=self.model(torch.tensor([context]))
                prob=F.softmax(logits,dim=1)
                ix=torch.multinomial(prob,num_samples=1,).item()
                out.append(itos[ix])
                context=context[1:]+[ix]
                if ix==0:
                    break
            print(''.join(out))



wavenet=wavenet_model()
for i  in range(max_steps):
    if i % evaluation_iters==0:
        losses=wavenet.estimate_loss()
        print(f"steps{i}\t train_loss{losses['train']}\t val_loss{losses['dev']}")
    xb,yb=get_batch('train')
    lr=0.1 if i<100000 else 0.01
    wavenet.forward(xb,yb,lr)

for split in ['train','dev','test']:
    wavenet.evaluate(split)

wavenet.generate()



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


