import numpy as np
import torch

from torch.utils.data import TensorDataset

               
def generate_recog_data(T=2000, d=50, R=1, P=0.5, interleave=True, multiRep=False, xDataVals='+-', softLabels=False):
    """Generates "image recognition dataset" sequence of (x,y) tuples. 
    x[t] is a d-dimensional random binary vector, 
    y[t] is 1 if x[t] has appeared in the sequence x[0] ... x[t-1], and 0 otherwise
    
    if interleave==False, (e.g. R=3) ab.ab.. is not allowed, must have a..ab..b.c..c (dots are new samples)
    if multiRep==False a pattern will only be (intentionally) repeated once in the trial
    
    T: length of trial
    d: length of x
    R: repeat interval
    P: probability of repeat
    """  
    if np.isscalar(R):
        Rlist = [R]
    else:
        Rlist = R
    
    data = []
    repeatFlag = False
    r=0 #countdown to repeat
    for t in range(T): 
        #decide if repeating
        R = Rlist[np.random.randint(0, len(Rlist))]
        if interleave:
            repeatFlag = np.random.rand()<P
        else:
            if r>0:
                repeatFlag = False
                r-=1
            else:
                repeatFlag = np.random.rand()<P 
                if repeatFlag:
                    r = R
                
        #generate datapoint
        if t>=R and repeatFlag and (multiRep or data[t-R][1].round()==0):
            x = data[t-R][0]
            y = 1
        else:
            if xDataVals == '+-': #TODO should really do this outside the loop...
                x = 2*np.round(np.random.rand(d))-1
            elif xDataVals.lower() == 'normal':
                x = np.sqrt(d)*np.random.randn(d)    
            elif xDataVals.lower().startswith('uniform'):
                upper, lower = parse_xDataVals_string(xDataVals)
                x = np.random.rand(d)*(upper-lower)+lower
            elif xDataVals == '01':
                x = np.round(np.random.rand(d))
            else:
                raise ValueError('Invalid value for "xDataVals" arg')           
            y = 0
            
        if softLabels:
            y*=(1-2*softLabels); y+=softLabels               
        data.append((x,np.array([y]))) 
        
    return data_to_tensor(data)

 
def generate_recog_data_batch(T=2000, batchSize=1, d=25, R=1, P=0.5, interleave=True, multiRep=False, softLabels=False, xDataVals='+-', device='cpu'):
    """Faster version of recognition data generation. Generates in batches and uses torch directly    
    Note: this is only faster when approx batchSize>4
    """  
    if np.isscalar(R):
        Rlist = [R]
    else:
        Rlist = R
    
    if xDataVals == '+-':
        x = 2*torch.rand(T,batchSize,d, device=device).round()-1 #faster than (torch.rand(T,B,d)-0.5).sign()
    elif xDataVals.lower() == 'normal':
        x = torch.randn(T,batchSize,d, device=device)    
    elif xDataVals.lower().startswith('uniform'):
        upper, lower = parse_xDataVals_string(xDataVals)
        x = torch.rand(T,batchSize,d, device=device)*(upper-lower)+lower
    elif xDataVals == '01':
        x = torch.rand(T,batchSize,d, device=device).round()
    else:
        raise ValueError('Invalid value for "xDataVals" arg')  
    
    y = torch.zeros(T,batchSize, dtype=torch.bool, device=device)
    
    for t in range(max(Rlist), T):
        R = Rlist[np.random.randint(0, len(Rlist))] #choose repeat interval   
        
        if interleave:
            repeatMask = torch.rand(batchSize, device=device)>P
        else:
            raise NotImplementedError
        
        if not multiRep:
            repeatMask = repeatMask*(~y[t-R]) #this changes the effective P=n/m to P'=n/(n+m)
          
        x[t,repeatMask] = x[t-R,repeatMask]            
        y[t,repeatMask] = 1
        
    y = y.unsqueeze(2).float()
    if softLabels:
        y = y*0.98 + 0.01

    return TensorDataset(x, y)


class GenRecogClassifyData():
    def __init__(self, d=None, teacher=None, datasize=int(1e4), sampleSpace=None, save=False, device='cpu'):
        if sampleSpace is None:
            x = torch.rand(datasize,d, device=device).round()*2-1
            if teacher is None:
                c = torch.randint(2,(datasize,1), device=device, dtype=torch.float)
            else:
                c = torch.empty(datasize,1, device=device, dtype=torch.float)
                for i,xi in enumerate(x):
                    c[i] = teacher(xi)
                c = (c-c.mean()+0.5).round()
            self.sampleSpace = TensorDataset(x,c)
            if save:
                if type(save) == str:
                    fname = save
                else:
                    fname = 'sampleSpace.pkl'
                torch.save(self.sampleSpace, fname)
        elif type(sampleSpace) == str:
            self.sampleSpace = torch.load(sampleSpace) 
        elif type(sampleSpace) == TensorDataset:
            self.sampleSpace = sampleSpace
            
        self.datasize, self.d = self.sampleSpace.tensors[0].shape            
        
        
    def __call__(self, T, R, P=0.5, batchSize=-1, multiRep=False, device='cpu'):
        if np.isscalar(R):
            Rlist = [R]
        else:
            Rlist = R
        
        squeezeFlag=False
        if batchSize is None:
            batchSize=1
            squeezeFlag=True
        elif batchSize < 0:
            batchSize = self.datasize/T
            
        randomSubsetIdx = torch.randperm(len(self.sampleSpace))[:T*batchSize]
        x,c = self.sampleSpace[randomSubsetIdx]
        x = x.reshape(T,batchSize,self.d)
        c = c.reshape(T,batchSize,1)
        y = torch.zeros(T,batchSize, dtype=torch.bool, device=device)
        for t in range(max(Rlist), T):    
            R = Rlist[np.random.randint(0, len(Rlist))] #choose repeat interval   
                   
            repeatMask = torch.rand(batchSize)>P   
            if not multiRep:
                repeatMask = repeatMask*(~y[t-R]) #this changes the effective P
              
            x[t,repeatMask] = x[t-R,repeatMask] 
            c[t,repeatMask] = c[t-R,repeatMask]            
            y[t,repeatMask] = 1
         
        y = y.unsqueeze(2).float()
        y = torch.cat((y,c), dim=-1)        
        data = TensorDataset(x,y)
        
        if squeezeFlag:
            data = TensorDataset(*data[:,0,:])
    
        return data
    

#%%############
### Helpers ###   
###############
def parse_xDataVals_string(xDataVals):
    assert xDataVals.lower().startswith('uniform')
    delimIdx = xDataVals.find('_')
    if delimIdx > 0:
        assert delimIdx==7
        lims = xDataVals[delimIdx+1:]
        lower = float(lims[:lims.find('_')])
        upper = float(lims[lims.find('_')+1:])
    else:
        lower = -1
        upper = 1
    return upper, lower


def prob_repeat_to_frac_novel(P, multiRep=False):
    if multiRep:
        return P
    n,m = P.as_integer_ratio()
    return 1 - float(n)/(m+n)
    

def check_recognition_data(data, R):
    """Make sure there are no spurious repeats"""
    if len(data) == 0:
        return False
    for i in range(len(data)):
        for j in range(0,i-1):
            if all(data[i][0] == data[j][0]):   
                if i-j != R:
                    print( 'bad R', i, j )
                    return False
                if not data[i][1]:
                    print( 'unmarked', i, j )
                    return False
    return True

             
def recog_chance(data):
    """Calculates expected performance if network simply guesses based on output statistics
    i.e. the number of zeroes in the data"""
    return 1-np.sum([xy[1] for xy in data], dtype=np.float)/len(data) 


def batch(generate_data, batchsize=1, batchDim=1, **dataKwargs):
    dataList = []
    for b in range(batchsize):
        dataList.append( generate_data(**dataKwargs) )
    x = torch.cat([data.tensors[0].unsqueeze(batchDim) for data in dataList], dim=batchDim)
    y = torch.cat([data.tensors[0].unsqueeze(batchDim) for data in dataList], dim=batchDim) 
    return TensorDataset(x,y)


def data_to_tensor(data, y_dtype=torch.float, device='cpu'):
    '''Convert from list of (x,y) tuples to TensorDataset'''
    x,y = zip(*data)
    return TensorDataset(torch.as_tensor(x, dtype=torch.float, device=device), 
                   torch.as_tensor(y, dtype=y_dtype, device=device))
    
    