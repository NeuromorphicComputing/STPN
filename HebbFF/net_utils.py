import os, time, importlib
import pdb

from typhon.utils import Timer

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

from STPN.Scripts.utils import RESULTS

#%%#################
### Base Classes ###
####################
   
class NetworkBase(nn.Module): 
    def __init__(self):
        """Subclasses must either implement self.loss_fn or override self.average_loss()"""
        super(NetworkBase, self).__init__()
        self.eval() #start the module in evaluation mode
        self.hist = None
        self.name = self.__class__.__name__
        self.loss_fn = None 
        self.acc_fn = None
    
    
    def fit(self, train_fn, *args, **kwargs):
        if train_fn == 'dataset':
            train_fn = train_dataset            
        elif train_fn == 'infinite':
            train_fn = train_infinite           
        elif train_fn == 'curriculum':
            train_fn = train_curriculum
        elif train_fn == 'multiR':
            train_fn = train_multiR_curriculum
        elif hasattr(train_fn, '__call__'):
            pass        
        else:
            ValueError("train_fn must be a function or valid keyword")
            
        folder = os.path.normpath(kwargs.pop('folder', ''))
        filename = kwargs.pop('filename', None)
        if filename is not None and not os.path.commonprefix([os.getcwd()[::-1],folder[::-1]]):
            if folder != '' and not os.path.exists(folder):
                os.makedirs(folder)
            os.chdir(folder)
        # so filename seems to be used for storing model and self.hist (which i dont know what it is)
        self.autosave = Autosave(self, filename)
        if kwargs.pop('overwrite', False):
            self.autosave.lastSave = 0
        self.autosave(force=True)
        if filename is not None and folder is not None:
            # so this writes to tensorboad using same filename as autosave, except last 3 chars
            # i assume this is the ckp extension, they use .pkl
            writerPath = os.path.join('tensorboard', self.autosave.filename[:-4])
            # this is in case you forgot sth wrt current working directory?
            # so maybe folder needs to have from the top
            if not os.path.commonprefix([os.getcwd()[::-1],folder[::-1]]):
                writerPath = os.path.join(folder, writerPath)
            self.writer = SummaryWriter(writerPath) #remove extension from filename
                
        self.optimizer = torch.optim.Adam(self.parameters(), lr=kwargs.pop('learningRate', 1e-3))       
        self.train() #put Module in train mode (e.g. for dropout)
        with Timer() as timer:
            train_fn(self, *args, **kwargs)                   
        self.eval() #put Module back in evaluation mode
        
        self.autosave(force=True)
        self.hist['time'] = timer.elapsed        

    
    def _train_epoch(self, trainData, validBatch=None, batchSize=1, earlyStop=True, earlyStopValid=False, validStopThres=None):
        for b in range(0, trainData.tensors[0].shape[1], 1 if batchSize is None else batchSize):  #trainData is shape [T,D,N]
            trainBatch = trainData[:,b,:] if batchSize is None else trainData[:,b:b+batchSize,:] 
            
            self.optimizer.zero_grad()
            out = self.evaluate(trainBatch) #expects shape [T,batchSize,N] or [T,N] if batchSize=None
            loss = self.average_loss(trainBatch, out=out)            
            loss.backward()
            self.optimizer.step() 
            
            self._monitor(trainBatch, validBatch=validBatch, out=out, loss=loss)  
            
            if earlyStopValid and len(self.hist['valid_loss'])>1 and self.hist['valid_loss'][-1] > self.hist['valid_loss'][-2]:
                return True
            if validStopThres is not None and self.hist['valid_acc'][-1]>validStopThres:
                return True
            if earlyStop and sum(self.hist['train_acc'][-5:]) >= 4.99: #not a proper early-stop criterion but useful for infinite data regime
                return True
        return False
                       
    
    def evaluate(self, batch): #TODO: figure out nice way to infer shape of y and don't pass in
        """batch is (X,Y) tuple of Tensors, with X.shape=[T,B,N] or [T,N], Y.shape=[T,1]"""
        out = torch.empty_like(batch[1]) #shape=[T,B,Ny] if minibatched, otherwise [T,Ny] 
        for t,x in enumerate(batch[0]): #shape=[T,B,Nx] or [T,Nx]
            if len(x.shape) > 1:
                assert x.shape[0]  == 1
                x = x.squeeze(0)
            out[t] = self(x) #shape=[B,Nx] or [Nx]
        return out


    def evaluate_debug(self, batch):
        """Override this"""
        raise NotImplementedError


    def accuracy(self, batch, out=None):
        """batch is (x,y) tuple of Tensors, with x.shape=[T,B,Nx] or [T,Nx]"""
        if out is None:
            # assert batch.shape[0] == 1
            # batch = batch.squeeze(0)
            out = self.evaluate(batch)
        return self.acc_fn(out, batch[1])
   
                
    def average_loss(self, batch, out=None):
        """batch is (x,y) tuple of Tensors, with x.shape=[T,B,Nx] or [T,Nx]"""
        if out is None:
            out = self.evaluate(batch)
        return self.loss_fn(out, batch[1])
    
    
    @torch.no_grad()
    def _monitor_init(self, trainBatch, validBatch=None):
        if self.hist is None:
            self.hist = {'epoch' : 0,
                         'iter' : -1, #gets incremented when _monitor() is called
                         'train_loss' : [], 
                         'train_acc' : [],
                         'grad_norm': []}
            if validBatch:
                self.hist['valid_loss'] = []
                self.hist['valid_acc']  = []
                
            self._monitor(trainBatch, validBatch=validBatch)
        else: 
            print('Network already partially trained. Continuing from iter {}'.format(self.hist['iter']))    
       

    @torch.no_grad()
    def _monitor(self, trainBatch, validBatch=None, out=None, loss=None, acc=None):  
        #TODO: rewrite with same format as SummaryWriter and automatically write to both 
        self.hist['iter'] += 1 
        
        if self.hist['iter']%10 == 0: #TODO: allow choosing monitoring interval
            if out is None:
                out = self.evaluate(trainBatch)                
            if loss is None:
                loss = self.average_loss(trainBatch, out)
            if acc is None:
                acc = self.accuracy(trainBatch, out)
            gradNorm = self.grad_norm()
            
            self.hist['train_loss'].append( loss.item() )
            self.hist['train_acc'].append( acc.item() ) 
            self.hist['grad_norm'].append( gradNorm )            
            if hasattr(self, 'writer'):
                self.writer.add_scalar('train/loss', loss.item(), global_step=self.hist['iter'])   
                self.writer.add_scalar('train/acc', acc.item(), global_step=self.hist['iter'])   
                self.writer.add_scalar('info/grad_norm', gradNorm, global_step=self.hist['iter'])
            displayStr = 'Iter:{} grad:{:.1f} train_loss:{:.4f} train_acc:{:.3f}'.format(self.hist['iter'], gradNorm, loss, acc)                    
            
            if validBatch is not None:
                out = self.evaluate(validBatch)
                loss = self.average_loss(validBatch, out=out)  
                acc = self.accuracy(validBatch, out=out) 
                
                self.hist['valid_loss'].append( loss.item() )                
                self.hist['valid_acc'].append( acc.item() )  
                if hasattr(self, 'writer'):                                 
                    self.writer.add_scalar('validation/loss', loss.item(), global_step=self.hist['iter'])
                    self.writer.add_scalar('validation/acc', acc.item(), global_step=self.hist['iter']) 
                displayStr += ' valid_loss:{:.4f} valid_acc:{:.3f}'.format(loss, acc)

            print(displayStr)
            self.autosave()            
            return out 
           
            
    def grad_norm(self):
        return torch.cat([p.reshape(-1,1) for p in self.parameters() if p.requires_grad]).norm().item()
        
           
    def save(self, filename, overwrite=False):
        """Doing this the recommended way, see: 
            https://pytorch.org/docs/stable/notes/serialization.html#recommend-saving-models
        To load, initialize a new network of the same type and do net.load(filename) or use load_from_file()        
        """   
        device = next(self.parameters()).device
        if device != torch.device('cpu'):   
            self.to('cpu') #make sure net is on CPU to prevent loading issues from GPU 
        
        directory = os.path.join(RESULTS, os.path.split(filename)[0])
        if directory != '' and not os.path.exists(directory):
            print(f"Creating directory {directory} to store results")
            pdb.set_trace()
            os.makedirs(directory)
            
        if not overwrite:
            base, ext = os.path.splitext(filename)
            n = 2
            while os.path.exists(filename):
                # adds n until same file doesn't exist. not to overwrite
                filename = '{}_({}){}'.format(base, n, ext)
                n+=1
        
        state = self.state_dict()
        state.update({'hist':self.hist})
        torch.save(state, filename) 
        
        if device != torch.device('cpu'):
            self.to(device) #put the net back to device it started from
        return filename
        
    
    def load(self, filename):
        state = torch.load(filename)
        self.hist = state.pop('hist')
        
        try:
            self.load_state_dict(state, strict=False)
        except:
            for k in state.keys():
                checkptParam = state[k]
                                               
                kList = k.split('.')
                localParam = getattr(self, kList[0])
                for k in kList[1:]:
                    localParam = getattr(localParam, k)    
                    
                if localParam.shape != checkptParam.shape:
                    print('WARNING: shape mismatch between local {} ({}) and checkpoint ({}). Setting local {} to be {}'.format(k, localParam.shape, checkptParam.shape, k, checkptParam.shape))
                    localParam.data = torch.empty_like(checkptParam)
            self.load_state_dict(state, strict=False)



class StatefulBase(NetworkBase):
    """Networks that have an internal state that evolves with time, e.g. RNNs, LSTMs, HebbNets"""
    def reset_state(self):
        raise NotImplementedError


    def evaluate(self, batch, preserveState=False):
        """NOTE: current state of A will be lost!"""
        self.reset_state()
        out = super(StatefulBase,self).evaluate(batch)
        return out


#%%#############
### Training ###
################
    
def train_dataset(net, trainData, validBatch=None, epochs=100, batchSize=None, earlyStop=True, validStopThres=None, earlyStopValid=False): 
    """trainData is a TensorDataset"""    
    net._monitor_init(trainData[:,0,:], validBatch=validBatch)
    while net.hist['epoch'] < epochs:
        net.hist['epoch'] += 1
        converged = net._train_epoch(trainData, validBatch=validBatch, batchSize=batchSize, earlyStop=earlyStop, validStopThres=validStopThres, earlyStopValid=earlyStopValid)  
        if converged:
            print('Converged, stopping early.')
            break                                    
        
    
def train_infinite(net, gen_data, iters=float('inf'), batchSize=None, earlyStop=True):     
    trainBatch = gen_data()[:,0,:] if batchSize is None else gen_data()[:,:batchSize,:] 
    net._monitor_init(trainBatch)
    
    iter0 = net.hist['iter'] 
    
    while net.hist['iter'] < iters:
        trainCache = gen_data() #generate a large cache of data, then minibatch over it
        converged = net._train_epoch(trainCache, batchSize=batchSize, earlyStop=earlyStop)    
        
        #TODO: this is a hack to force the net to add at least 5 entries to hist['acc'] 
        #before evaluating convergence. Ensures that if we're starting from a converged network
        #we re-evaluate on 5 *new* entries.        
        if net.hist['iter'] < iter0+50: #hist['acc'] updated every 10 iters      
            converged = False
        
        if converged:
            print('Converged, stopping early.')
            break   

      
def train_curriculum(net, gen_data, iters=float('inf'), itersToQuit=2e6, batchSize=None, R0=1, Rf=float('inf'), increment=lambda R:R+1):          
    R = R0
    trainBatch = gen_data(R)[:,0,:] if batchSize is None else gen_data(R)[:,:batchSize,:] 
    net._monitor_init(trainBatch)
    if 'increment_R' not in net.hist:
        net.hist['increment_R'] = [(0, R)]
    itersSinceIncrement = 0
    latestIncrementIter = net.hist['iter']

    converged = False
    while net.hist['iter'] < iters and R < Rf and itersSinceIncrement < itersToQuit:
        if hasattr(net, 'writer'):
            net.writer.add_scalar('info/R', R, net.hist['iter'])
        if converged:
            R = increment(R)
            latestIncrementIter = net.hist['iter']
            print('Converged. Setting R<--{} \n'.format(R))
            net.hist['increment_R'].append( (net.hist['iter'], R) )    
            net.autosave(force=True)                      
                 
        trainCache = gen_data(R)   
        converged = net._train_epoch(trainCache, batchSize=batchSize)  
        itersSinceIncrement = net.hist['iter'] - latestIncrementIter
        
        #TODO: this is a hack to force the net to add at least 5 entries to hist['acc'] 
        #before evaluating convergence. Ensures that once it's converged on R, it doesn't
        #assume it's converged on R+1 since the convergence depends on the last 5 entries of hist['acc']        
        if net.hist['iter'] < net.hist['increment_R'][-1][0]+50:
            converged = False 


def train_multiR_curriculum(net, gen_data, Rlo=1, Rhi=2, spacing=range, batchSize=1, itersToQuit=2e6, increment=None ):   
    """
    Default: train on [1..Rchance]
    """
    if increment is None:
        def increment(Rlo, Rhi):
            return Rlo, Rhi+1           
    
    Rlist = spacing(Rlo, Rhi)
    trainBatch = gen_data(Rlist)[:,0,:] if batchSize is None else gen_data(Rlist)[:,:batchSize,:] 
    validBatch = gen_data([Rlist[-1]])[:,0,:] if batchSize is None else gen_data(Rlist)[:,:batchSize,:] 
    net._monitor_init(trainBatch, validBatch)
    if 'increment_R' not in net.hist:
        net.hist['increment_R'] = [(0, Rlist[-1])]
    itersSinceIncrement = 0
    latestIncrementIter = net.hist['iter']
        
    converged = False
    itersSinceIncrement = 0
    while itersSinceIncrement < itersToQuit:
        if hasattr(net, 'writer'):
            net.writer.add_scalar('info/Rlo', Rlist[0], net.hist['iter'])
            net.writer.add_scalar('info/Rhi', Rlist[-1], net.hist['iter'])
        if converged:
            Rlo, Rhi = increment(Rlo, Rhi)
            Rlist = spacing(Rlo, Rhi)
            latestIncrementIter = net.hist['iter']
            print('acc(Rchance)>0.55. Setting Rlist=[{}...{}] \n'.format(Rlist[0], Rlist[-1]))
            net.hist['increment_R'].append( (net.hist['iter'], Rlist[0], Rlist[-1]) )    
            net.autosave(force=True)  
        itersSinceIncrement = net.hist['iter'] - latestIncrementIter    
   
        trainCache = gen_data(Rlist)   
        
        #TODO: this is dumb, I'm generating 1000x more data than I'm using for validation
        validBatch = gen_data([Rlist[-1]])[:,0,:] if batchSize is None else gen_data(Rlist)[:,:batchSize,:] 
        converged = net._train_epoch(trainCache, validBatch, batchSize=batchSize, earlyStop=False, validStopThres=0.55)
               

#%%############
### Metrics ###
###############
        
def nan_mse_loss(out, y, reduction='mean'):
    """Computes MSE loss, ignoring any nan's in the data"""
    idx = ~torch.isnan(y)
    return F.mse_loss(out[idx], y[idx], reduction=reduction)


def nan_bce_loss(out, y, reduction='mean'):
    idx = ~torch.isnan(y)
    return F.binary_cross_entropy(out[idx], y[idx], reduction=reduction)


def nan_recall_accuracy(out, y):
    """Computes accuracy on the recall task, ignoring any nan's in the data"""
    idx = ~torch.isnan(y).all(dim=-1)
    return (out[idx].sign()==y[idx]).all(dim=-1).float().mean()    
 
    
def binary_thres_classifier_accuracy(out, y, thres=0.5):
    """Accuracy for binary-classifier-like task"""
    return ((out>thres) == y.bool()).float().mean()


def binary_classifier_accuracy(out, y):
    """Accuracy for binary-classifier-like task"""
    return (out.round() == y.round()).float().mean() #round y in case using soft labels


def nan_binary_classifier_accuracy(out, y):
    idx = ~torch.isnan(y)
    return binary_classifier_accuracy(out[idx], y[idx])    

    
#%%############    
### Helpers ### 
############### 
import numpy as np #TODO: move to torch, remove numpy  
    
def check_dims(W, B=None):
    """Verify that the dimensions of the weight matrices are compatible"""
    dims = [W[0].shape[1]]
    for l in range(len(W)-1):
        assert(W[l].shape[0] == W[l+1].shape[1]) 
        dims.append( W[l].shape[0] )
        if B: 
            assert(W[l].shape[0] == B[l].shape[0])
    if B: 
        assert(W[-1].shape[0] == B[-1].shape[0]) 
    dims.append(W[-1].shape[0])
    return dims
        

def random_weight_init(dims, bias=False):
    W,B = [], []
    for l in range(len(dims)-1):
        W.append( np.random.randn(dims[l+1], dims[l])/np.sqrt(dims[l]) )
        if bias:        
            B.append( np.random.randn(dims[l+1]) )
        else:
            B.append( np.zeros(dims[l+1]) )
    check_dims(W,B) #sanity check
    return W,B
 
     
class Autosave():
    def __init__(self, net, filename, saveInterval=3600):
        self.filename = filename
        self.lastSave = -1
        self.net = net
        self.saveInterval = saveInterval
              
    def __call__(self, force=False):
        if self.filename is not None and (force or time.time()-self.lastSave>self.saveInterval):
            self.filename = self.net.save(self.filename, overwrite=False if self.lastSave<0 else True)
            self.lastSave = time.time()


def load_from_file(fname, NetClass=None, dims=None):
    """If NetClass and dims not provided, filename format must be of the format NetName[dims]*.pkl 
    where NetName is the name of a class in networks.py, dims is a comma-separated list of integers, 
    * can be anything, and .pkl is literal e.g. HebbNet[25,50,1]_R=1.pkl"""    
    folder, filename = os.path.split(fname)
    
    if NetClass is None:
        idx = filename.find('[')
        if idx == -1:
            idx = filename.find('_')
        if idx == -1:
            raise ValueError('Invalid filename')
        # so in order to plot this, the filename has to be [model_name]_*
        NetClass = getattr(importlib.import_module('STPN.HebbFF.networks'), filename[:idx])

    if dims is None:
        dims = filename[filename.find('[')+1:filename.find(']')]
        dims = [int(d) for d in dims.split(',')]
    
    if NetClass.__name__ == 'HebbDiags':
        idx = filename.find('Ng=')+3
        Ng = ''
        while filename[idx].isdigit():
            Ng += filename[idx]
            idx += 1
        net = NetClass([dims, int(Ng)])
    elif NetClass.__name__ == 'HebbFeatureLayer':
        idx = filename.find('_Nx=')+4
        Nx = ''
        while filename[idx].isdigit():
            Nx += filename[idx]
            idx += 1
        net = NetClass(dims, Nx=int(Nx))
    else:
        net = NetClass(dims)
    
    net.load(fname)
    return net    

            



        

    

