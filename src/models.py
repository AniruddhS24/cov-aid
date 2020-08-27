import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class RNN(nn.Module):
    def __init__(self,inp_size,hid_size,op_size,num_layers):
        super(RNN,self).__init__()
        self.rnn = nn.LSTM(inp_size,hid_size,num_layers=num_layers,batch_first=True,dropout=0.1)
        self.dense = nn.Linear(hid_size,op_size)

        nn.init.xavier_uniform_(self.dense.weight)

    def forward(self, x):
        ho = torch.zeros(self.rnn.num_layers,x.size(0),self.rnn.hidden_size)
        co = torch.zeros(self.rnn.num_layers,x.size(0),self.rnn.hidden_size)
        op,(h,c) = self.rnn(x,(ho,co))
        #op is (batch_size, seq_len, hid_size)
        op = self.dense(op)
        #op is (batch_size, seq_len, 1)
        return op

class COVIDTimeSeriesModel():
    def __init__(self,inp_size,hid_size,op_size,num_layers, scale):
        self.scl = scale
        self.net = RNN(inp_size,hid_size,op_size,num_layers)


    def predict(self, x,st_idx=None):
        self.net.eval()
        pred = self.net.forward(x)
        #pred = (pred * (self.scl[0] - self.scl[1]) + self.scl[2]*self.scl[1])/self.scl[2]
        pred = self.invert_scale(pred,st_idx)
        return pred

    def invert_scale(self, x, st_idx=None):
        if st_idx==None:
            return (x * (self.scl[0] - self.scl[1]) + self.scl[2] * self.scl[1]) / self.scl[2]
        else:
            return (x * (self.scl[0][st_idx] - self.scl[1][st_idx]) + self.scl[2]*self.scl[1][st_idx])/self.scl[2]

def get_batches(x_train,y_train,inp_size,batch_size,seq_len,offset,num_samples):
    x_smp = torch.zeros(num_samples,seq_len,inp_size)
    y_smp = torch.zeros(num_samples,seq_len)
    pos_a = torch.floor(torch.rand(num_samples)*x_train.shape[0]).long()
    pos_b = torch.floor(torch.rand(num_samples)*(x_train.shape[1]-seq_len-offset)).long()
    for i in range(num_samples):
        x_smp[i] = torch.from_numpy(x_train[pos_a[i],pos_b[i]:pos_b[i]+seq_len])
        y_smp[i] = torch.from_numpy(y_train[pos_a[i],pos_b[i]+offset:pos_b[i]+offset+seq_len])

    x,y = [],[]
    for i in range(0,num_samples,batch_size):
        if num_samples-i <= batch_size:
            x.append(x_smp[i:])
            y.append(y_smp[i:])
        else:
            x.append(x_smp[i:i+batch_size])
            y.append(y_smp[i:i+batch_size])

    return x,y,len(y)

def train_rnn(x_train,y_train,scale,batch_size,seq_len,offset,num_samples):

    # >> MODEL SETTINGS:
    inp_size = 2
    hid_size = 25
    op_size = 1
    num_layers = 3
    model = COVIDTimeSeriesModel(inp_size,hid_size,op_size,num_layers,scale)
    # >> TRAINING SETTINGS:
    EPOCHS = 17
    lr = 0.0005
    criterion = nn.SmoothL1Loss() #use smoothL1?
    #criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.net.parameters(), lr=lr)
    for epoch in range(EPOCHS):
        x,y,num_batches = get_batches(x_train,y_train,inp_size,batch_size,seq_len,offset,num_samples)
        tot_loss = 0.0
        for i in range(num_batches):
            model.net.train()
            optimizer.zero_grad()
            pred = model.net.forward(x[i]).squeeze()
            loss = criterion(pred,y[i])
            tot_loss += loss
            loss.backward()
            optimizer.step()
        '''if epoch%5==0:
            for i in range(10, 30):
                prd = model.predict(torch.from_numpy(x_train[i:i + 1, :]).float()).squeeze().detach().numpy()
                appd = np.concatenate((model.invert_scale(y_train[i, :]), prd[-offset:]), 0)
                # prd = model.predict(torch.from_numpy(x_data[i:i+1, :]).float(),i).squeeze().detach().numpy()
                # appd = np.concatenate((model.invert_scale(y_data[i,:],i),prd[-offst:]),0)
                plt.plot(appd)
            plt.show()'''
        print("Epoch: %-3d   Loss: %10.6f" % (epoch,tot_loss/num_batches))

    return model