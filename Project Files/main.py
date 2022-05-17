########################################################################################
##            ##########        ###               ###      ###          ###           ##
##            ###     ###        ###             ###       ###          ###           ##
##            ###     ###         ###           ###        ###          ###           ##
##            ###    ##            ###         ###         ###          ###           ##
##            ########              ###       ###          ################           ##
##            ###    ###             ###     ###           ################           ##
##            ###     ###             ###   ###            ###          ###           ##
##            ###    ###               #######             ###          ###           ##
##            #########                 #####              ###          ###           ##
########################################################################################
#                                  File Details
# Project
# Purpose
# Description
########################################################################################
#                                     Importer
import numpy as np
import torch
import matplotlib.pyplot as plt
import json
from pyDOE import lhs
from collections import OrderedDict
# ######################################################################################
#                                  Function files
class NeuNet(torch.nn.Module):
    def __init__(self,arr,acti,Dtype):
        super(NeuNet,self).__init__()
        self.activation = acti # torch.nn.Tanh

        layer_list = []
        for i in range(len(arr)-1):
            layer_list.extend((('layer_%d' % (i), torch.nn.Linear(arr[i], arr[i + 1])), ('activation_%d' % (i), acti())))
        #layer_list.append(('layer_%d' % (len(arr)-1), torch.nn.Linear(arr[-2],arr[-1])))
        layerDict = OrderedDict(layer_list)

        self.layers = torch.nn.Sequential(layerDict).to(dtype=Dtype)
    def forward(self,x):
        return self.layers(x)

def MSE(ten):
    return torch.mean(torch.pow(ten,2))

def EBC_data_gen(x,xi,EBC_params):
    a = EBC_params[0]
    b = EBC_params[1]
    n = EBC_params[2]
    if xi == None:
        return np.vstack([np.hstack([x[j],np.hstack([x[j]+0.1*(2*np.random.rand(1)-1)*(b-a) for i in range(n)])]) for j in range(x.shape[0])])
    else:
        return np.hstack([x,np.hstack([(((i+1)/(n+1))*xi + ((n-i)/(n+1))*x).reshape(-1,1) for i in range(n)])])
# ######################################################################################
#                                      Input
with open('NN_Dictionary.json') as json_file:
    data = json.load(json_file)
NN_arr = np.array(data["NN_array"])
Lr = data['Learning_Rate']
N_epoch = data['Num_epoch']
N_write = data['Num_write']
exec(data["Datatype"])
del json_file, data

from BVP_Dictionary import P2,P1,P0,a,b,ia,ib,alpha,beta,Num_Node_int,Num_Node_bca, Num_Node_bcb

with open('EBC_Dictionary.json') as json_file:
    data = json.load(json_file)
EBC = [data["S"],data["B"],data["E"]]
Order_ebc = data["Order"]
del json_file, data

# Initialize the Neural Network
Nn = NeuNet(NN_arr,torch.nn.Tanh,Dtype)
# Initialize the Optimiser
optimizer = torch.optim.LBFGS(
            Nn.parameters(),
            lr=Lr, 
            max_iter=20, 
            max_eval=25, 
            history_size=50,
            tolerance_grad=1e-20, 
            tolerance_change=1e-20,#1.0 * np.finfo(float).eps,
            line_search_fn="strong_wolfe"       # can be "strong_wolfe"
            )
# Initialize the Datasets
x_bca = a+0.1*(b-a)*lhs(Num_Node_bca,1).reshape(-1,1)
x_bcb = b-0.1*(b-a)*lhs(Num_Node_bcb,1).reshape(-1,1)
x_int = a+0.1*(b-a) + 0.8*(b-a)*lhs(Num_Node_int,1).reshape(-1,1)
x_tot = np.vstack([x_bca,x_int,x_bcb])
x_ebc1 = np.vstack([a+(x_bca.T-a)*i for i in np.linspace(1/(Order_ebc+1),Order_ebc/(Order_ebc+1),Order_ebc)])
x_ebc2 = np.vstack([b-(b-x_bcb.T)*i for i in np.linspace(Order_ebc/(Order_ebc+1),1/(Order_ebc+1),Order_ebc)])
if EBC[0]=="Weighted":
    from Weights import W1,W2
    X_bca = EBC_data_gen(x_bca,a , [a,b,Order_ebc])
    X_bcb = EBC_data_gen(x_bcb,b , [a,b,Order_ebc])
    X_int = EBC_data_gen(x_int,None, [a,b,Order_ebc])
    X_d = torch.tensor(np.vstack([X_bca,X_int,X_bcb]),requires_grad=True,dtype=Dtype)
    x_orig = np.vstack([x_bca,x_int,x_bcb])
    w_a = torch.tensor(W1(x_orig),dtype = Dtype)
    w_b = torch.tensor(W2(x_orig),dtype = Dtype)
    del x_orig,x_bca,x_bcb,x_int,x_ebc1,x_ebc2,X_bca,X_bcb,X_int
    
else: # Datasets
    X_bca = torch.tensor(EBC_data_gen(x_bca,a , [a,b,Order_ebc]),requires_grad=True,dtype=Dtype)
    X_bcb = torch.tensor(EBC_data_gen(x_bcb,b , [a,b,Order_ebc]),requires_grad=True,dtype=Dtype)
    X_d = torch.tensor(x_tot,requires_grad=True,dtype=Dtype)
#     X_a = torch.tensor(x_ebc1,requires_grad=True, dtype=Dtype)
#     X_b = torch.tensor(x_ebc2,requires_grad=True, dtype=Dtype)
#     X_d = torch.tensor(x_int,requires_grad=True,dtype=Dtype)
# del x_bca,x_bcb,x_int,x_ebc1,x_ebc2,x_ebc0,x_ebc

from Taylor import R_arr
if EBC[0]=="Weighted":
    R_ten_a = torch.tensor(R_arr(X_d.clone().detach().numpy(),"a",ia,[a,b,Order_ebc]),dtype=Dtype)
    R_ten_b = torch.tensor(R_arr(X_d.clone().detach().numpy(),"b",ib,[a,b,Order_ebc]),dtype=Dtype)
else:
    R_ten_a = torch.tensor(R_arr(X_bca.clone().detach().numpy(),"a",ia,[a,b,Order_ebc]),dtype=Dtype)
    R_ten_b = torch.tensor(R_arr(X_bcb.clone().detach().numpy(),"b",ib,[a,b,Order_ebc]),dtype=Dtype)
# ######################################################################################
#                                  Function files
def Differentiator(xi):
    ui = Nn.forward(xi.reshape(-1,1))
    ux  = torch.autograd.grad(ui,xi,grad_outputs=torch.ones_like(ui),retain_graph=True,create_graph=True)[0]
    uxx = torch.autograd.grad(ux,xi,grad_outputs=torch.ones_like(ux),retain_graph=True,create_graph=True)[0]
    return [ui,ux,uxx]

def Loss_w():
    optimizer.zero_grad()
    loss = torch.tensor([0.0],dtype = Dtype,requires_grad=True)
    for i in range(X_d.shape[0]):
        xi = X_d[i][0]
        ui = Nn.forward(xi.reshape(-1,1))
        ux  = torch.autograd.grad(ui,xi,grad_outputs=torch.ones_like(ui),retain_graph=True,create_graph=True)[0]
        uxx = torch.autograd.grad(ux,xi,grad_outputs=torch.ones_like(ux),retain_graph=True,create_graph=True)[0]
        P2_t = P2(xi)
        P1_t = P1(xi)
        P0_t = P0(xi)
        uf = Nn.forward(X_d[i:i+1].T)
        l0 = MSE(P2_t*uxx + P1_t*ux + P0_t*xi)
        l1 = w_a[i]*MSE(torch.dot(R_ten_a[i],uf.flatten())-alpha)
        l2 = w_b[i]*MSE(torch.dot(R_ten_b[i],uf.flatten())-beta)
        
        loss = loss + l0+l1+l2
    loss.backward(retain_graph=True)
    if(epoc%25==0):
        print('\rEpoch : %d\tLoss: %3.2e\tL0: %3.2e\tL1: %3.2e\tL2: %3.2e'%(epoc,loss,l0,l1,l2))
        loss_log.append([epoc,loss.item(),l0.item(),l1.item(),l2.item()])
    return loss
def Loss_d():
    optimizer.zero_grad()
    loss = torch.tensor([0.0],dtype = Dtype,requires_grad=True)
    l1 = torch.tensor([0.0],dtype = Dtype,requires_grad=True)
    l2 = torch.tensor([0.0],dtype = Dtype,requires_grad=True)
    for i in range(X_bca.shape[0]):
        xf = X_bca[i:i+1]
        uf = Nn.forward(xf.reshape(-1,1))
        l1=l1+(1/Num_Node_bca)*MSE(torch.dot(R_ten_a[i],uf.flatten())-alpha)
    for i in range(X_bcb.shape[0]):
        xf = X_bcb[i:i+1]
        uf = Nn.forward(xf.reshape(-1,1))
        l2=l2+(1/Num_Node_bcb)*MSE((torch.dot(R_ten_b[i],uf.flatten())-beta))
    x = X_d[:,0:1]
    u = Nn.forward(x)
    ux = torch.autograd.grad(u,x,grad_outputs=torch.ones_like(u),retain_graph=True,create_graph=True)[0]
    uxx = torch.autograd.grad(ux,x,grad_outputs=torch.ones_like(ux),retain_graph=True,create_graph=True)[0]
    l0 = MSE(uxx*P2(x)+ux*P1(x)+u*P0(x))
    loss = l0 + l1 + l2
    loss.backward(retain_graph=True)
    if(epoc%25==0):
        print('\rEpoch : %d\tLoss: %3.2e\tL0: %3.2e\tL1: %3.2e\tL2: %3.2e'%(epoc,loss,l0,l1,l2))
        loss_log.append([epoc,loss.item(),l0.item(),l1.item(),l2.item()])
    return loss
# Initialize Neural Network and 
epoc = 1
if EBC[0]=="Weighted":
    Loss = Loss_w
else:
    Loss = Loss_d

print(Loss())
def Loss_plain():
    optimizer.zero_grad()
    loss = torch.tensor([0.0],dtype = Dtype,requires_grad=True)
    for i in range(X_d.shape[0]):
        xi = X_d[i][0]
        ui = Nn.forward(xi.reshape(-1,1))
        ux  = torch.autograd.grad(ui,xi,grad_outputs=torch.ones_like(ui),retain_graph=True,create_graph=True)[0]
        uxx = torch.autograd.grad(ux,xi,grad_outputs=torch.ones_like(ux),retain_graph=True,create_graph=True)[0]
        P2_t = P2(xi)
        P1_t = P1(xi)
        P0_t = P0(xi)
        l0 = MSE(P2_t*uxx + P1_t*ux + P0_t*xi)
        l1 = MSE(Nn.forward(torch.tensor([a],requires_grad=True,dtype=Dtype).reshape(-1,1))-alpha)
        l2 = MSE(Nn.forward(torch.tensor([b],requires_grad=True,dtype=Dtype).reshape(-1,1))-beta)
        
        loss = loss + l0+l1+l2
    loss.backward(retain_graph=True)
    if(epoc%25==0):
        print('\rEpoch : %d\tLoss: %3.2e\tL0: %3.2e\tL1: %3.2e\tL2: %3.2e'%(epoc,loss,l0,l1,l2))
        loss_log.append([epoc,loss.item(),l0.item(),l1.item(),l2.item()])
    return loss
print(Loss_plain())
########################################################################################
#                                       Body
########################################################################################
#                                   Visualisation
########################################################################################
#                                      Output
########################################################################################
#                        AUTHOR: VEERENDRA HARSHAL BUDHI
########################################################################################
