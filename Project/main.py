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
import sys
import matplotlib.pyplot as plt
import json
from pyDOE import lhs
from collections import OrderedDict
from Resources import plotting
#import plotting
import time
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

#   Neural Network Dictionary Input
with open('Input/NN_Dictionary.json') as json_file:
    data = json.load(json_file)
NN_arr = np.array(data["NN_array"])
Lr = data['Learning_Rate']
N_epoch = data['Num_epoch']
N_write = data['Num_write']
N_test = data["Num_test"]
exec(data["Datatype"])
del json_file, data

#   BVP Dictionary Input
from Input.BVP_Dictionary import Exact,bvp_string,head_string,P2,P1,P0,Q,a,b,ia,ib,alpha,beta,Num_Node_int,Num_Node_bca, Num_Node_bcb

#   EBC Dictionary Input
with open('Input/EBC_Dictionary.json') as json_file:
    data = json.load(json_file)
EBC = [data["S"],data["B"],data["E"]]
Order_ebc = data["Order"]
del json_file, data
if EBC[2] == "Evergreen":
    N_epoch_EBC = N_epoch
    N_epoch_Plain = 0
else:
    N_epoch_EBC = int(N_epoch*0.4)
    N_epoch_Plain = N_epoch - N_epoch_EBC

N_epoch_EBC = 500
N_epoch_Plain = 0# N_epoch
#   Print Details of the Simulation
print("\nBoundary Value Problem\n"+bvp_string)
print("\nNeural Network details: \n  Neural Network architecture : %s\n  Learning Rate : %f"%(np.array2string(NN_arr),Lr))
print("\nEBC Details\n  Schema used : %s\n  Basis Function used : %s\n  Extent of EBC : %s\n  Order of EBC : %d\n\n"%(EBC[0],EBC[1],EBC[2],Order_ebc))

#   Initialize the Neural Network
Nn = NeuNet(NN_arr,torch.nn.Tanh,Dtype)
# Initialize the Optimiser
optimizer = torch.optim.LBFGS(
            Nn.parameters(),
            lr=Lr,
            max_iter=20,
            max_eval=20,
            history_size=50,
            tolerance_grad=1e-20,
            tolerance_change=1e-20,#1.0 * np.finfo(float).eps,
            line_search_fn="strong_wolfe"       # can be "strong_wolfe"
            )

#   Initialize the Datasets
x_bca = a+0.1*(b-a)*lhs(Num_Node_bca,1).reshape(-1,1)
x_bcb = b-0.1*(b-a)*lhs(Num_Node_bcb,1).reshape(-1,1)
x_int = a+0.1*(b-a) + 0.8*(b-a)*lhs(Num_Node_int,1).reshape(-1,1)
x_tot = np.vstack([x_bca,x_int,x_bcb])
x_ebc1 = np.vstack([a+(x_bca.T-a)*i for i in np.linspace(1/(Order_ebc+1),Order_ebc/(Order_ebc+1),Order_ebc)])
x_ebc2 = np.vstack([b-(b-x_bcb.T)*i for i in np.linspace(Order_ebc/(Order_ebc+1),1/(Order_ebc+1),Order_ebc)])

x_test = torch.tensor(np.linspace(a,b,N_test),dtype=Dtype,requires_grad=True).reshape(-1,1)
x_test_bc = torch.tensor([[a],[b]],requires_grad=True,dtype = Dtype)

if EBC[0]=="Weighted":
    from Resources.Weights import W1,W2
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

exec("from Bases."+EBC[1]+" import R_arr")

if EBC[0]=="Weighted":
    R_ten_a = torch.tensor(R_arr(X_d.clone().detach().numpy(),"a",ia,[a,b,Order_ebc]),dtype=Dtype)
    R_ten_b = torch.tensor(R_arr(X_d.clone().detach().numpy(),"b",ib,[a,b,Order_ebc]),dtype=Dtype)
else:
    R_ten_a = torch.tensor(R_arr(X_bca.clone().detach().numpy(),"a",ia,[a,b,Order_ebc]),dtype=Dtype)
    R_ten_b = torch.tensor(R_arr(X_bcb.clone().detach().numpy(),"b",ib,[a,b,Order_ebc]),dtype=Dtype)
# ######################################################################################
#                                  Function files
loss_log = []
loss_ch=torch.tensor([0.0,0.0,0.0],dtype=Dtype,requires_grad=True)
def Differentiator(xi):
    ui = Nn.forward(xi.reshape(-1,1))
    ux  = torch.autograd.grad(ui,xi,grad_outputs=torch.ones_like(ui),retain_graph=True,create_graph=True)[0]
    uxx = torch.autograd.grad(ux,xi,grad_outputs=torch.ones_like(ux),retain_graph=True,create_graph=True)[0]
    return [ui,ux,uxx]
def Test_Loss_Eval(Nn):
    u = Nn.forward(x_test.reshape(-1,1))
    ux  = torch.autograd.grad(u,x_test,grad_outputs=torch.ones_like(u),retain_graph=True,create_graph=True)[0]
    uxx = torch.autograd.grad(ux,x_test,grad_outputs=torch.ones_like(ux),retain_graph=True,create_graph=True)[0]
    
    PDE_Loss = MSE(P2(x_test)*uxx + P1(x_test)*ux + P0(x_test)*u + Q(x_test))
    u_bc = Nn.forward(x_test_bc)
    ux_bc =  torch.autograd.grad(u_bc,x_test_bc,grad_outputs=torch.ones_like(u_bc),retain_graph=True,create_graph=True)[0]
    BC_arr = np.zeros((2,2))
    BC_arr[0:1,:] = u_bc.detach().numpy().transpose()
    BC_arr[1:2,:] = ux_bc.detach().numpy().transpose()
    BC_Loss = (BC_arr[ia,0]-alpha)**2+(BC_arr[ib,1]-beta)**2
    return [PDE_Loss,BC_Loss]
def Loss_w():
    optimizer.zero_grad()
    loss = torch.tensor([0.0],dtype = Dtype,requires_grad=True)
    global loss_ch
    loss_ch=torch.tensor([0.0,0.0,0.0],dtype=Dtype,requires_grad=True)
    for i in range(X_d.shape[0]):
        xi = X_d[i][0]
        ui = Nn.forward(xi.reshape(-1,1))
        ux  = torch.autograd.grad(ui,xi,grad_outputs=torch.ones_like(ui),retain_graph=True,create_graph=True)[0]
        uxx = torch.autograd.grad(ux,xi,grad_outputs=torch.ones_like(ux),retain_graph=True,create_graph=True)[0]
        P2_t = P2(xi)
        P1_t = P1(xi)
        P0_t = P0(xi)
        Q_t = Q(xi)
        uf = Nn.forward(X_d[i:i+1].T)
        l0 = MSE(P2_t*uxx + P1_t*ux + P0_t*ui + Q_t)
        l1 = w_a[i]*MSE(torch.dot(R_ten_a[i],uf.flatten())-alpha)
        l2 = w_b[i]*MSE(torch.dot(R_ten_b[i],uf.flatten())-beta)

        loss = loss + l0+l1+l2
        loss_ch = loss_ch+(torch.tensor([l0,l1,l2])/X_d.shape[0])
    loss.backward(retain_graph=True)
    return loss
def Loss_d():
    optimizer.zero_grad()
    loss = torch.tensor([0.0],dtype = Dtype,requires_grad=True)
    global loss_ch
    loss_ch=torch.tensor([0.0,0.0,0.0],dtype=Dtype,requires_grad=True)
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
    l0 = MSE(uxx*P2(x)+ux*P1(x)+u*P0(x)+Q(x))
    loss = l0 + l1 + l2
    loss_ch = loss_ch+(torch.tensor([l0,l1,l2]))
    loss.backward(retain_graph=True)
    return loss

# Initialize Neural Network

if EBC[0]=="Weighted":
    Loss = Loss_w
else:
    Loss = Loss_d

def Loss_plain():
    optimizer.zero_grad()
    loss = torch.tensor([0.0],dtype = Dtype,requires_grad=True)
    global loss_ch
    loss_ch=torch.tensor([0.0,0.0,0.0],dtype=Dtype,requires_grad=True)
    for i in range(X_d.shape[0]):
        xi = X_d[i][0]
        ui = Nn.forward(xi.reshape(-1,1))
        ux  = torch.autograd.grad(ui,xi,grad_outputs=torch.ones_like(ui),retain_graph=True,create_graph=True)[0]
        uxx = torch.autograd.grad(ux,xi,grad_outputs=torch.ones_like(ux),retain_graph=True,create_graph=True)[0]
        P2_t = P2(xi)
        P1_t = P1(xi)
        P0_t = P0(xi)
        Q_t = Q(xi)
        l0 = MSE(P2_t*uxx + P1_t*ux + P0_t*ui + Q_t)
        l1 = MSE(Nn.forward(torch.tensor([a],requires_grad=True,dtype=Dtype).reshape(-1,1))-alpha)
        xb = torch.tensor([b],requires_grad=True,dtype=Dtype)
        temp = Nn.forward(xb)
        temp2 = torch.autograd.grad(temp,xb,grad_outputs=torch.ones_like(temp),retain_graph=True,create_graph=True)[0]
        l2 = MSE(temp2-beta)

        loss = loss + l0+l1+l2
        loss_ch = loss_ch+(torch.tensor([l0,l1,l2]))/X_d.shape[0]

    loss.backward(retain_graph=True)
    return loss

Nn.train()
loss_log = []
test_log = []
loss_ch=torch.tensor([0.0,0.0,0.0],dtype=Dtype,requires_grad=True)

#   Training with EBC
los = Loss()
loss_log.append([0,los.item(),loss_ch[0].item(),loss_ch[1].item(),loss_ch[2].item()])
sys.stdout.write('\nEpoch : %d\tLoss: %3.2e\tL0: %3.2e\tL1: %3.2e\tL2: %3.2e'%(0,los,loss_ch[0],loss_ch[1],loss_ch[2]))
time_1 = time.time()
for epoc in range(N_epoch_EBC):
    los = optimizer.step(Loss)
    if epoc%N_write==0:
        sys.stdout.write('\nEpoch : %d\tLoss: %3.2e\tL0: %3.2e\tL1: %3.2e\tL2: %3.2e'%(epoc,los,loss_ch[0],loss_ch[1],loss_ch[2]))
        loss_log.append([epoc,los.item(),loss_ch[0].item(),loss_ch[1].item(),loss_ch[2].item()])
        #Test_Loss_Eval(Nn)
time_2 = time.time()

#   Training without EBC

for epoc in range(N_epoch_EBC,N_epoch_EBC+N_epoch_Plain):
    los = optimizer.step(Loss_plain)
    if epoc%N_write==0:
        sys.stdout.write('\nEpoch : %d\tLoss: %3.2e\tL0: %3.2e\tL1: %3.2e\tL2: %3.2e'%(epoc,los,loss_ch[0],loss_ch[1],loss_ch[2]))
        loss_log.append([epoc,los.item(),loss_ch[0].item(),loss_ch[1].item(),loss_ch[2].item()])
        #test_log.append(np.hstack([epoc,Test_Loss_Eval(Nn)]))
time_3 = time.time()

print("\nSuccessfully Executed")
torch.save(Nn, "Output/"+head_string+" Trained_Model.pt")
if N_epoch_EBC != 0:
    print("Average time consumed per epoch for \nTraining with EBC : %3.2f\n"%((time_2-time_1)/N_epoch_EBC))
if N_epoch_Plain != 0:
    print("Training w/o EBC : %3.2f\n"%((time_3-time_2)/N_epoch_Plain))
xa = torch.tensor([a],requires_grad=True,dtype=Dtype)
temp = Nn.forward(xa)
print("u(a) = %3.4f"%(temp.item()))
del xa,temp
xb = torch.tensor([b],requires_grad=True,dtype=Dtype)
temp = Nn.forward(xb)
temp2 = torch.autograd.grad(temp,xb,grad_outputs=torch.ones_like(temp),retain_graph=True,create_graph=True)[0]
print("u'(b) = %3.4f"%(temp2.item()))
del xb,temp,temp2

########################################################################################
#                                   Visualisation

#   Plotting Residual Plot
loss_arr = np.array(loss_log)
fig = plt.figure(figsize=(8,6),dpi=500)
ax = fig.add_subplot(111)
ax.semilogy(loss_arr[:,0],loss_arr[:,1],color='tab:blue', linewidth = 2)
ax.set_xlabel('Epochs')
ax.set_ylabel('Loss')
ax.set_title(head_string+" - Residual", fontsize = 15)

for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(15)

plt.savefig("Output/"+head_string+" Residual.pdf",bbox_inches='tight', pad_inches=0.5)
#plt.savefig("Output/"+head_string+" Residual.eps",bbox_inches='tight', pad_inches=0.5)
plt.show()
np.savetxt("Output/Residual.csv",loss_arr,delimiter=",")
#   Plot Predicted Solution against Exact Solution
x_arr = torch.tensor(np.linspace(a,b,101),dtype=Dtype).reshape(-1,1)
u_pred = Nn.forward(x_arr)
u_exac = Exact(x_arr)
fig = plt.figure(figsize=(8,6),dpi=500)
ax = fig.add_subplot(111)
ax.plot(x_arr.clone().detach().numpy(),u_pred.clone().detach().numpy(),label='Predicted',color='tab:blue', linewidth = 2)
ax.plot(x_arr.clone().detach().numpy(),u_exac.clone().detach().numpy(),'r--',label='Exact', linewidth = 2)
ax.set_xlabel('$x$')
ax.set_ylabel('$u$')
ax.set_xlim([a,b])
ax.set_title(head_string+" - Solution Comparison", fontsize = 15)
ax.legend()

for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(15)
np.savetxt("Output/"+head_string+" Solution.csv",np.hstack([x_arr.clone().detach().numpy(),u_pred.clone().detach().numpy()]),delimiter=",")

plt.savefig("Output/"+head_string+" Solution.pdf",bbox_inches='tight', pad_inches=0.5)
#plt.savefig("Output/"+head_string+" Solution.eps",bbox_inches='tight', pad_inches=0.5)
plt.show()
########################################################################################
#                                       Body
########################################################################################
#                                      Output
########################################################################################
#                        AUTHOR: VEERENDRA HARSHAL BUDHI
########################################################################################
