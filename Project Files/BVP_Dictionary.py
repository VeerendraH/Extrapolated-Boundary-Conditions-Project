import torch
def P2(x):
	return 2*torch.ones_like(x)
def P1(x):
	return 0
def P0(x):
	return 1*torch.ones_like(x)
	
a = 0
ia = 0
b = 1
ib = 1
alpha = 0
beta = 0
Num_Node_int = 50
Num_Node_bca = 25
Num_Node_bcb = 25