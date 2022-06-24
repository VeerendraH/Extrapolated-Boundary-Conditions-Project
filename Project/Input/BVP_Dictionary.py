import torch
import numpy as np
def A(x):
	return 1.9 + 0.2*x
def P2(x):
	return A(x)#2*torch.ones_like(x)
def P1(x):
	return 0.2*torch.ones_like(x)
def P0(x):
	return 0*torch.ones_like(x)
def Q(x):
	return x
def Exact(x):
	r = 2.1/1.9
	t1 = -1/(4*1.9*(r-1)**3)
	t2 = (r-1)*x*(-2 + (r-1)*x)
	t3 = 2*(2*(r-1)**2 + r*(r-2))*np.log(1+x*(r-1))
	return t1*(t2-t3)
bvp_string = "  Ed/dx(A(x)du/dx) + q(x) = 0 for x in [0,1]\n  u(0) = 0.0000\n  u'(1) = 0.4762"
head_string = "2nd Order BVP"
a = 0
ia = 0
b = 1
ib = 1
alpha = 0
beta = 1/(2.1)
Num_Node_int = 50
Num_Node_bca = 25
Num_Node_bcb = 25