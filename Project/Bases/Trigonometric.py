import numpy as np
def EBC_r_j(xi,j,x_bc,n):
	x = xi - x_bc
	if n%2==0:
		n=n+1
	A = np.zeros((n,n))
	m = int((n)/2)
	A[:,0] = 1
	for i in range(m):
	    A[:,1+i] = np.sin((i+1)*x);
	    A[:,m+1+i] = np.cos((i+1)*x);
	if j == 0:
		D = np.zeros((1,n))	
		D[0,0] = 1
		for i in range(m):
		    D[0,1+i] = np.sin((i+1)*0);
		    D[0,m+1+i] = np.cos((i+1)*0);
		return D@np.linalg.inv(A)
	elif j == 1:
		D = np.zeros((1,n))	
		D[0,0] = 0
		for i in range(m):
		    D[0,1+i] =  (i+1)*np.cos((i+1)*0);
		    D[0,m+1+i] = -(i+1)*np.sin((i+1)*0);
		return D@np.linalg.inv(A)

def R(X,x,o,EBC_params):
    a = EBC_params[0]
    b = EBC_params[1]
    n = EBC_params[2]
    if o>n:
        print("Error! = Order of BC must be less than Order of EBC")
        return None
    if x == "a":
        return EBC_r_j(X,o,a,n)
    elif x == "b":
        return EBC_r_j(X,o,b,n)
    else:
        print("Error! = Given boundary not at boundary")
        return None

def R_arr(X_d,x,o,EBC_params):
    R_arr = np.zeros((X_d.shape[0],EBC_params[2]+1))
    for i in range(X_d.shape[0]):
        R_arr[i,:] = R(X_d[i,:],x,o,EBC_params)
    return R_arr