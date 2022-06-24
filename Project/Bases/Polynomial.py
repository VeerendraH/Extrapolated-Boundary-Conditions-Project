import numpy as np
def EBC_r_j(xi,j,x_bc,n):
	x = xi - x_bc
	A = np.zeros((n+1,n+1))
	m = int((n)/2)
	for i in range(n+1):
	    A[:,i] = x**i;
	if j == 0:
		D = np.zeros((1,n+1))
		for i in range(n):
		    D[0,i] = 0**i;
		return D@np.linalg.inv(A)
	elif j == 1:
		D = np.zeros((1,n+1))
		D[0,0] = 0
		for i in range(1,n+1):
			D[0,i] = i*(0**(i-1))
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