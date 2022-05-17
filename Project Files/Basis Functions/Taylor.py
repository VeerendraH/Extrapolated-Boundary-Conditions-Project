import numpy as np
def EBC_r_0(X_d,i,a,n):
    xi = X_d[i:i+1,:]
    dx = xi.T - a
    A = np.hstack([dx**i for i in range(n+1)])
    A = np.delete(A,0,1)
    r = np.ones(n+1,)
    for i in range(n+1):
        r[i] = ((-1)**(i)) * np.linalg.det(np.delete(A.copy(),i,0))
    return r

def EBC_r_1(X_d,i,b,n):
    xi = X_d[i:i+1,:]
    dx = xi.T - b
    A = np.hstack([dx**i for i in range(n+1)])
    A = np.delete(A,0,1)
    r = np.ones(n+1,)
    for i in range(n+1):
        r[i] = ((-1)**(i)) * np.linalg.det(np.delete(A.copy(),i,0))
    return r

def EBC_del(X_d,i,j,a,b,n):
    xi = X_d[i,:]
    dx = xi.T - [a,b][j]
    A = np.vstack([dx**i for i in range(n+1)]).T
    return np.linalg.det(A)

def R(X,x,o,EBC_params):

	a = EBC_params[0]
	b = EBC_params[1]
	n = EBC_params[2]
	if o>n:
		print("Error! = Order of BC must be less than Order of EBC")
		return None
	if x == "a":
		return EBC_r_0(X,o,a,n)/EBC_del(X,o,0,a,b,n)
	elif x == "b":
		return EBC_r_1(X,o,b,n)/EBC_del(X,o,0,a,b,n)
	else:
		print("Error! = Given boundary not at boundary")
		return None
R_arr_0 = np.zeros((N,n+1))
for i in range(N):
    R_arr_0[i,:] = EBC_r_0(X_d,i)/EBC_del(X_d,i,0)

R_arr_1 = np.zeros((N,n+1))
for i in range(N):
    R_arr_1[i,:] = EBC_r_1(X_d,i)/EBC_del(X_d,i,1) 
def R_tensor(X_d,x,o,EBC_params):
	for i in range(X_d):
		R_arr[i,:] = R(X_d[i,:],x,o,EBC_params)
	return R_arr