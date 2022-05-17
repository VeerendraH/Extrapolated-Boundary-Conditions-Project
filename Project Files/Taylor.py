import numpy as np

def EBC_r_0(xi,i,a,n):
    #xi = X_d[i:i+1,:]
    dx = xi.reshape(n+1,1) - a
    A = np.hstack([dx**i for i in range(n+1)])
    A = np.delete(A,i,1)
    r = np.ones(n+1,)
    for j in range(n+1):
        r[j] = ((-1)**(i+j)) * np.linalg.det(np.delete(A.copy(),i,0))
    return r

def EBC_r_1(xi,i,b,n):
    #xi = X_d[i:i+1,:]
    dx = xi.reshape(n+1,1) - b
    A = np.hstack([dx**i for i in range(n+1)])
    A = np.delete(A,i,1)
    r = np.ones(n+1,)
    for j in range(n+1):
        r[j] = ((-1)**(i+j)) * np.linalg.det(np.delete(A.copy(),i,0))
    return r
def EBC_r_j(xi,j,x_bc,n):
    #xi = X_d[i:i+1,:]
    dx = xi.reshape(-1,1) - x_bc
    A = np.hstack([dx**i for i in range(n+1)])
    A = np.delete(A,j,1)
    r = np.ones(n+1,)
    for i in range(n+1):
        r[i] = ((-1)**(i+j)) * np.linalg.det(np.delete(A.copy(),i,0))
    return r
def EBC_del(xi,x_bc,n):
    #xi = X_d[i,:]
    dx = xi.reshape(-1,1) - x_bc
    A = np.hstack([dx**i for i in range(n+1)]).T
    return np.linalg.det(A)

def R(X,x,o,EBC_params):

    a = EBC_params[0]
    b = EBC_params[1]
    n = EBC_params[2]
    if o>n:
        print("Error! = Order of BC must be less than Order of EBC")
        return None
    if x == "a":
        return EBC_r_j(X,o,a,n)/EBC_del(X,a,n)
    elif x == "b":
        return EBC_r_j(X,o,b,n)/EBC_del(X,b,n)
    else:
        print("Error! = Given boundary not at boundary")
        return None

def R_arr(X_d,x,o,EBC_params):
    R_arr = np.zeros((X_d.shape[0],EBC_params[2]+1))
    for i in range(X_d.shape[0]):
        R_arr[i,:] = R(X_d[i,:],x,o,EBC_params)
    return R_arr