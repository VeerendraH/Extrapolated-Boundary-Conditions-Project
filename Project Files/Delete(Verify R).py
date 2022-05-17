'''
fig, (ax1, ax2) = plt.subplots(1, 2,figsize=[12,5],dpi=400)

def f(x):
    return 2*x+3
n=Order_ebc
testlist1 = []
checklist1 = []

x_test1 = np.random.randint(0,25,10)
for i in range(10):
    x_d_i = X_d[x_test1[i],:]-a
    A = np.vstack([x_d_i**i for i in range(n+1)]).T
    A1 = A.copy()
    A1[:,o1] = f(x_d_i+a)
    testlist1.append(np.linalg.det(A1)/np.linalg.det(A))
    checklist1.append(np.dot(R_arr_0[x_test1[i],:],f(x_d_i+a)))

ax1.plot(testlist1)
ax1.set_title("Extrapolated y(0) value of 2x+3 from near x = 0 dataset")
ax1.plot(np.array(checklist1),"--r")
ax1.legend(["Exact","Predict"])

ax1.set_ylim([1,4])

testlist2 = []
checklist2 = []


x_test2 = np.random.randint(75,100,10)
for i in range(10):
    x_d_i = X_d[x_test2[i],:]-b
    A = np.vstack([x_d_i**i for i in range(n+1)]).T
    A1 = A.copy()
    A1[:,o2] = f(x_d_i+b)
    testlist2.append(np.linalg.det(A1)/np.linalg.det(A))
    checklist2.append(np.dot(R_arr_1[x_test2[i],:],f(x_d_i+b)))
ax2.plot(testlist2)
ax2.plot(np.array(checklist2),"--r")
ax2.set_title("Extrapolated y(1) value of 2x+3 from near x = 1 dataset")
ax2.legend(["Exact","Predict"])

ax2.set_ylim([1,6])

fig.suptitle("Verification of Extrapolated Boundary Conditions predicting Boundary Solution",fontsize=18)

plt.show()
'''