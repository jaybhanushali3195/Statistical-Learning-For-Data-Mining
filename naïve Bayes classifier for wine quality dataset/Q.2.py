
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
#dataset = pd.read_csv('D:\Datasets\test.csv')  

N=100 # Number of rows
M=50 # Number of predictors
x0 = 5
#seed=17
hats=[]
names=[]
names.append(M)
np.random.seed()

#generate new data and refit the model for multiple replicates
for j in range(0, 100, 1): 
    X = np.random.random_integers(0, 10+1,(N,M)) # generate N rows and M columns of random normals
    print(X.shape)
    e = np.random.normal(0, 1,N) # generate N rows of random errors
    # true model y = 5 - 2x0 + X^2 + e
    print(e.shape)
    y= 5-2*X[:,0]+ X**2[:] + e[:]# generate the simulated data
    print (y.shape)
    #poly1d([1,-2,5])
    modelnow=linear_model.LinearRegression()
    modelnow.fit(X,y) # fit the model
    yhat = modelnow.intercept_+modelnow.coef_[0]*x0 #calculate predicted value at x0
    hats.append(yhat)

# boxplot of predictions at x0
fig = plt.figure()
fig.suptitle('Box plot')
ax = fig.add_subplot(111)
plt.boxplot(hats)
ax.set_xticklabels(names)
plt.show()