from sklearn.linear_model import Ridge
import numpy as np

import matplotlib.pyplot as plt

#########################################################################
# input
#########################################################################

#input points to be fit to
x = [ 1, 4, 5, 6]
y = [ 2, 2, 5, 7]

#polynomial degree of function to be fit
degree = 1


#########################################################################
# main
#########################################################################

X = np.array([np.power(x,deg) for deg in range(degree,-1,-1)]).transpose()

clf = Ridge(alpha=0, fit_intercept=False)
clf.fit(X, y)

x_test = np.linspace(min(x), max(x), 20)
X_test = np.array([np.power(x_test,deg) for deg in range(degree,-1,-1)]).transpose()
y_test = np.dot(X_test, np.reshape(clf.coef_,(-1,1)))

print("the following coefficients were found:", clf.coef_)

plt.scatter(x,y)
plt.plot(x_test,y_test)
plt.show()