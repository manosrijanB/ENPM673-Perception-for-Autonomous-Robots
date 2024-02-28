
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as Eig 


data =  np.genfromtxt("/home/chaosmachete/Downloads/data.csv", delimiter = ',' , skip_header = 1)
# print(data)
 
a = age = data[:, 0]
c = cost = data[: , -1]
X = np.vstack((a , c)).T
n= len(a)
plt.scatter(age,cost)


# finding covariance 
def covar(a, c):
    abar, cbar = a.mean(), c.mean()
    return np.sum((a - abar)*(c - cbar))/(n - 1)

# covariance matrix 
def covarmat(X):
    return np.array ([[covar(X[0], X[0]), covar(X[0], X[1])], \
                     [covar(X[1], X[0]), covar(X[1], X[1])]])
                     
# calculating covariance matrix for 
sigma = covarmat(X.T) 
# print(sigma)
# eigen vectors for covariance matrix
eig_cv = Eig.eig(sigma)
eigen_values = eig_cv[0]
print(eigen_values)
eigen_vectors = eig_cv[1]
origin = (a.mean(), c.mean())
print(eigen_vectors)
# plt.show()
eig_vec1 = eigen_vectors[:, 0]
eig_vec2 = eigen_vectors[:, 1]

# This line below plots the 2d points
# plt.scatter(np_array[:,0], np_array[:,1])

plt.quiver(*origin, *eig_vec1, color=['r'], scale=10)   # using * to unpack the tuple as we defined orgin as a tuple
plt.quiver(*origin, *eig_vec2, color=['b'], scale=14)
# plt.show()

#  linear least square 
E = np.vstack([a, np.ones(len(a))]).T 
m = np.linalg.inv(E.T@ E) @E.T @ c 
 
newy = E @ m
plt.plot(a, newy, 'k')
# # plt.show()

def TLS(a,c):

    # a = data[:,0]
    # c = data[:,-1]
    a_mean = np.mean(a)
    c_mean = np.mean(c)
    
    U = np.vstack(((a - a_mean), (c - c_mean))).T
    #print("U size = ", U.shape)

    A = np.dot(U.transpose(), U)
    #print("A size = ", A.shape)

    B = np.dot(A.transpose(), A)    

    w, v = np.linalg.eig(B)
    index = np.argmin(w)  
    coef = v[:, index]
    x, y = coef
    d =  x * a_mean + y * c_mean
    # coef = np.array([x, y, d])   
    # a_min = np.min(a)
    # a_max = np.max(a)
    # a_curve = np.linspace(a_min-100, a_max+100, 300)     
    # c_curve = (d - x*a)/y
    # c_curve = y

    # plt.scatter(a,c)
    # plt.plot(a, c_curve, 'r')
    # plt.show()
    # plt.savefig(name)

    return x, y, d 

TLS(a,c)


def ransac(a, c, outliers = 0.25, probability = 0.95, threshold = 100):
    # a = data[:,0]
    # c = data[:,-1] 
    
    iter = 0
    
    iter_max = np.log(1 - probability)/np.log(1-(1- outliers)**2)
    
    best_inliers = 0
    best_coeffs = 0

    
    while iter < iter_max:

        i, j = np.random.randint(n, size = 2)
        p1 = np.array([a[i], a[j]])
        p2 = np.array([c[i], c[j]])

        x, y, d = TLS(p1, p2)
        er = (a*x + c*y - d)**2
        
        inliers = 0

        for k in range(n):
            if float(er[k]) < threshold:
               inliers += 1

        if inliers> best_inliers:
            best_inliers = inliers
            best_coeffs = x, y, d
            
        iter += 1
    return best_coeffs

x,y, d = TLS(a, c)
c_curve = (d - a*x)/y
plt.plot( a, c_curve, '-g')

x, y, d = ransac(a, c)
c_curve = (d - a*x)/y
plt.plot( a, c_curve, '-r')
plt.show()