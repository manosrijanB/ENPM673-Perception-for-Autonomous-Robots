import numpy as np

xi, yi = [5, 150, 150, 5], [5, 5, 150, 150]

xpi, ypi  = [100,200,220,100], [100,80,80,200]


rows = 8
cols = 9

A = []
for k in range(rows//2):
    A += [[-xi[k], -yi[k], -1, 0, 0, 0, xi[k]*xpi[k], yi[k]*xpi[k], xpi[k]],
          [0, 0, 0, -xi[k], -yi[k], -1, xi[k]*ypi[k], yi[k]*ypi[k], ypi[k]]]
A = np.array(A)

# print(A.shape)

# Computing SVD

def SVD(X):
    rows, cols = A.shape
    
    AAT = X @ (X.T)
    print(AAT)
    ATA = (X.T) @ X
    # print (U)
    #  eigen vectors
    e1, U = np.linalg.eig(AAT)
    e2, V = np.linalg.eig(ATA)
    index_1 = np.flip(np.argsort(e1))
    e1 = e1[index_1]
    U = U[:, index_1]
    index_2 = np.flip(np.argsort(e2))
    e2 = e2[index_2]
    V = V[:, index_2]
    i,k = X.shape
    E = np.zeros((rows,cols))

    for p in range(min(rows,cols)):
        E[p,p] = np.abs(np.sqrt(e1[p]))
    print("U:", U)
    print("V:", V)
    print("E:", E)
    return U, E, V

SVD(A)

# Computing Homography
def Homography(X):
    U, E, V = SVD(X)
    H_matrix = V[:, -1]
    H_matrix = H_matrix.reshape(3,3)
    H_matrix_normalized = H_matrix/H_matrix[2,2]
    print ("Homography Matrix:", H_matrix_normalized)
    return H_matrix_normalized

Homography(A)






# def SVD(X):
#     # rows, cols = A.shape
#     AAT = X @ (X.T)
#     ATA = (X.T) @ X
#     # print (U)
#     e1, U = np.linalg.eig(AAT)
#     e2, V = np.linalg.eig(ATA)
#     index_1 = np.flip(np.argsort(e1))
#     e1 = e1[index_1]
#     U = U[:, index_1]
#     index_2 = np.flip(np.argsort(e2))
#     e2 = e2[index_2]
#     V = V[:, index_2]
#     rows, cols = X.shape
#     E = np.zeros((rows,cols))

#     for p in range(min(rows,cols)):
#         E[p,p] = np.abs(np.sqrt(e1[p]))
#     print("U:", U)
#     print("V:", V)
#     print("E:", E)
#     return U, E, V

# SVD(A)

# # Computing Homography
# def Homography(X):
#     U, E, V = SVD(X)
#     H_matrix = V[:, -1]
#     H_matrix = H_matrix.reshape(3,3)
#     H_matrix_normalized = H_matrix/H_matrix[2,2]
#     print ("Homography Matrix:", H_matrix_normalized)
#     return H_matrix_normalized
# Homography(A)