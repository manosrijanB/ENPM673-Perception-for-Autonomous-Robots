import cv2 as cv2
import numpy as np

# given world points 
X_W = [0,0,0,0,7,0,7,0]
Y_W = [0,3,7,11,1,11,9,1]
Z_W = [0,0,0,0,0,7,0,7]
# given image points
X_I = [757,758,758,759,1190,329,1204,340]
Y_I = [213,415,686,966,172,1041,850,159]

M = []
for i in range(8):
    r1 = np.array([X_W[i], Y_W[i], Z_W[i], 1,0, 0, 0, 0,-X_W[i]*X_I[i], -Y_W[i]*X_I[i], -X_I[i]*Z_W[i],-X_I[i]])
    M.append(r1)
    r2 = np.array([0, 0, 0, 0, X_W[i], Y_W[i], Z_W[i], 1, -X_W[i]*Y_I[i], -Y_W[i]*Y_I[i], -Y_I[i]*Z_W[i],-Y_I[i]])
    M.append(r2)


M = np.array(M)

U, E, VT = np.linalg.svd(M)
V = VT.transpose()
A_vertical = V[:, V.shape[1] - 1]
A = A_vertical.reshape([3,4])

P = A[0:3,0:3]
R,K = np.linalg.qr(P)
print(K)
