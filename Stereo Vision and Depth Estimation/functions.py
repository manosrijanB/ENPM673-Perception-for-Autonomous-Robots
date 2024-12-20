from fundamentalmat import *
from scipy import linalg





def EssentialMatrixFromFundamentalMatrix(K_left,K_right, FM):
    E = (K_right.T).dot(FM).dot(K_left)
    U,D,VT = np.linalg.svd(E)
    D = [1,1,0]
    EM = np.dot(U,np.dot(np.diag(D),VT))
    return EM


def ExtractCameraPose(E):
    U, D, VT = np.linalg.svd(E)
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])

    R = []
    C = []
    C.append(U[:, 2])
    R.append(np.dot(U, np.dot(W, VT)))
    C.append(-U[:, 2])
    R.append(np.dot(U, np.dot(W, VT)))
    C.append(U[:, 2])
    R.append(np.dot(U, np.dot(W.T, VT)))
    C.append(-U[:, 2])
    R.append(np.dot(U, np.dot(W.T, VT)))

    for i in range(4):
        if (np.linalg.det(R[i]) < 0):
            R[i] = -R[i]
            C[i] = -C[i]

    return R, C


def Triangulation(PL, PR, ptsL, ptsR):
 
    A = [ptsL[1]*PL[2,:] - PL[1,:], PL[0,:] - ptsL[0]*PR[2,:], ptsR[1]*PR[2,:] - PR[1,:], PR[0,:] - ptsR[0]*PR[2,:]]

    A = np.array(A).reshape((4,4))
 
    A = np.dot(A.T, A)
    
    U, D, VT = linalg.svd(A, full_matrices = False)
 
    return VT[3,0:3]/VT[3,3]


def Cheirality_Condition(P_3D,C_,R_3):
    num_positive = 0
    for P in P_3D:
        P = P.reshape(-1,1)
        if R_3.dot(P - C_) > 0 and P[2]>0:
            num_positive+=1
    return num_positive