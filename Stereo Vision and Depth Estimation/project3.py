import cv2 as cv
from cv2 import imshow
from cv2 import drawMatches
import numpy as np
import glob
import matplotlib.pyplot as plt
from scipy import linalg
import time
from tqdm import tqdm
from functions import *
from fundamentalmat import *
from disparity import *



def rescale(image, scale):

    width = int(image.shape[1] * scale)
    height = int(image.shape[0] * scale)
    dim = (width, height)
    resized = cv.resize(image, dim, interpolation = cv.INTER_AREA)
    return resized

# give input here
print("Enter the Data which to be used curule(1),octagon(2) or pendulum(3): ")
dataset = int(input())

if dataset <= 0 or dataset >=4:
    print("Invalid Dataset Number")
    exit(0)

if dataset == 1:

    K_left = np.array([[1758.23, 0, 977.42], [0, 1758.23, 552.15], [0, 0, 1]])
    K_right = np.array([[1758.23, 0, 977.42], [0, 1758.23, 552.15], [0, 0, 1]])
    Baseline,Focallength,width,height,ndisp,vmin,vmax=88.39,1758.23,1920,1080,220,55,195
    folder_name = r"/home/chaosmachete/Documents/project1 667/project 3/data/curule/*.png"

elif dataset == 2:
    K_left = np.array([[1742.11, 0, 804.90], [0, 1742.11, 541.22], [0, 0, 1]])
    K_right = np.array([[1742.11, 0, 804.90], [0, 1742.11, 541.22], [0, 0, 1]])
    Baseline,Focallength,width,height,ndisp,vmin,vmax =221.76,1742.11,1920,1080,100,29,61
    folder_name = r"/home/chaosmachete/Documents/project1 667/project 3/data/octagon/*.png"

elif dataset == 3:
    K_left = np.array([[1729.05, 0, -364.24], [0, 1729.05, 552.22], [0, 0, 1]])
    K_right = np.array([[1729.05, 0 ,-364.24], [0, 1729.05, 552.22], [0, 0, 1]])
    Baseline,Focallength,width,height,ndisp,vmin,vmax =174.019,1729.05,1920,1080,180,25,150
    folder_name = r"/home/chaosmachete/Documents/project1 667/project 3/data/pendulum/*.png"
        
    

images = []

path = glob.glob(folder_name)
for file in path:
    image = cv.imread(file)
    images.append(image)

# using sift to detect features
sift = cv.SIFT_create()

image_Left = images[0].copy()
image_Right = images[1].copy()

image_Left = rescale(image_Left,0.6)
image_Right = rescale(image_Right,0.6)

h1, w1 = image_Left.shape[:2]
h2, w2 = image_Right.shape[:2]

image_Left_gray = cv.cvtColor(image_Left, cv.COLOR_BGR2GRAY) 
image_Right_gray = cv.cvtColor(image_Right, cv.COLOR_BGR2GRAY)

kp1, des1 = sift.detectAndCompute(image_Left_gray, None)
kp2, des2 = sift.detectAndCompute(image_Right_gray, None)

#  using brute force matcher
bf = cv.BFMatcher()
matches = bf.match(des1,des2)
matches = sorted(matches, key = lambda x:x.distance)
matched_image = cv.drawMatches(image_Left,kp1,image_Right,kp2,matches[:100],None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plt.imshow(matched_image)
plt.show()
chosen_matches = matches[:100]
matched_pts_left = np.array([kp1[m.queryIdx].pt for m in chosen_matches]).reshape(-1, 2)
matched_pts_right = np.array([kp2[m.trainIdx].pt for m in chosen_matches]).reshape(-1, 2)
matched_pts = (matched_pts_left,matched_pts_right)

                                       # calculating fundamental matrix
F_M = FundamentalMatrix()
ransac_model = Ransac(F_M)
FM, inlier_mask = ransac_model.fit(matched_pts)

matched_pts_left_chosen = matched_pts_left[np.where(inlier_mask.ravel()==1)]
matched_pts_right_chosen = matched_pts_right[np.where(inlier_mask.ravel()==1)]
# defing ess
def EssentialMatrixFromFundamentalMatrix(K_left,K_right, FM):
    E = (K_right.T).dot(FM).dot(K_left)
    U,D,VT = np.linalg.svd(E)
    D = [1,1,0]
    EM = np.dot(U,np.dot(np.diag(D),VT))
    return EM
EM = EssentialMatrixFromFundamentalMatrix(K_left,K_right, FM)

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

R, C = ExtractCameraPose(EM)

Pts_3D = []
R1  = np.identity(3)
C1  = np.zeros((3, 1))
I = np.identity(3)


def Triangulation(PL, PR, ptsL, ptsR):
 
    A = [ptsL[1]*PL[2,:] - PL[1,:], PL[0,:] - ptsL[0]*PR[2,:], ptsR[1]*PR[2,:] - PR[1,:], PR[0,:] - ptsR[0]*PR[2,:]]

    A = np.array(A).reshape((4,4))
 
    A = np.dot(A.T, A)
    
    U, D, VT = linalg.svd(A, full_matrices = False)
 
    return VT[3,0:3]/VT[3,3]


for i in range(len(R)):
    R2 =  R[i]
    C2 =   C[i].reshape(3,1)
    ProjectionM_left = np.dot(K_left, np.dot(R1, np.hstack((I, -C1.reshape(3,1)))))
    ProjectionM_right = np.dot(K_right, np.dot(R2, np.hstack((I, -C2.reshape(3,1)))))

    for xL,xR in zip(matched_pts_left_chosen, matched_pts_right_chosen):
        pts_3d = Triangulation(ProjectionM_left, ProjectionM_right, np.float32(xL), np.float32(xR) )
        pts_3d = np.array(pts_3d)
        Pts_3D.append(pts_3d)

def Cheirality_Condition(P_3D,C_,R_3):
    num_positive = 0
    for P in P_3D:
        P = P.reshape(-1,1)
        if R_3.dot(P - C_) > 0 and P[2]>0:
            num_positive+=1
    return num_positive

best_i = 0
max_Positive = 0

for i in range(len(R)):
    R_, C_ = R[i],  C[i].reshape(-1,1)
    R_3 = R_[2].reshape(1,-1)
    num_Positive = Cheirality_Condition(Pts_3D,C_,R_3)

    if num_Positive > max_Positive:
        best_i = i
        max_Positive = num_Positive

R_Config, C_Config, P3D = R[best_i], C[best_i], Pts_3D[best_i]

print(" R of camera Pose",R_Config)
print("C of camera Pose", C_Config)


def drawlines(img1,img2,lines,pts1,pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r,c = img1.shape
    img1 = cv.cvtColor(img1,cv.COLOR_GRAY2BGR)
    img2 = cv.cvtColor(img2,cv.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv.circle(img2,tuple(pt2),5,color,-1)
    return img1,img2




matched_pts_left_chosen = np.float64(matched_pts_left)
matched_pts_right_chosen = np.float64(matched_pts_right)



_, H1, H2 = cv.stereoRectifyUncalibrated(np.float32(matched_pts_left_chosen), np.float32(matched_pts_right_chosen), FM, imgSize=(w1, h1))

print(" H1 ", H1)
print(" H2 ", H2)

pts1 = np.zeros((matched_pts_left_chosen.shape), dtype=int)
pts2 = np.zeros((matched_pts_right_chosen.shape), dtype=int)
# doing rectification
image_Left_rectified = cv.warpPerspective(image_Left, H1, (w1, h1))
image_Right_rectified = cv.warpPerspective(image_Right, H2, (w2, h2))

matched_pts_left_chosen_rectified = cv.perspectiveTransform(matched_pts_left_chosen.reshape(-1, 1, 2), H1).reshape(-1,2)
matched_pts_right_chosen_rectified = cv.perspectiveTransform(matched_pts_right_chosen.reshape(-1, 1, 2), H2).reshape(-1,2)
H2_T_inv =  np.linalg.inv(H2.T)
H1_inv = np.linalg.inv(H1)
FM_rectified = np.dot(H2_T_inv, np.dot(FM, H1_inv))

# using inbulit function for calculating epilines
linesL_rectified = cv.computeCorrespondEpilines(matched_pts_left_chosen_rectified.reshape(-1,1,2), 2,FM_rectified)
lines1 = linesL_rectified.reshape(-1,3)

linesR_rectified = cv.computeCorrespondEpilines(matched_pts_right_chosen_rectified.reshape(-1, 1, 2),2, FM_rectified)
linesR_rectified   = linesR_rectified[:,0]



image_Left_rectified_resized = rescale(image_Left_rectified,0.6)
image_Right_rectified_resized = rescale(image_Right_rectified,0.6)

imgL_Rectified_gray = cv.cvtColor(image_Left_rectified_resized,cv.COLOR_BGR2GRAY)
imgR_Rectified_gray = cv.cvtColor(image_Right_rectified_resized,cv.COLOR_BGR2GRAY)

eight,width,_ = image_Right_rectified_resized.shape


 # calling dispairty function to calculate disparity
disparity_map_unscaled, disparity_map_scaled = ssd_correspondence(imgL_Rectified_gray, imgR_Rectified_gray)

plt.figure(1)
plt.title('Disparity Map Graysacle')
plt.imshow(disparity_map_scaled, cmap='gray')
plt.figure(2)
plt.title('Disparity Map Hot')
plt.imshow(disparity_map_scaled, cmap='hot')

 # calling depth function to calculate depth
    
depth_map, depth_array = disparitydepth(Baseline, Focallength, disparity_map_unscaled)


    
plt.figure(3)
plt.title('Depth Map Graysacle')
plt.imshow(depth_map, cmap='gray')
plt.figure(4)
plt.title('Depth Map Hot')
plt.imshow(depth_map, cmap='hot')
plt.show()
