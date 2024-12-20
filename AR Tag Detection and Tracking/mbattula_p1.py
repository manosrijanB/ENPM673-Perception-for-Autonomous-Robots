import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import math

vid = cv.VideoCapture("/home/chaosmachete/Desktop/dummy/output.avi")
# vid is a variable

testudo = cv.imread("/home/chaosmachete/Desktop/dummy/testudo.png") 
# resizing for the cude 
testudo = cv.resize(testudo,(160, 160))
#   for saving the video  
fourcc = cv.VideoWriter_fourcc(*'XVID')
out = cv.VideoWriter('3D projection.avi', fourcc, 30.0, (1280, 720))
#   here 1280 and 720 are resolution of video 
while True:
    isTrue, frame = vid.read()
#  converting to grayscale to convert into 2 channel
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # using blur to reduce the noise
    blur = cv.GaussianBlur(gray,(27,27),cv.BORDER_DEFAULT)
    cv.imshow("blur",blur)
    # inbulit threshloing 
    retval,threshed = cv.threshold(blur,140,200,cv.THRESH_BINARY)
   
    # to show threshloing 
    cv.imshow('threshed',threshed)
   
    
    # creating a projection matrix
    def projectionMatrix(h, K):  
        h1 = h[:,0]          #taking column vectors h1,h2 and h3
        h2 = h[:,1]
        h3 = h[:,2]

        #calculating lamda
        lamda = 2 / (np.linalg.norm(np.matmul(np.linalg.inv(K),h1)) + np.linalg.norm(np.matmul(np.linalg.inv(K),h2)))
        b_t = lamda * np.matmul(np.linalg.inv(K),h)

        #check if determinant is greater than 0 ie. has a positive determinant when object is in front of camera
        det = np.linalg.det(b_t)

        if det > 0:
            b = b_t
        else:
      #else make it positive
            b = -1 * b_t  
            
        row1 = b[:, 0]
        row2 = b[:, 1]      
        #extract rotation and translation vectors
        row3 = np.cross(row1, row2)
        
        t = b[:, 2]
        Rt = np.column_stack((row1, row2, row3, t))

        P = np.matmul(K,Rt)  
        return(P,Rt,t)

# camera parameters
    K =np.array([[1346.1005953,0,932.163397529403],
        [ 0, 1355.93313621175,654.898679624155],
        [ 0, 0,1]])

#  creating a warp funtion
    def warpPerspective(img, H, size):
        if len(img.shape) == 3: 
            # creating rows and coloums with shape of the image
            h, w, _ = img.shape
            result = np.zeros([size[1], size[0], 3], np.uint8)
        else:
            h, w = img.shape
            result = np.zeros([size[1], size[0]], np.uint8)

        x, y = np.indices((w, h))
        
        img_coords = np.vstack((x.flatten(), 
                                y.flatten(), 
                                [1]*x.size))
        
        new_coords = H @ img_coords
        # normalizing the new cordinates
        new_coords = new_coords/(new_coords[2] + 1e-6)
        

        new_x, new_y , _= np.int0(np.round(new_coords))
        
        new_x[np.where(new_x < 0)] = 0
        new_y[np.where(new_y < 0)] = 0
        new_x[np.where(new_x > size[0] - 1)] = size[0] - 1
        new_y[np.where(new_y > size[1] - 1)] = size[1] - 1
        
        result[new_y, new_x] = img[y.flatten(), x.flatten()]
        
        return result
    
    #FAST FOURIER TRANSFORMS AND INVERSE FOURIER TRANSFORMS
    
    ft = cv.dft(np.float32(threshed), flags=cv.DFT_COMPLEX_OUTPUT)
    
    ft_shift = np.fft.fftshift(ft)

    mag_spectrum = 20 * np.log(cv.magnitude(ft_shift[:, :, 0], ft_shift[:, :, 1]))

    rows, cols = threshed.shape
    center_row, center_col = int(rows / 2), int(cols / 2)  # center

    # MASKING
    # Circular HPF mask, center circle is 0

    mask = np.ones((rows, cols, 2), np.uint8)
    r = 500
    # radius for masking 
    center = [center_row, center_col]
    x, y = np.ogrid[:rows, :cols]
    mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r*r
    mask[mask_area] = 0 #value 0 for high pass filter

    # apply mask and inverse DFT
    fshift = ft_shift * mask

    fshift_mask_mag = 2000 * np.log(cv.magnitude(fshift[:, :, 0], fshift[:, :, 1]))
# creating a inverse fft
    fi_shift = np.fft.ifftshift(fshift)
    img_back = cv.idft(fi_shift)
    img_back = cv.magnitude(img_back[:, :, 0], img_back[:, :, 1])
    cv.imshow("inversefft",img_back) 
#  finding outer corners of sheet using corner
    dst = cv.cornerHarris(img_back,2,3,0.04)
            #   thresholding the corners
    corners = np.array(np.where(dst>0.1*dst.max())).T
    corners = corners[:, ::-1]
# inversing the rows and colums
    x, y = corners.T
    # finding the min and max points
    i=y.argmax()
    j=x.argmin()
    k=y.argmin()
    l=x.argmax()
    # defining the points of the corner
    c0, c1, c2, c3 = corners[(i,j,k,l),]
    cn = (c0+c1+c2+c3)/4
    dt = math.dist(cn, c0)


    cornerpoints = np.array([c0,c1,c2,c3])
    # creating inner corners of the sheet where tag will be placed
    inner_corners = []
    for pt in corners:
        if math.dist(pt, cn) < 0.6*dt:
            inner_corners.append(pt)
    # converting into an array
    inner_corners = np.array(inner_corners)
            
    x, y = inner_corners.T
    i=y.argmax()
    j=x.argmin()
    k=y.argmin()
    l=x.argmax()
    c0, c1, c2, c3 = inner_corners[(i,j,k,l),]

  
    for x, y in inner_corners:
        # creating points to be shown
        cv.circle(frame, (x, y), 2, [0,0,255], -1)
 
    
    cv.circle(frame,c0,2,255,-1)
    cv.circle(frame,c1,2,255,-1)
    cv.circle(frame,c2,2,255,-1)
    cv.circle(frame,c3,2,255,-1) 
    
    cv.putText(frame, 'C0', c0, cv.FONT_HERSHEY_SIMPLEX, 1,((209, 80, 0, 255)),1 )
    cv.putText(frame, 'C1', c1, cv.FONT_HERSHEY_SIMPLEX, 1,((209, 80, 0, 255)),1 )
    cv.putText(frame, 'C2', c2, cv.FONT_HERSHEY_SIMPLEX, 1,((209, 80, 0, 255)),1 )
    cv.putText(frame, 'C3', c3, cv.FONT_HERSHEY_SIMPLEX, 1,((209, 80, 0, 255)),1 )
    
    # print(cornerpoints)
    cv.imshow("corner",frame)
    
    def findHomography(l1,l2):
        try:
            A = []
            for i in range(0, len(l1)):
                x, y = l1[i][0], l1[i][1]
                u, v = l2[i][0], l2[i][1]
                A.append([x, y, 1, 0, 0, 0, -u*x, -u*y, -u])
                A.append([0, 0, 0, x, y, 1, -v*x, -v*y, -v])
            A = np.asarray(A)
            U, S, Vh = np.linalg.svd(A) #Using SVD file
            L = Vh[-1,:] / Vh[-1,-1]
            H = L.reshape(3, 3)
            return(H)
        except:
            pass
    
    # a, d are row and colums of testudo image
    a,d, ch = testudo.shape 
    size = (a,d)
    # rows and colums of gray shape
    w,v = gray.shape
    # inner sheet points where testudo to be placed 
    inner_points= np.array([c0, c1, c2, c3]) 
    # testudo corner points for homography
    desired_tag_corner = np.array([[0, 0],[a,0],[a,d],[0,d]])
    
    
# homography matrix of inner corner and testudo
    h = findHomography(desired_tag_corner,inner_points)
    
   
    #size of the video is in 1280 x 720 pixels and calling warp funtion 
    dst = warpPerspective(testudo, h, (1280, 720))
    # creating a blank black image to superimpose the testudo on it 
    cv.fillConvexPoly(frame,inner_points,(0,0,0))
    # superimposing the image
    frame= frame + dst
    
    
    # for decoding of the tag but the image is showing to black
    def decode_tag(ref_tag_image):
        
        size = 160
        tag_gray = cv.cvtColor(ref_tag_image, cv.COLOR_BGR2GRAY)
        thresh = cv.threshold(tag_gray, 230 ,255,cv.THRESH_BINARY)[1]
        thresh_resize = cv.resize(thresh, (size, size))
        grid_size = 8
        stride = int(size/grid_size)
        grid = np.zeros((8,8))
        x = 0
        y = 0
        for i in range(0, grid_size, 1):
            for j in range(0, grid_size, 1):
                cell = thresh_resize[y:y+stride, x:x+stride]
                if cell.mean() > 255//2:
                    grid[i][j] = 255
                x = x + stride
            x = 0
            y = y + stride
        inner_grid = grid[2:6, 2:6]


        i = 0
        while not inner_grid[3,3] and i<4 :
            inner_grid = np.rot90(inner_grid,1)
            i = i + 1

        
        info_grid = inner_grid[1:3,1:3]
        info_grid_array = np.array((info_grid[0,0],info_grid[0,1],info_grid[1,1],info_grid[1,0]))
        tag_id = 0
        tag_id_bin = []
        for j in range(0,4):
            if(info_grid_array[j]) :
                tag_id = tag_id + 2**(j)
                tag_id_bin.append(1)
            else:
                tag_id_bin.append(0)

            rect = np.zeros((4, 2), dtype="float32")
            # print(type(inner_grid))

        return inner_grid, tag_id, tag_id_bin,i
    decoded, i, o, p = decode_tag(dst)
    cv.imshow("decoded", np.rot90(decoded))
    
    
    ##projection matrix where h is homography and k is the camera parameters
    P,Rt,t = projectionMatrix(h,K)
    x1,y1,z1 = np.matmul(P,[0,0,0,1])
    x2,y2,z2 = np.matmul(P,[0,160,0,1])
    x3,y3,z3 = np.matmul(P,[160,0,0,1])
    x4,y4,z4 = np.matmul(P,[160,160,0,1])
    x5,y5,z5 = np.matmul(P,[0,0,-160,1])
    x6,y6,z6 = np.matmul(P,[0,160,-160,1])
    x7,y7,z7 = np.matmul(P,[160,0,-160,1])
    x8,y8,z8 = np.matmul(P,[160,160,-160,1])
    
    
    #Joining  the coordinates by using line function and dividing by z to normalize it
    
    cv.line(frame,(int(x1/z1),int(y1/z1)),(int(x5/z5),int(y5/z5)), (0,0,255), 2)
    cv.line(frame,(int(x2/z2),int(y2/z2)),(int(x6/z6),int(y6/z6)), (0,0,255), 2)
    cv.line(frame,(int(x3/z3),int(y3/z3)),(int(x7/z7),int(y7/z7)), (0,0,255), 2)
    cv.line(frame,(int(x4/z4),int(y4/z4)),(int(x8/z8),int(y8/z8)), (0,0,255), 2)

    cv.line(frame,(int(x1/z1),int(y1/z1)),(int(x2/z2),int(y2/z2)), (0,255,0), 2)
    cv.line(frame,(int(x1/z1),int(y1/z1)),(int(x3/z3),int(y3/z3)), (0,255,0), 2)
    cv.line(frame,(int(x2/z2),int(y2/z2)),(int(x4/z4),int(y4/z4)), (0,255,0), 2)
    cv.line(frame,(int(x3/z3),int(y3/z3)),(int(x4/z4),int(y4/z4)), (0,255,0), 2)

    cv.line(frame,(int(x5/z5),int(y5/z5)),(int(x6/z6),int(y6/z6)), (255,0,0), 2)
    cv.line(frame,(int(x5/z5),int(y5/z5)),(int(x7/z7),int(y7/z7)), (255,0,0), 2)
    cv.line(frame,(int(x6/z6),int(y6/z6)),(int(x8/z8),int(y8/z8)), (255,0,0), 2)
    cv.line(frame,(int(x7/z7),int(y7/z7)),(int(x8/z8),int(y8/z8)), (255,0,0), 2)
    
    cv.imshow("superimposed cube and testudo",frame)
    
    
    out.write(frame)  
    
    plt.subplot(2, 2, 1), plt.imshow(threshed, cmap='gray')
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(2, 2, 2), plt.imshow(mag_spectrum, cmap='gray')
    plt.title('After FFT'), plt.xticks([]), plt.yticks([])
    plt.subplot(2, 2, 3), plt.imshow(fshift_mask_mag, cmap='gray')
    plt.title('FFT + Mask'), plt.xticks([]), plt.yticks([])
    plt.subplot(2, 2, 4), plt.imshow(img_back, cmap='gray')
    plt.title('After FFT Inverse'), plt.xticks([]), plt.yticks([])
    
    # ###### below lines of code for plotting of fft, blur , threshed in one image

    
    # fx, plts = plt.subplots(2,2,figsize = (15, 10))
    # plts[0][0].imshow(threshed, cmap = 'gray')
    # plts[0][0].set_title('Thresholded image')

    # # plts[0][1].imshow(20*np.log(np.abs(fshift)), cmap = 'gray')
    # plts[0][1].set_title('Shifted FFT')

    # plts[1][0].imshow(20*np.log(np.abs(fshift_mask_mag)), cmap = 'gray')
    # plts[1][0].set_title('Mask + FFT')

    # # plts[1][1].imshow(, cmap = 'gray')
    # # plts[1][1].set_title('Edges detected')

    # plt.savefig("edge_detection.jpg")
    # plt.show()

    
    if cv.waitKey(20) & 0xFF==ord("q"):
        break
out.release()    
vid.release()

plt.show()
cv.destroyAllWindows()

