import sys
sys.path.append('RAFT')

import argparse
import cv2
import torch
import numpy as np
from scipy import io

from RAFT import RAFT, flow_viz

DEVICE = 'cpu'

args = argparse.Namespace(alternate_corr = False, mixed_precision = False, 
                          model = 'RAFT/models/raft-kitti.pth', path = 'mots1/frames', small = False)

model = torch.nn.DataParallel(RAFT(args))
model.load_state_dict(torch.load(args.model, map_location = DEVICE))

model = model.module
model.to(DEVICE)
model.eval()

name = sys.argv[1]
print(name)
cap = cv2.VideoCapture(f'data/{name}/video.mp4')
#out = cv2.VideoWriter('optical_flow.mp4', cv2.VideoWriter_fourcc('M','J','P','G'), 10, (272, 960))

print('hi')
i = 1
prev_img = None
with torch.no_grad():
    while(cap.isOpened()):

        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
            print(i)
            nm = str(i).zfill(3)
            frame = cv2.resize(frame, (480, 272))

            cv2.imwrite(f'data/{name}/frames/{name}_{nm}.png',frame)

            img = torch.from_numpy(frame).permute(2, 0, 1).float()
            img = img[None].to(DEVICE)

            if prev_img is not None:
                image1 = prev_img
                image2 = img

                flow_low, flow_up = model(image1, image2, iters = 30, test_mode=True)

                flov = flow_up[0].permute(1,2,0).cpu().numpy()
                imgn = image1[0].permute(1,2,0).cpu().numpy()

                mat = {"uv": flov}

                nm1 = str(i-1).zfill(3)
                
                io.savemat(f"data/{name}/opticalflow/OF{nm1}.mat", mat)

                # map flow to rgb image
                flo = flow_viz.flow_to_image(flov)
                img_flo = np.concatenate([imgn, flo], axis = 1).astype(np.uint8)

                cv2.imshow('flow', img_flo)
                #out.write(np.uint8(img_flo))
            i = i + 1
                
            prev_img = img
            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        # Break the loop
        else: break

# When everything done, release the video capture object
cap.release()
#out.release()

# Closes all the frames
cv2.destroyAllWindows()


