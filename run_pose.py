# To use Inference Engine backend, specify location of plugins:
# export LD_LIBRARY_PATH=/opt/intel/deeplearning_deploymenttoolkit/deployment_tools/external/mklml_lnx/lib:$LD_LIBRARY_PATH
import cv2 as cv
import numpy as np
import argparse
# import imutils
import time
import glob

parser = argparse.ArgumentParser(
        description='This script is used to demonstrate OpenPose human pose estimation network '
                    'using MPI pretrained model and OpenCV. '
                    'The sample and model are simplified and could be used for a single person on the frame.')
parser.add_argument('--thr', default=0.1, type=float, help='Threshold value for pose parts heat map')
parser.add_argument('--width', default=368, type=int, help='Resize input to specific width.')
parser.add_argument('--height', default=368, type=int, help='Resize input to specific height.')

args = parser.parse_args()

BODY_PARTS = { "Head": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
                "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
                "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "Chest": 14,
                "Background": 15 }

POSE_PAIRS = [ ["Head", "Neck"], ["Neck", "RShoulder"], ["RShoulder", "RElbow"],
                ["RElbow", "RWrist"], ["Neck", "LShoulder"], ["LShoulder", "LElbow"],
                ["LElbow", "LWrist"], ["Neck", "Chest"], ["Chest", "RHip"], ["RHip", "RKnee"],
                ["RKnee", "RAnkle"], ["Chest", "LHip"], ["LHip", "LKnee"], ["LKnee", "LAnkle"] ]

inWidth = args.width
inHeight = args.height

net = cv.dnn.readNetFromCaffe('pose/mpi/pose_deploy_linevec_faster_4_stages.prototxt', 'pose/mpi/pose_iter_160000.caffemodel')

for image in glob.glob("samples/*.jpg"):
    frame = cv.imread(image)
    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]

    inp = cv.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight),
                                (0, 0, 0), swapRB=False, crop=False)
    net.setInput(inp)
    start_t = time.time()
    out = net.forward()
        
    print("time is ",time.time()-start_t)
    # print(inp.shape)
    kwinName="Pose Estimation Demo: Cv-Tricks.com"
    cv.namedWindow(kwinName, cv.WINDOW_AUTOSIZE)
    #assert(len(BODY_PARTS) == out.shape[1])

    points = []
    for i in range(len(BODY_PARTS)):
        # Slice heatmap of corresponging body's part.
        heatMap = out[0, i, :, :]

        # Originally, we try to find all the local maximums. To simplify a sample
        # we just find a global one. However only a single pose at the same time
        # could be detected this way.
        _, conf, _, point = cv.minMaxLoc(heatMap)
        x = (frameWidth * point[0]) / out.shape[3]
        y = (frameHeight * point[1]) / out.shape[2]

        # Add a point if it's confidence is higher than threshold.
        points.append((int(x), int(y)) if conf > args.thr else None)

    for pair in POSE_PAIRS:
        partFrom = pair[0]
        partTo = pair[1]
        assert(partFrom in BODY_PARTS)
        assert(partTo in BODY_PARTS)

        idFrom = BODY_PARTS[partFrom]
        idTo = BODY_PARTS[partTo]
        if points[idFrom] and points[idTo]:
            cv.line(frame, points[idFrom], points[idTo], (255, 74, 0), 3)
            cv.ellipse(frame, points[idFrom], (4, 4), 0, 0, 360, (255, 255, 255), cv.FILLED)
            cv.ellipse(frame, points[idTo], (4, 4), 0, 0, 360, (255, 255, 255), cv.FILLED)
            cv.putText(frame, str(idFrom), points[idFrom], cv.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255),2,cv.LINE_AA)
            cv.putText(frame, str(idTo), points[idTo], cv.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255),2,cv.LINE_AA)
        
    t, _ = net.getPerfProfile()
    freq = cv.getTickFrequency() / 1000
    cv.putText(frame, '%.2fms' % (t / freq), (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255),2,cv.LINE_AA)

    cv.imshow(kwinName, frame)
    cv.imwrite('result_'+image[8:],frame)