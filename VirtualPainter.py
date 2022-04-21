import cv2
import numpy as np
import time
import os
import math
import HandTrackingModule as htm

folderPath = "Virtual Hand painter"
mylist = os.listdir(folderPath)
overlayList = []

for imPath in mylist:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)

header = overlayList[0]
drawColor = (87,87,255) #coral red

cap = cv2.VideoCapture(0)

cap.set(3, 1080)

detector = htm.handDetector(detectionCon=0.85)
brushThickness = 10
eraserThickness = 50

xp, yp = 0, 0
tempSuc, tempImg = cap.read()
imgCanvas = np.zeros((tempImg.shape[0],tempImg.shape[1],3), np.uint8)

while True:

    #1. Import Image
    success, img = cap.read()
    img = cv2.flip(img, 1)

    #2. find hand landmarks
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw = False)

    if len(lmList) != 0:

        #tips of index finger x1, y1, middle finger, x2,y2
        x1, y1 = lmList[8][1], lmList[8][2]
        x2, y2 = lmList[12][1], lmList[12][2]
        #tip of the thumb x0 ,y0
        x0, y0 = lmList[4][1], lmList[4][2]


        #3. checking which fingers are up
        fingers = detector.fingersUp()

        # print(fingers)
# ------------------------------------------------------------------------------------------------------------------
        #4. if selection mode i.e. 2 fingers are up, then we have to select, not draw
        if fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 0 :
            xp, yp = 0, 0

            # print("selection mode")

            #checking for the click
            if y1 < 125:
                if 120 < x1 < 360:
                    header = overlayList[2]
                    drawColor = (55, 128, 0) #green

                elif 400 < x1 <500:
                    header = overlayList[0]
                    drawColor = (87,87,255) #coral red

                elif 600 < x1 < 750:
                    header = overlayList[3]
                    drawColor = (255, 113, 82)  # royal blue

                #eraser
                if 800 < x1 < 1080:
                    header = overlayList[1]
                    drawColor = (0, 0, 0)  # black

            cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), drawColor, cv2.FILLED)
# ------------------------------------------------------------------------------------------------------------------
        #5. check drawing mode, index finger is up
        if fingers[1] == 1 and fingers[2] == 0 and fingers[3] == 0:
            cv2.circle(img, (x1,y1), 10, drawColor, cv2.FILLED)

            # print("drawing mode")

            if xp == 0 and yp == 0:
                xp, yp = x1,y1

            if drawColor == (0,0,0):
                cv2.line(img, (xp, yp), (x1, y1), drawColor, eraserThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraserThickness)
            else:
                cv2.line(img, (xp,yp), (x1,y1), drawColor, brushThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)

            xp,yp = x1,y1
#------------------------------------------------------------------------------------------------------------------
        #6. Add in a brush change size mode which is initiated by holding 3 fingers up

        if (fingers[1] and fingers[2] and fingers[3] == True) and fingers[4] == False:
            cx, cy = (x1 + x0) // 2, (y1 + y0) // 2

            cv2.circle(img, (x1, y1), 10, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x0, y0), 10, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)

            cv2.line(img, (x1, y1), (x0, y0), (255, 0, 255), 3)

            length = math.hypot(x1 - x0, y1 - y0)

            #check what is selected right now, according to that we will change the size of
            #either the paint brush or the eraser

            if drawColor == (0,0,0):
                eraserThickness = int(np.interp(length, [15, 170], [25, 100]))

            else:
                brushThickness = int(np.interp(length, [15, 170], [10, 100]))

# ------------------------------------------------------------------------------------------------------------------

    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, imgCanvas)


    #setting the header image
    img[0:125, 0:1080] = header
    # img = cv2.addWeighted(img, 0.5, imgCanvas, 0.5, 0)
    # cv2.imshow("Canvas", imgCanvas)
    cv2.imshow("Virtual Painter", img)
    cv2.waitKey(1)
