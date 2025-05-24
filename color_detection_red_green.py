import numpy as np
import cv2
import matplotlib.pyplot as plt

web = cv2.VideoCapture(0)
while(1):
    _, img = web.read()
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    green_L = np.array([30, 55, 75], np.uint8)
    green_U = np.array([105, 250, 250], np.uint8)
    greenMask = cv2.inRange(hsv, green_L, green_U)
    kernel = np.ones((5, 5), 'uint8')
    greenMask = cv2.dilate(greenMask, kernel)
    green = cv2.bitwise_and(img, img, mask=greenMask)

    contours1, hierarchy1 = cv2.findContours(greenMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour1 in contours1:
        area1 = cv2.contourArea(contour1)
        if (area1 > 300):
            x, y, w, h = cv2.boundingRect(contour1)
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img, "Green Colour", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


    red_L = np.array([135, 85, 115], np.uint8)
    red_U = np.array([185, 250, 250], np.uint8)
    redMask = cv2.inRange(hsv, red_L, red_U)
    kernel = np.ones((5, 5), 'uint8')
    redMask = cv2.dilate(redMask, kernel)
    red = cv2.bitwise_and(img, img, mask=redMask)

    contours, hierarchy = cv2.findContours(redMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if (area > 300):
            x,y,w,h = cv2.boundingRect(contour)
            img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
            cv2.putText(img, "Red Colour", (x,y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)


    cv2.imshow('Detect color', img)
    if cv2.waitKey(27) & 0xFF == ord ('q'):
        web.release()
        cv2.destroyAllWindows()
        break
