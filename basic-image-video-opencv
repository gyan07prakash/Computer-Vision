# opening an image
import cv2
img = cv2.imread('spidy.jpg')
img = cv2.resize(img, (569,329))
cv2.imshow("image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# rgb to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("GRAYimage", gray)
cv2.waitKey(0)
cv2.destroyAllWindows()

# rgb to HSV
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
cv2.imshow("GRAYimage", hsv)
cv2.waitKey(0)
cv2.destroyAllWindows()


# working on video
vid = cv2.VideoCapture("video.mp4")

while True:
    res, frame = vid.read()
    if not res:
        break

    cv2.imshow("video.mp4", frame)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()

#for webcam
vid = cv2.VideoCapture(0)
while True:
    try:
        res, frame = vid.read()
        cv2.imshow("video.mp4", frame)
    except:
        pass
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
