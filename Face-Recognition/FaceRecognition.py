import cv2
import numpy as np
import os
from PIL import Image

def create_user(f_id, name):
    web = cv2.VideoCapture(0)
    web.set(3,640)
    web.set(4,480)

    faces = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    f_dir = 'dataset'
    path = os.path.join(f_dir, name)
    os.makedirs(path, exist_ok=True)

    counter = 0
    while True:
        ret, img = web.read()
        img = cv2.flip(img, 1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        multi_face = faces.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in multi_face:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            counter += 1

            cv2.imwrite("{}/{}.{}.{}{}".format(path, name, f_id, counter, ".jpg"), gray[y:y+h, x:x+w])
            cv2.imshow("Image",img)
        k = cv2.waitKey(100) & 0xff
        if k == 27:
            break
        elif counter>=40:
            break
    web.release()
    cv2.destroyAllWindows()

def train():
    database = 'dataset'
    img_dir = [x[0] for x in os.walk(database)][1::]
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    detector = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')
    faceSamples = []
    ids = []
    name_dict={}

    for idx, path in enumerate(img_dir):
        name = os.path.basename(path)
        name_dict[idx] = name
        imagePaths = [os.path.join(path,f) for f in os.listdir(path)]

        for imagePath in imagePaths:
            PIL_img = Image.open(imagePath).convert('L')
            img_numpy = np.array(PIL_img, 'uint8')

            faces = detector.detectMultiScale(img_numpy)
            for (x, y, w, h) in faces:
                faceSamples.append(img_numpy[y:y+h, x:x+w])
                ids.append(idx)

    recognizer.train(faceSamples, np.array(ids))
    recognizer.write('trainer.yml')

    with open("names.txt", "w") as f:
        for k, v in name_dict.items():
            f.write(f"{k}:{v}\n")

    print('\n[INFO] {0} faces trained . Exiting Program'.format(len(np.unique(ids))))

def load_name_dict():
    name_dict = {}
    if os.path.exists("names.txt"):
        with open("names.txt", "r") as f:
            for line in f:
                key, val = line.strip().split(":")
                name_dict[int(key)] = val
    return name_dict

def recognize(names):
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('trainer.yml')
    cascade = "haarcascade_frontalface_default.xml"
    cascadePath = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    faceCascade = cv2.CascadeClassifier(cascadePath)

    font = cv2.FONT_HERSHEY_SIMPLEX
    prev_name = ''
    face_count = 0

    cam = cv2.VideoCapture(0)
    cam.set(3,640)
    cam.set(4,480)

    minW = 0.1 * cam.get(3)
    minH = 0.1 * cam.get(4)
    while True:
        ret, img = cam.read()
        img = cv2.flip(img, 1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors = 5,
            minSize = (int(minW), int(minH))
        )
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            id, confidence = recognizer.predict(gray[y:y+h, x:x+w])
            if confidence < 70:
                name_from_dict = names.get(id, "Unknown")
                display_text = f"{name_from_dict} ({round(100 - confidence)}%)"
            else :
                name_from_dict = 'unknown'
                display_text = 'Not recognized'

            if prev_name == name_from_dict:
                face_count += 1
                if face_count>21:
                    face_count = -100
            else:
                prev_name = name_from_dict
                face_count = 0

            cv2.putText(img, display_text, (x + 5, y - 5), font, 1, (0, 0, 255), 2)

        cv2.imshow("Camera",img)
        k = cv2.waitKey(100) & 0xff
        if k == 27:
            break

    cam.release()
    cv2.destroyAllWindows()

def main():
    while True:
        print("\n========= FACE RECOGNITION SYSTEM =========")
        print("1. Add New User")
        print("2. Train Model")
        print("3. Recognize Faces")
        print("4. Exit")
        choice = input("Enter your choice: ")

        if choice == '1':
            f_id = input("Enter user ID (integer): ")
            name = input("Enter user name: ")
            create_user(f_id, name)
        elif choice == '2':
            train()
        elif choice == '3':
            name_dict = load_name_dict()
            if not name_dict:
                print("[ERROR] No trained data found. Please train first.")
            else:
                recognize(name_dict)
        elif choice == '4':
            print("Exiting program.")
            break
        else:
            print("Invalid choice. Try again.")

if __name__ == "__main__":
    main()
