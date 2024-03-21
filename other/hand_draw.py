import cv2
import mediapipe as mp

print(cv2.__version__)
width = 1280
height = 720
cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
cam.set(cv2.CAP_PROP_FPS, 30)
cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
hands = mp.solutions.hands.Hands(False, 2, 1, 0.5, 0.5) # false because it wont be a static image, 1 number of hand, .5 confidence tracking 
mpDraw = mp.solutions.drawing_utils

while True:
    myHands = []
    ignore, frame = cam.read()
    frameRGB = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    results = hands.process(frameRGB)

    if results.multi_hand_landmarks != None:
        for handLandMarks in results.multi_hand_landmarks:
            myHand = []
            # mpDraw.draw_landmarks(frame, handLandMarks, mp.solutions.hands.HAND_CONNECTIONS)
            for Landmark in handLandMarks.landmark:
                myHand.append((int(Landmark.x*width), int(Landmark.y*height)))
            
            print(' ')
            print(myHand)
            cv2.circle(frame, myHand[20], 25, (255,0,255),-1)
            myHands.append(myHand)
            print(myHands)
            print(' ')

    cv2.imshow("my WEBcam", frame)
    cv2.moveWindow("my WEBcam", 1, 1)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cam.release()
