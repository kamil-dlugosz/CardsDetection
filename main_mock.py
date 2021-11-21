import pickle
import cv2
from game import game


if __name__ == '__main__':
    with open('../det_img.pickle', 'rb') as handle:
        detections, image = pickle.load(handle)
    print(detections)
    image = game.step(image, detections)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.imshow('mock', image)
    cv2.waitKey(0)
