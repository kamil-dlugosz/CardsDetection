from config import *
from card import *

from tqdm import tqdm
import numpy as np
import cv2


if __name__ == '__main__':
    cards_list = Card.read_cards(source_dir=CARD_PICTURES_DIR,
                                 target_lights='03',
                                 target_numbers='01',
                                 target_decks='02',
                                 target_ranks='JA')
    for card in cards_list:
        bgr = card.image
        bgr = cv2.resize(bgr, dsize=(600, 800))
        cv2.imshow('bgr', bgr)

        blur = cv2.GaussianBlur(bgr, ksize=(5, 5), sigmaX=5)
        cv2.imshow('blur', blur)

        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
        # cv2.imshow('HSV', hsv)
        mask = cv2.inRange(hsv, lowerb=np.array([20, 0, 0]), upperb=np.array([60, 255, 255]))
        mask = cv2.bitwise_not(mask)
        cv2.imshow('mask', mask)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        smooth_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=5)
        cv2.imshow('smooth_mask', smooth_mask)

        cards_img = cv2.bitwise_and(bgr, bgr, mask=smooth_mask)
        cv2.imshow('cards_img', cards_img)

        # todo invert mask, close/open,
        cv2.waitKey(0)
        cv2.destroyAllWindows()
