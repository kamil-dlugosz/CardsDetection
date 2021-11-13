import cv2
import os


def batch_int(*args):
    return tuple(int(a) for a in args)


class Game:
    def __init__(self):
        self.placeholders = dict()
        for filename in os.listdir('../placeholder-cards'):
            key = os.path.splitext(filename)[0]
            file_path = os.path.join(os.getcwd(), '..', 'placeholder-cards', filename)
            self.placeholders[key] = cv2.imread(file_path)

    def group_cards(self, detections):
        cards = dict()
        for name, confidence, (x, y, w, h) in detections:
            if name not in cards:
                cards[name] = list()
            cards[name].append((x, y, w, h))
        return cards

    def draw(self, image, detections):
        cards = self.group_cards(detections)
        for name, boxes in cards.items():
            try:
                (x0, y0, w0, h0) = boxes[0]
                (x1, y1, w1, h1) = boxes[1]
            except IndexError:
                continue
            x0, x1, y0, y1, w0, w1, h0, h1 = batch_int(x0, x1, y0, y1, w0, w1, h0, h1)
            xl, xr, yt, yd = min(x0, x1), max(x0, x1), min(y0, y1), max(y0, y1)
            placeholder = self.placeholders[name].copy()
            placeholder = cv2.resize(placeholder, (xr-xl, yd-yt))
            image[yt:yd, xl:xr] = placeholder
        return image


game = Game()
