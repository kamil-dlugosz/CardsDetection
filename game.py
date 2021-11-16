import cv2
import os


def batch_int(*args):
    return tuple(int(a) for a in args)


class Game:
    def __init__(self):
        self.state = 'calm'
        self.placeholders = dict()
        for filename in os.listdir('../placeholder-cards'):
            key = os.path.splitext(filename)[0]
            file_path = os.path.join(os.getcwd(), '..', 'placeholder-cards', filename)
            image = cv2.imread(file_path, flags=-2)
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
            self.placeholders[key] = image
        self.score_1 = 0
        self.score_2 = 0
        self.cards_power = {
            'AH': 14,     'AD': 14,     'AC': 14,     'AS': 14,
            'KH': 13,     'KD': 13,     'KC': 13,     'KS': 13,
            'QH': 12,     'QD': 12,     'QC': 12,     'QS': 12,
            'JH': 11,     'JD': 11,     'JC': 11,     'JS': 11,
            'TH': 10,     'TD': 10,     'TC': 10,     'TS': 10,
            '9H': 9,      '9D': 9,      '9C': 9,      '9S': 9,
            '8H': 8,      '8D': 8,      '8C': 8,      '8S': 8,
            '7H': 7,      '7D': 7,      '7C': 7,      '7S': 7,
            '6H': 6,      '6D': 6,      '6C': 6,      '6S': 6,
            '5H': 5,      '5D': 5,      '5C': 5,      '5S': 5,
            '4H': 4,      '4D': 4,      '4C': 4,      '4S': 4,
            '3H': 3,      '3D': 3,      '3C': 3,      '3S': 3,
            '2H': 2,      '2D': 2,      '2C': 2,      '2S': 2,
        }
        # self.cards_detections = {
        #     'AH': False,  'AD': False,  'AC': False,  'AS': False,
        #     'KH': False,  'KD': False,  'KC': False,  'KS': False,
        #     'QH': False,  'QD': False,  'QC': False,  'QS': False,
        #     'JH': False,  'JD': False,  'JC': False,  'JS': False,
        #     'TH': False,  'TD': False,  'TC': False,  'TS': False,
        #     '9H': False,  '9D': False,  '9C': False,  '9S': False,
        #     '8H': False,  '8D': False,  '8C': False,  '8S': False,
        #     '7H': False,  '7D': False,  '7C': False,  '7S': False,
        #     '6H': False,  '6D': False,  '6C': False,  '6S': False,
        #     '5H': False,  '5D': False,  '5C': False,  '5S': False,
        #     '4H': False,  '4D': False,  '4C': False,  '4S': False,
        #     '3H': False,  '3D': False,  '3C': False,  '3S': False,
        #     '2H': False,  '2D': False,  '2C': False,  '2S': False,
        # }

    def group_cards(self, detections):
        cards = dict()
        for name, confidence, (x, y, w, h) in detections:
            if name not in cards:
                cards[name] = list()
            cards[name].append((x, y, w, h))
        return cards

    # def smooth_detection(self, detections):
    #     smooth_level = 2
    #     smoothed_detections = dict()
    #     for name, _, (x, y, w, h) in detections:
    #         past_detection = self.cards_detections[name]

    def assign_cards_to_players(self, cards, threshold):
        p1_cards, p2_cards = list(), list()
        for name, boxes in cards.items():
            try:
                (_, y1, _, _) = boxes[0]
                (_, y2, _, _) = boxes[1]
            except IndexError:
                continue
            if (y1 + y2)/2 < threshold:
                p1_cards.append((name, *cards[name]))
            else:
                p2_cards.append((name, *cards[name]))
        return p1_cards, p2_cards

    def count_points(self, p1_cards, p2_cards):
        if self.state == 'calm' and len(p1_cards) == len(p2_cards) == 1:
            self.state = 'fight'
            p1_card, p2_card = p1_cards[0], p2_cards[0]
            if self.cards_power[p1_card[0]] > self.cards_power[p2_card[0]]:
                self.score_1 += 1
            elif self.cards_power[p1_card[0]] < self.cards_power[p2_card[0]]:
                self.score_2 += 1
            # else:
            #     self.state = 'war'
        elif len(p1_cards) + len(p2_cards) > 2:
            self.state = 'too_many'
        else:
            self.state = 'calm'

    def step(self, image, detections):
        cards = self.group_cards(detections)
        p1_cards, p2_cards = self.assign_cards_to_players(cards, image.shape[1]/2)
        self.count_points(p1_cards, p2_cards)
        return self.draw(image, p1_cards, p2_cards)

    def draw(self, image, p1_cards, p2_cards):
        # detections = self.smooth_detection(detections)
        hi, wi, ci = image.shape
        padded_image = cv2.copyMakeBorder(src=image, top=hi, bottom=hi, left=wi, right=wi,
                                          borderType=cv2.BORDER_CONSTANT, value=0)
        for name, (x0, y0, w0, h0), (x1, y1, w1, h1) in p1_cards + p2_cards:
            x0, x1, y0, y1, w0, w1, h0, h1 = batch_int(x0, x1, y0, y1, w0, w1, h0, h1)
            xl, xr, yt, yd = min(x0-w0, x1-w1), max(x0+w0, x1+w1), min(y0-h0, y1-h1), max(y0+h0, y1+h1)

            placeholder = self.placeholders[name].copy()
            placeholder = cv2.resize(src=placeholder, dsize=(xr-xl, yd-yt))

            alpha = placeholder[:, :, 3] / 255.0
            beta = 1.0 - alpha
            for channel in range(0, 3):
                padded_image[yt+hi:yd+hi, xl+wi:xr+wi, channel] = \
                    (alpha * placeholder[:, :, channel] + beta * padded_image[yt+hi:yd+hi, xl+wi:xr+wi, channel])
        image = padded_image[hi:hi*2, wi:wi*2]

        image = cv2.resize(src=image, dsize=(image.shape[0]*2, image.shape[1]*2))
        hi, wi = hi*2, wi*2

        # HUD
        image = cv2.line(img=image, pt1=(0, int(hi/2)), pt2=(wi-1, int(hi/2)), color=(150, 20, 150), thickness=6)
        image = cv2.putText(img=image, text=f"state: {self.state}", org=(wi-200, 20),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(150, 20, 150))

        # Player 1
        image = cv2.putText(img=image, text=f"player 1 score: {self.score_1}", org=(10, 20),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, color=(50, 180, 15))
        p1_cards_names = list()
        for name, _, _ in p1_cards:
            p1_cards_names.append(name)
        image = cv2.putText(img=image, text=f"player 1 cards: {' '.join(p1_cards_names)}", org=(10, 40),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(50, 180, 15))

        # Player 2
        image = cv2.putText(img=image, text=f"player 2 score: {self.score_2}", org=(10, hi-20),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, color=(180, 15, 50))
        p2_cards_names = list()
        for name, _, _ in p2_cards:
            p2_cards_names.append(name)
        image = cv2.putText(img=image, text=f"player 2 cards: {' '.join(p2_cards_names)}", org=(10, hi-40),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(180, 15, 50))

        return image


game = Game()
