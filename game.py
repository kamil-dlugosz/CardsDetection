import cv2
import os
import numpy as np
from PIL import Image, ImageDraw
from math import copysign


CARD_COS = 0.5812381937190965
CARD_SIN = 0.813733471206735


class Game:
    def __init__(self):
        self.state = 'calm'
        self.score_1 = 0
        self.score_2 = 0
        self.placeholders = dict()
        self.msg = list()
        for filename in os.listdir('../placeholder-cards'):
            key = os.path.splitext(filename)[0]
            file_path = os.path.join(os.getcwd(), '..', 'placeholder-cards', filename)
            image = Image.open(file_path)
            image = image.convert(mode='RGBA')
            self.placeholders[key] = image
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

    def batch_int(self, *args):
        """    Cast floats to ints.    """
        return tuple(int(a) for a in args)

    def sort_boxes(self, box1, box2):
        """    Sort 2 yolo bounding boxes by y value (upper point is first).    """
        _, y1, _, _ = box1
        _, y2, _, _ = box2
        if y1 < y2:
            return box1, box2
        else:
            return box2, box1

    def group_cards(self, detections):
        """    Create dict containing both detections of corners for each card.    """
        cards = dict()
        for name, confidence, (x, y, w, h) in detections:
            if name not in cards:
                cards[name] = list()
            # x, y, w, h = self.batch_int(x, y, w, h)
            cards[name].append((x, y, w, h))
        return cards

    # def smooth_detection(self, detections):
    #     smooth_level = 2
    #     smoothed_detections = dict()
    #     for name, _, (x, y, w, h) in detections:
    #         past_detection = self.cards_detections[name]

    def assign_cards_to_players(self, cards, threshold):
        """   Divide cards to two lists, one for each player based on treshold.   """
        p1_cards, p2_cards = list(), list()
        for name, boxes in cards.items():
            try:
                (_, y1, _, _) = boxes[0]
                (_, y2, _, _) = boxes[1]
            except IndexError:
                continue
            card = name, *self.sort_boxes(boxes[0], boxes[1])
            if (y1 + y2)/2 < threshold:
                p1_cards.append(card)
            else:
                p2_cards.append(card)
        return p1_cards, p2_cards

    def count_points(self, p1_cards, p2_cards):
        """    Count points for each player and change state of game.    """
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
        elif len(p1_cards) + len(p2_cards) == 0:
            self.state = 'calm'

    def step(self, image, detections):
        """   Make game step.   """
        self.msg.clear()
        cards = self.group_cards(detections)
        p1_cards, p2_cards = self.assign_cards_to_players(cards, image.shape[1]/2)
        self.count_points(p1_cards, p2_cards)
        return self.draw(image, p1_cards, p2_cards)

    def rotate_pv(self, point, angle=np.pi/2, cos=None, sin=None, origin=(0, 0)):
        """    Rotate a point counter-clockwise by a given angle (radians) around a given origin.    """
        if point.shape != (2,):
            raise UserWarning
        ox, oy = origin
        px, py = point

        if cos is None or sin is None:
            cos = np.cos(angle)
            sin = np.sin(angle)

        qx = ox + cos * (px - ox) - sin * (py - oy)
        qy = oy + sin * (px - ox) + cos * (py - oy)

        return np.array([qx, qy])

    def unit_vector(self, *args):
        """    Return unit vector from given vector.    """
        vector = np.array(args)
        return vector / np.linalg.norm(vector)

    def card_rect(self, x1, y1, x3, y3):
        """    Return card vectors and corners.    """
        diagonal_vector = np.array([x3 - x1, y3 - y1])
        diagonal = np.linalg.norm(diagonal_vector)

        short_vector = self.rotate_pv(point=diagonal_vector, cos=CARD_COS, sin=-CARD_SIN)
        short_vector = self.unit_vector(*short_vector)*diagonal*CARD_COS

        long_vector = self.rotate_pv(point=diagonal_vector, cos=CARD_SIN, sin=CARD_COS)
        long_vector = self.unit_vector(*long_vector)*diagonal*CARD_SIN

        x2, y2 = (x1, y1) + long_vector
        x4, y4 = (x1, y1) + short_vector

        return (diagonal_vector, short_vector, long_vector), ((x1, y1), (x2, y2), (x3, y3), (x4, y4))

    def card_angle(self, normal_vector):
        """    Calculate angle of whole card rotation.    """
        # Init unit vectors, those axes are perpendicular, both axes are rotated, so angle when card is staight is 0*
        # x_axis_vector = self.unit_vector(3.5, -2.5)
        # y_axis_vector = self.unit_vector(2.5, 3.5)
        # card_vector = self.unit_vector(*args)
        x_axis_vector = self.unit_vector(1, 0)
        y_axis_vector = self.unit_vector(0, 1)
        normal_vector = self.unit_vector(*normal_vector)

        # Dot product of two vectors => cosinus of angle, ensure that cosinus is in [-1, 1]
        cosinus_x = np.dot(x_axis_vector, normal_vector)
        cosinus_x = np.clip(cosinus_x, -1.0, 1.0)
        cosinus_y = np.dot(y_axis_vector, normal_vector)
        cosinus_y = np.clip(cosinus_y, -1.0, 1.0)

        # Calculate angle and it's sign
        sign = cosinus_x
        value = np.degrees(np.arccos(cosinus_y))
        angle = copysign(value, sign)

        self.msg.append(f"angle = {round(angle, 2):<5}*")
        self.msg.append(f"value = {round(value, 2):<5}*")
        self.msg.append(f"sign  = {round(sign, 2):<5}*")

        return angle

    def draw(self, image, p1_cards, p2_cards):
        """    Draw game specific items on image.    """
        # detections = self.smooth_detection(detections)
        # Convert to PIL
        image = Image.fromarray(image)
        wi, hi = image.size

        # Pad image with black (so cards can be easily pasted outside image bounds)
        padded_image = Image.new(mode='RGBA', size=(wi*3, hi*3), color=(0, 0, 0, 0))
        padded_image.paste(im=image, box=(wi, hi))

        # Paste cards to padded image
        for name, (x1, y1, w1, h1), (x3, y3, w3, h3) in p1_cards + p2_cards:
            # Find bounding box of whole card
            left, right, top, bottom = self.batch_int(min(x1-w1, x3-w3), max(x1+w1, x3+w3),
                                                      min(y1-h1, y3-h3), max(y1+h1, y3+h3))
            (diagonal_vector, short_vector, long_vector), ((x1, y1), (x2, y2), (x3, y3), (x4, y4)) \
                = self.card_rect(x1, y1, x3, y3)

            # Get, rotate and resize card placeholder
            angle = self.card_angle(long_vector)
            placeholder = self.placeholders[name].copy()
            placeholder = placeholder.resize(size=(right-left, bottom-top))
            placeholder = placeholder.rotate(angle=angle,
                                             expand=True, fillcolor=(0, 0, 0, 0))
            # Cover card with placeholder
            padded_image.alpha_composite(im=placeholder, dest=(wi+left, hi+top))

            # Draw bouiding boxes of card and placeholder
            drawer = ImageDraw.Draw(padded_image)
            drawer.rectangle(xy=(wi+left, hi+top, wi+right, hi+bottom), outline="red")
            drawer.ellipse(xy=(wi+x1-5, hi+y1-5, wi+x1+5, hi+y1+5), outline="blue")
            drawer.ellipse(xy=(wi+x2-5, hi+y2-5, wi+x2+5, hi+y2+5), outline="yellow")
            drawer.ellipse(xy=(wi+x3-5, hi+y3-5, wi+x3+5, hi+y3+5), outline="green")
            drawer.ellipse(xy=(wi+x4-5, hi+y4-5, wi+x4+5, hi+y4+5), outline="purple")

        # Crop image from padded image
        image = padded_image.crop(box=(wi, hi, wi*2, hi*2))

        # Double size
        image = image.resize(size=(wi*2, hi*2))
        wi, hi = image.size

        # Convert back to OpenCV
        image = np.asarray(a=image)

        # Draw HUD
        image = cv2.line(img=image, pt1=(0, int(hi/2)), pt2=(wi-1, int(hi/2)), color=(150, 20, 150), thickness=6)
        image = cv2.putText(img=image, text=f"state: {self.state}", org=(wi-200, 30),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6, color=(150, 20, 150), thickness=2)
        for i, temp in enumerate(self.msg):
            image = cv2.putText(img=image, text=temp, org=(50, 300+30*i),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.2, color=(100, 0, 100), thickness=2)

        # Draw Player 1
        image = cv2.putText(img=image, text=f"player 1 score: {self.score_1}", org=(10, 40),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(50, 180, 15), thickness=2)
        p1_cards_names = list()
        for name, _, _ in p1_cards:
            p1_cards_names.append(name)
        image = cv2.putText(img=image, text=f"cards: {' '.join(p1_cards_names)}", org=(10, 80),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=(50, 180, 15), thickness=2)

        # Draw Player 2
        image = cv2.putText(img=image, text=f"player 2 score: {self.score_2}", org=(10, hi-20),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(220, 50, 80), thickness=2)
        p2_cards_names = list()
        for name, _, _ in p2_cards:
            p2_cards_names.append(name)
        image = cv2.putText(img=image, text=f"cards: {' '.join(p2_cards_names)}", org=(10, hi-60),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=(220, 50, 80), thickness=2)

        return image


game = Game()
