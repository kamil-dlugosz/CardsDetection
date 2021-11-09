from config import *
from tqdm import tqdm
import cv2
import os


class Card:
    def __init__(self, filename, source_dir=''):
        # if not filename:
        #     raise ValueError('argument filename is not valid')
        # if not source_dir:
        #     raise ValueError('argument source_dir is not valid')
        self.filename = filename
        self.source_dir = source_dir
        self.full_path = os.path.join(self.source_dir, self.filename)
        image = cv2.imread(self.full_path)
        if image is None or len(image.shape) != 3:
            raise ValueError(f"Card.__init__(): Error when loading image \"{filename}\" from \"{source_dir}\"")
        self.image = image
        self.rank, self.light,  self.deck, self.number, self.extention = Card.__decode_filename(self.filename)

    def __str__(self):
        return (f"  Card info:        Image info:\n"
                f"    - rank:   {self.rank}      - filename:  {self.filename}\n"
                f"    - light:  {self.light}      - source_dir: {self.source_dir}\n"
                f"    - deck:   {self.deck}      - full_path:  {self.full_path}\n"
                f"    - number: {self.number}      - extention:  {self.extention}")

    @staticmethod
    def __decode_filename(filename):
        rank = filename[0:2]
        light = filename[3:5]
        deck = filename[6:8]
        number = filename[9:11]
        _, extention = os.path.splitext(filename)
        return rank, light, deck, number, extention

    @staticmethod
    def distribution(cards, verbose=1):
        distr = dict(ranks=dict(), lights=dict(), decks=dict(), numbers=dict(), extentions=dict())
        for card in cards:
            if verbose >= 2:
                print(card)
            try:
                distr['ranks'][card.rank] += 1
            except KeyError:
                distr['ranks'][card.rank] = 1
            try:
                distr['lights'][card.light] += 1
            except KeyError:
                distr['lights'][card.light] = 1
            try:
                distr['decks'][card.deck] += 1
            except KeyError:
                distr['decks'][card.deck] = 1
            try:
                distr['numbers'][card.number] += 1
            except KeyError:
                distr['numbers'][card.number] = 1
            try:
                distr['extentions'][card.extention] += 1
            except KeyError:
                distr['extentions'][card.extention] = 1
        if verbose >= 1:
            print(f"  Cards distribution:\n"
                  f"    - ranks:      {distr['ranks']}\n"
                  f"    - lights:     {distr['lights']}\n"
                  f"    - decks:      {distr['decks']}\n"
                  f"    - numbers:    {distr['numbers']}\n"
                  f"    - extentions: {distr['extentions']}")
        return distr

    @staticmethod
    def read_cards(source_dir, target_ranks=None, target_lights=None, target_decks=None,
                   target_numbers=None, target_extentions=None, verbose=1):

        # todo extract this and make filter function ?
        filenames = list()
        for filename in os.listdir(source_dir):
            rank, light, deck, number, extention = Card.__decode_filename(filename)
            if (target_ranks is None or rank in target_ranks) and \
               (target_lights is None or light in target_lights) and \
               (target_decks is None or deck in target_decks) and \
               (target_numbers is None or number in target_numbers) and \
               (target_extentions is None or extention in target_extentions):
                filenames.append(filename)

        if verbose >= 1:
            print(f"Card.read_cards(): Reading {len(filenames)} cards from \"{source_dir}\"")

        cards = list()
        for filename in tqdm(filenames):
            new_card = Card(filename=filename, source_dir=source_dir)
            cards.append(new_card)
            if verbose >= 2:
                print(f"Card.read_cards(): Read cards {len(cards)}/{len(filenames)}: \"{filename}\"")

        if verbose >= 1:
            print(f"Card.read_cards(): Read {len(cards)} cards from \"{source_dir}\"")
        return cards

    @staticmethod
    def inspect_cards(cards, height=800, verbose=1):

        if verbose >= 1:
            print(f"Card.inspect_cards(): Inspecting {len(cards)} cards")

        for idx, card in enumerate(cards):
            if verbose >= 2:
                print(f"Card.inspect_cards(): Displaying card \"{card.full_path}\"")
            win_name = f"Card.inspect_cards() - {idx+1}/{len(cards)}"
            # cv2.namedWindow(win_name, flags=cv2.WINDOW_GUI_NORMAL)
            while True:
                width = int(card.bgr.shape[1] / card.bgr.shape[0] * height)
                card_preview = cv2.resize(src=card.bgr, dsize=(width, height))
                cv2.imshow(winname=win_name, mat=card_preview)
                key = cv2.waitKey(50)

                if key == ord('n') or cv2.getWindowProperty(winname=win_name, prop_id=cv2.WND_PROP_VISIBLE) == 0:
                    break
                if key == ord('q'):
                    if verbose >= 1:
                        print(f"Card.inspect_cards(): Inspected {idx+1}/{len(cards)} cards - cancelled by user")
                    return
                elif key == ord('i'):
                    print(card)
                elif key == ord('r'):
                    card.bgr = cv2.rotate(src=card.bgr, rotateCode=cv2.ROTATE_90_CLOCKWISE)
                elif key == ord('s'):
                    if cv2.imwrite(filename=card.full_path, img=card.bgr):
                        if verbose >= 1:
                            print(f"Card.inspect_cards(): Image successfully saved at \"{card.full_path}\"")
                        card_preview = cv2.putText(img=card_preview, text='SAVED', org=(30, 40),
                                                   fontFace=cv2.FONT_ITALIC, fontScale=height//600,
                                                   color=(50, 50, 255), thickness=2)
                        cv2.imshow(winname=win_name, mat=card_preview)
                        cv2.waitKey(500)
                    else:
                        if verbose >= 1:
                            print(f"Card.inspect_cards(): Image NOT saved at \"{card.full_path}\"")
                        card_preview = cv2.putText(img=card_preview, text='NOT SAVED', org=(30, 40),
                                                   fontFace=cv2.FONT_ITALIC, fontScale=height/600,
                                                   color=(50, 50, 255), thickness=2)
                        cv2.imshow(winname=win_name, mat=card_preview)
                        cv2.waitKey(500)
            cv2.destroyWindow(win_name)

        if verbose >= 1:
            print(f"Card.inspect_cards(): Inspected {len(cards)} cards")


if __name__ == '__main__':
    cards_list = Card.read_cards(source_dir=CARD_PICTURES_DIR, target_lights=['01', '02'], target_ranks='26')
    Card.distribution(cards_list)
    Card.inspect_cards(cards_list)
