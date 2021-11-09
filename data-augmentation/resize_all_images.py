import cv2
import sys
from glob import glob


def resize_from_path(img_path, dsize):
    image = cv2.imread(filename=img_path)
    image = cv2.resize(src=image, dsize=dsize)
    cv2.imwrite(filename=img_path, img=image)


images_dir = sys.argv[1]
for i, image_path in enumerate(glob(images_dir + "/*.jpg")):
    if (i+1) % 50 == 0:
        print(i+1)
    resize_from_path(image_path, (416, 416))
