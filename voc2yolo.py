import glob
import os
import xml.etree.ElementTree as ElementTree
from os import getcwd


dirs = ['images-voc-kaggle-original']
# dirs = ['test (copy)']
classes = ['Ah', 'Kh', 'Qh', 'Jh', 'Th', '9h', '8h', '7h', '6h', '5h', '4h', '3h', '2h',
           'Ad', 'Kd', 'Qd', 'Jd', 'Td', '9d', '8d', '7d', '6d', '5d', '4d', '3d', '2d',
           'Ac', 'Kc', 'Qc', 'Jc', 'Tc', '9c', '8c', '7c', '6c', '5c', '4c', '3c', '2c',
           'As', 'Ks', 'Qs', 'Js', 'Ts', '9s', '8s', '7s', '6s', '5s', '4s', '3s', '2s']


def get_images_in_dir(dir_path):
    image_list = []
    for filename in glob.glob(dir_path + '/*.jpg'):
        image_list.append(filename)
    return image_list


def convert(size, box):
    dw = 1./(size[0])
    dh = 1./(size[1])
    x = (box[0] + box[1])/2.0 - 1
    y = (box[2] + box[3])/2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return x, y, w, h


def convert_annotation(dir_path, output_path, image_path):
    basename = os.path.basename(image_path)
    basename_no_ext = os.path.splitext(basename)[0]

    in_file_path = f"{dir_path}/{basename_no_ext}.xml"
    out_file_path = f"{output_path}/{basename_no_ext}.txt"
    # print(f"in_file:                {in_file_path}")
    # print(f"out_file:               {out_file_path}")
    in_file = open(in_file_path)
    out_file = open(out_file_path, 'w')

    tree = ElementTree.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text),
             float(xmlbox.find('xmax').text),
             float(xmlbox.find('ymin').text),
             float(xmlbox.find('ymax').text))
        bb = convert((w, h), b)

        # print(f"    yolo line:              {str(cls_id)} {' '.join([str(a) for a in bb])}")
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

    in_file.close()
    out_file.close()
    os.remove(in_file_path)


def run_convert():
    cwd = getcwd()

    for dir_path in dirs:
        full_dir_path = cwd + '/' + dir_path
        output_path = full_dir_path

        print(f"full_dir_path:          {full_dir_path}")
        print(f"output_path:            {output_path}")

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        image_paths = get_images_in_dir(full_dir_path)

        print(f"image_paths[0]:         {image_paths[0]}")
        print(f"full_dir_path + '.txt': {full_dir_path + '.txt'}")

        list_file = open(full_dir_path + '.txt', 'w')

        for image_path in image_paths:
            list_file.write(image_path + '\n')
            convert_annotation(full_dir_path, output_path, image_path)
        list_file.close()

        print(f"Finished processing: {dir_path}")


if __name__ == '__main__':
    run_convert()
