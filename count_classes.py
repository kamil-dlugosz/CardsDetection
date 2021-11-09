import os
import json
from tqdm import tqdm


def get_classes_names():
    classes_names = dict()
    with open('data/obj.names', 'r') as f:
        for idx, line in enumerate(f):
            classes_names[idx] = line.strip()
    # pretty_print_dict(classes_names)
    return classes_names


def pretty_print_dict(dictionary: dict):
    print(json.dumps(dictionary, sort_keys=True, indent=4))


def count_classes(names_file_path):
    counter = dict()
    classes_names = get_classes_names()
    counter['opened_files'] = 0
    with open(names_file_path, 'r') as names_file:
        for name in tqdm(names_file):
            name.strip()
            name = os.path.splitext(name)[0] + '.txt'
            name = name.replace('\\', '/')
            counter['opened_files'] += 1
            with open(name, 'r') as f:
                for line in f:
                    class_id = line.split(sep=' ')[0]
                    class_name = classes_names[int(class_id)]
                    if class_name in counter:
                        counter[class_name] += 1
                    else:
                        counter[class_name] = 1
    pretty_print_dict(counter)


if __name__ == '__main__':
    count_classes('data/train.txt')
    count_classes('data/valid.txt')
