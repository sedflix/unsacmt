import pickle

from keras.utils import to_categorical



def read_data(train_path: str, test_path: str):
    """

    Args:
        train_path: path to the train dataset
        test_path:  path to the test datset

    Returns: en_x, en_y, es_x, es_y, cm_x, cm_y, test_x, test_y

    """
    en_x, en_y, es_x, es_y, cm_x, cm_y = [], [], [], [], [], []
    with open(train_path, encoding="utf8") as f:
        all_lines = f.readlines()
    false_counter = 0

    for line in all_lines:
        line = line.strip().split("\t")
        text = line[0]
        try:
            label = line[1]
        except:
            false_counter = false_counter + 1
            continue
        if line[2] == "en":
            en_x.append(text)
            en_y.append(label)
        if line[2] == "es":
            es_x.append(text)
            es_y.append(label)
        if line[2] == "cm":
            cm_x.append(text)
            cm_y.append(label)

    with open(test_path, encoding="utf8") as f:
        all_lines = f.readlines()

    test_x, test_y = [], []
    for line in all_lines:
        line = line.strip().split("\t")
        text = line[0]
        try:
            label = line[1]
        except:
            false_counter = false_counter + 1
            continue
        test_x.append(text)
        test_y.append(label)

    return en_x, to_categorical(en_y, num_classes=3), \
           es_x, to_categorical(es_y, num_classes=3), \
           cm_x, to_categorical(cm_y, num_classes=3), \
           test_x, to_categorical(test_y, num_classes=3)
