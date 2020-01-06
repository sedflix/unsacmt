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


def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.
        Only computes a batch-wise average of recall.
        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.
        Only computes a batch-wise average of precision.
        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


def get_class_weight(y):
    """
    Used from: https://stackoverflow.com/a/50695814
    TODO: check validity and 'balanced' option
    :param y: A list of one-hot-encoding labels [[0,0,1,0],[0,0,0,1],..]
    :return: class-weights to be used by keras model.fit(.. class_weight="") -> {0:0.52134, 1:1.adas..}
    """
    y_integers = np.argmax(y, axis=1)
    class_weights = compute_class_weight('balanced', np.unique(y_integers), y_integers)
    d_class_weights = dict(enumerate(class_weights))
    return d_class_weights


from tensorflow.keras import backend as K


def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.
        Only computes a batch-wise average of recall.
        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.
        Only computes a batch-wise average of precision.
        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


from sklearn.utils.class_weight import compute_class_weight


def get_class_weight(y):
    """
    Used from: https://stackoverflow.com/a/50695814
    TODO: check validity and 'balanced' option
    :param y: A list of one-hot-encoding labels [[0,0,1,0],[0,0,0,1],..]
    :return: class-weights to be used by keras model.fit(.. class_weight="") -> {0:0.52134, 1:1.adas..}
    """
    y_integers = np.argmax(y, axis=1)
    class_weights = compute_class_weight('balanced', np.unique(y_integers), y_integers)
    d_class_weights = dict(enumerate(class_weights))
    return d_class_weights


from keras import backend as K
from keras import losses


def loss_ordinal(y_true, y_pred):
    weights = K.cast(K.abs(K.argmax(y_true, axis=1) - K.argmax(y_pred, axis=1)) / (K.int_shape(y_pred)[1] - 1),
                     dtype='float32')
    return (1.0 + weights) * losses.categorical_crossentropy(y_true, y_pred)
