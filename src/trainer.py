import numpy as np
from keras.callbacks import EarlyStopping, TensorBoard
from keras.models import Model

from data_prep import SentimentData


def train(model: Model, data: SentimentData, name: str):
    cbs = get_callbacks("./logs/" + name)

    metrics = {}

    print("Training on English")
    model.fit(
        x=data.en_x,
        y=data.en_y,
        validation_data=(data.cm_x, data.cm_y),
        epochs=30,
        initial_epoch=0,
        shuffle=True,
        callbacks=cbs,
    )

    metrics["en"] = model.evaluate(data.test_x, data.test_y)
    print(metrics)

    print("Training on Spanish")
    model.fit(
        x=data.es_x,
        y=data.es_y,
        validation_data=(data.cm_x, data.cm_y),
        epochs=60,
        initial_epoch=30,
        shuffle=True,
        callbacks=cbs,
    )

    metrics["es"] = model.evaluate(data.test_x, data.test_y)
    print(metrics)

    print("Training on concat")
    model.fit(
        x=np.concatenate([data.en_x, data.es_x]),
        y=np.concatenate([data.en_y, data.es_y]),
        validation_data=(data.cm_x, data.cm_y),
        epochs=90,
        initial_epoch=60,
        shuffle=True,
        callbacks=cbs,
    )

    metrics["both"] = model.evaluate(data.test_x, data.test_y)
    print(metrics)

    print("Training on cm")
    model.fit(
        x=data.cm_x,
        y=data.cm_y,
        validation_data=(data.cm_x, data.cm_y),
        epochs=120,
        initial_epoch=90,
        shuffle=True,
        callbacks=cbs,
    )
    metrics["cm"] = model.evaluate(data.test_x, data.test_y)
    print(metrics)

    return model, metrics


def get_callbacks(log_dir):
    tb = TensorBoard(log_dir=log_dir, histogram_freq=5, write_graph=False, write_images=False, update_freq='batch')
    es = EarlyStopping(monitor='val_f1', mode='max', verbose=1, patience=5, restore_best_weights=True)
    return [tb, es]
