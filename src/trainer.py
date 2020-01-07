import json

import numpy as np
from data_prep import SentimentData
from keras.callbacks import EarlyStopping, TensorBoard
from keras.models import Model


def train(model: Model, data: SentimentData, name: str):
    cbs = get_callbacks("./logs/" + name)

    metrics = {}

    # print("Training on English")
    # model.fit(
    #     x=data.en_x,
    #     y=data.en_y,
    #     validation_data=(data.cm_x, data.cm_y),
    #     epochs=20,
    #     initial_epoch=0,
    #     shuffle=True,
    #     callbacks=cbs,
    # )
    #
    # metrics["en"] = model.evaluate(data.test_x, data.test_y)
    # print(metrics)
    #
    # print("Training on Spanish")
    # model.fit(
    #     x=data.es_x,
    #     y=data.es_y,
    #     validation_data=(data.cm_x, data.cm_y),
    #     epochs=30,
    #     initial_epoch=20,
    #     shuffle=True,
    #     callbacks=cbs,
    # )
    #
    # metrics["es"] = model.evaluate(data.test_x, data.test_y)
    # print(metrics)
    #
    print("Training on concat")
    model.fit(
        x=np.concatenate([data.en_x, data.es_x]),
        y=np.concatenate([data.en_y, data.es_y]),
        validation_data=(data.cm_x, data.cm_y),
        epochs=100,
        initial_epoch=0,
        shuffle=True,
        callbacks=cbs,
    )

    metrics["both"] = model.evaluate(data.test_x, data.test_y)
    print(metrics)

    print("Training on cm")
    model.fit(
        x=data.cm_x,
        y=data.cm_y,
        validation_split=0.2,
        epochs=70,
        initial_epoch=50,
        shuffle=True,
        callbacks=cbs,
    )
    metrics["cm"] = model.evaluate(data.test_x, data.test_y)
    print(metrics)

    with open("./logs/" + name + "/metric.json", 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=4)

    return model, metrics


def get_callbacks(log_dir):
    tb = TensorBoard(log_dir=log_dir, histogram_freq=5, write_graph=False, write_images=False, update_freq='batch')
    es = EarlyStopping(monitor='val_f1', mode='max', verbose=1, patience=10, restore_best_weights=True)
    return [tb, es]
