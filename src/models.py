from keras.layers import Bidirectional, LSTM, Dense, Input
from keras.models import Sequential
from utills import f1


def get_seq_model():
    model = Sequential()
    model.add(Bidirectional(
        LSTM(
            units=50,
            dropout=0.3,
            recurrent_dropout=0.3,
            # return_sequences=True,
        )
    ))
    # model.add(AttentionWithContext(name="attention"))
    model.add(Dense(3, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy', f1])

    return model


def get_dense_model():
    model = Sequential()

    model.add(Dense(
        units=3,
        activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy', f1])

    return model


ft = Input(shape=(None, 32, 100))
multi = Input(shape=(None, 32, 300))
laser = Input(shape=(None, 1024))

ft = Bidirectional(
    LSTM(
        units=50,
        dropout=0.3,
        recurrent_dropout=0.3,
        # return_sequences=True,
    )(ft)
)

multi = Bidirectional(
    LSTM(
        units=50,
        dropout=0.3,
        recurrent_dropout=0.3,
        # return_sequences=True,
    )(multi)
)


