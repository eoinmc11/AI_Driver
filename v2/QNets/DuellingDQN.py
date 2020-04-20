import keras.backend as k
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Input, Add, Subtract, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

LEARNING_RATE = 0.005


class DuellingDQN:
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.model = self.create_model()
        self.learning_rate = 0.005

    def create_model(self):
        # TODO: Add LSTM and maybe dropout
        inp = Input(self.input_shape)
        out = Conv2D(data_format='channels_last',
                     filters=256,
                     kernel_size=(3, 3),
                     strides=(2, 2),
                     activation='relu')(inp)
        out = MaxPooling2D()(out)
        out = Conv2D(data_format='channels_last',
                     filters=256,
                     kernel_size=(3, 3),
                     strides=(2, 2),
                     activation='relu')(out)
        out = MaxPooling2D()(out)
        out = Flatten()(out)
        value = Dense(units=64,
                      kernel_initializer='zeros',
                      activation='relu')(out)
        value = Dense(units=1,
                      kernel_initializer='zeros',
                      activation='relu')(value)
        advantage = Dense(units=64,
                          kernel_initializer='zeros',
                          activation='relu')(out)
        advantage = Dense(units=self.output_shape,
                          kernel_initializer='zeros',
                          activation='softmax')(advantage)
        advantage_mean = Lambda(lambda x: k.mean(x, axis=1))(advantage)
        advantage = Subtract()([advantage, advantage_mean])
        out = Add()([value, advantage])

        model = Model(inputs=inp, outputs=out)
        model.compile(loss="mean_squared_error",
                      optimizer=Adam(lr=LEARNING_RATE))
        return model
