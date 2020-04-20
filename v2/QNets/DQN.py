from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

LEARNING_RATE = 0.005


class DQN:
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.model = self.create_model()
        self.learning_rate = 0.005

    def create_model(self):
        model = Sequential()
        model.add(Conv2D(input_shape=self.input_shape,
                         data_format='channels_last',
                         filters=256,
                         kernel_size=(3, 3),
                         strides=(2, 2),
                         activation='relu'))
        model.add(MaxPooling2D(2, 2))
        model.add(Conv2D(data_format='channels_last',
                         filters=256,
                         kernel_size=(3, 3),
                         strides=(2, 2),
                         activation='relu'))
        model.add(MaxPooling2D(2, 2))
        model.add(Dropout(0.2))
        model.add(Flatten())
        model.add(Dense(units=64, kernel_initializer='zeros', activation='relu'))
        model.add(Dense(units=self.output_shape, activation='softmax'))
        model.compile(loss="mean_squared_error", optimizer=Adam(lr=LEARNING_RATE))
        return model
