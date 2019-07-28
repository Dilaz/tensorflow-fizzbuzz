import tensorflow as tf
import numpy as np


INPUT_SHAPE = (8, )
MAX = 100
EPOCHS = 1000


def encode_fizzbuzz(num):
    if num % 3 == 0 and num % 5 == 0:
        # Fizzbuzz
        return (1, 0, 0, 0)
    elif num % 3 == 0:
        # Fizz
        return (0, 1, 0, 0)
    elif num % 5 == 0:
        # Buzz
        return (0, 0, 1, 0)
    else:
        # Number
        return (0, 0, 0, 1)


def decode_fizzbuzz(result, num):
    return ['FizzBuzz', 'Fizz', 'Buzz', num][result.argmax()]


def get_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, input_shape=INPUT_SHAPE,
                              activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(4, activation='softmax')
    ])
    model.compile(tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='categorical_crossentropy')

    return model


def generate_training_data():
    x = []
    y = []
    for i in range(MAX):
        x.append(num_to_input(i))
        y.append(encode_fizzbuzz(i))

    return np.array(x), np.array(y)


def num_to_input(num):
    return tuple(bin(num)[2:].zfill(8))


def train(model):
    x, y = generate_training_data()
    model.fit(x=x, y=y,
              verbose=2, shuffle=True, epochs=EPOCHS)


def main():
    model = get_model()
    train(model)

    correct = 0
    wrong = 0
    for i in range(1, 101):
        result = model.predict(np.array(num_to_input(i)).reshape(-1, 8))
        output = decode_fizzbuzz(result, i)
        print(i, '-', output)
        if output == decode_fizzbuzz(np.array(encode_fizzbuzz(i)), i):
            correct += 1
        else:
            wrong += 1

    print('Total correct:', correct)
    print('Wrong:', wrong)
    print('Correct percentage:', correct / (correct + wrong) * 100, '%')


if __name__ == "__main__":
    main()
