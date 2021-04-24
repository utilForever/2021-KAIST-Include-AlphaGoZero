from matplotlib import pyplot as plt
import numpy as np
from dlgo.nn.load_mnist import load_data
from dlgo.nn.layers import sigmoid_double


def average_digit(data, digit):
    filtered_data = [x[0] for x in data if np.argmax(x[1]) == digit]
    filtered_array = np.asarray(filtered_data)
    return np.average(filtered_array, axis=0)


train, test = load_data()
avg_eight = average_digit(train, 8)


img = (np.reshape(avg_eight, (28, 28)))
plt.imshow(img)
plt.show()

x_3 = train[2][0]
x_18 = train[17][0]

W = np.transpose(avg_eight)
np.dot(W, x_3)
np.dot(W, x_18)


def predict(x, W, b):
    return sigmoid_double(np.dot(W, x) + b)


b = -45

print(predict(x_3, W, b))
print(predict(x_18, W, b))


def evaluate(data, digit, threshold, W, b):
    total_samples = 1.0 * len(data)
    correct_predictions = 0
    for x in data:
        if predict(x[0], W, b) > threshold and np.argmax(x[1]) == digit:
            correct_predictions += 1
        if predict(x[0], W, b) <= threshold and np.argmax(x[1]) != digit:
            correct_predictions += 1
    return correct_predictions / total_samples


evaluate(data=train, digit=8, threshold=0.5, W=W, b=b)

evaluate(data=test, digit=8, threshold=0.5, W=W, b=b)

eight_test = [x for x in test if np.argmax(x[1]) == 8]
evaluate(data=eight_test, digit=8, threshold=0.5, W=W, b=b)
