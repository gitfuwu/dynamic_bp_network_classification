import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
train_data_path = 'C:/Users/Administrator/Desktop/BP_classification/train.csv'
validate_data_path = 'C:/Users/Administrator/Desktop/BP_classification/validate.csv'
test_data_path = 'C:/Users/Administrator/Desktop/BP_classification/test.csv'
data_type = 'test'

if data_type == 'train':
    samples = 500
    data = np.zeros([samples, 3])
    step = 0

    while True:
        x = random.uniform(-6, 6)
        y = random.uniform(-6, 6)
        if step == samples:
            break

        if x**2 + y**2 <= 2.5**2:
            data[step, 0] = x
            data[step, 1] = y
            data[step, 2] = 0
            step = step + 1

        if x**2 + y**2 > 2.5**2 and x**2 + y**2 <= 5**2:
            data[step, 0] = x
            data[step, 1] = y
            data[step, 2] = 1
            step = step + 1

    for i in range(samples):
        if data[i, 2] == 0:
            x = data[i, 0]
            y = data[i, 1]
            plt.plot(x, y, 'r*')

        if data[i, 2] == 1:
            x = data[i, 0]
            y = data[i, 1]
            plt.plot(x, y, 'b*')

    plt.show()

    train_data = pd.DataFrame(data[:, :], columns=['x1', 'x2', 'label'])
    train_data.to_csv(train_data_path, index=False)

if data_type == 'validation':
    samples = 500
    data = np.zeros([samples, 3])
    step = 0

    while True:
        x = random.uniform(-6, 6)
        y = random.uniform(-6, 6)
        if step == samples:
            break

        if x ** 2 + y ** 2 <= 2.5 ** 2:
            data[step, 0] = x
            data[step, 1] = y
            data[step, 2] = 0
            step = step + 1

        if x ** 2 + y ** 2 > 2.5 ** 2 and x ** 2 + y ** 2 <= 5 ** 2:
            data[step, 0] = x
            data[step, 1] = y
            data[step, 2] = 1
            step = step + 1

    for i in range(samples):
        if data[i, 2] == 0:
            x = data[i, 0]
            y = data[i, 1]
            plt.plot(x, y, 'r*')

        if data[i, 2] == 1:
            x = data[i, 0]
            y = data[i, 1]
            plt.plot(x, y, 'b*')

    plt.show()

    validate_data = pd.DataFrame(data[:, :], columns=['x1', 'x2', 'label'])
    validate_data.to_csv(validate_data_path, index=False)

if data_type == 'test':
    samples = 500
    data = np.zeros([samples, 2])
    step = 0

    while True:
        x = random.uniform(-6, 6)
        y = random.uniform(-6, 6)
        if step == samples:
            break

        if x ** 2 + y ** 2 <= 2.5 ** 2:
            data[step, 0] = x
            data[step, 1] = y
            step = step + 1

        if x ** 2 + y ** 2 > 2.5 ** 2 and x ** 2 + y ** 2 <= 5 ** 2:
            data[step, 0] = x
            data[step, 1] = y
            step = step + 1


    test_data = pd.DataFrame(data[:, :], columns=['x1', 'x2'])
    test_data.to_csv(test_data_path, index=False)
