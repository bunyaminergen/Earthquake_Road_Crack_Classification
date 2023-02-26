import os
import numpy as np
from PIL import Image
from typing import List, Tuple


# np.set_printoptions(threshold=sys.maxsize)

class ImageDataLoader():

    def __init__(self):
        pass

    def load_data(self, path: str, label: int, resize: Tuple[int, int] = (28, 28)) -> List[np.ndarray]:

        """

        Loads images from a directory and returns them as a list of flattened numpy arrays with labels.

        Args:
            path (str): Path to the directory containing the images.
            label (int): Label to assign to the images.
            resize (tuple): Tuple of integers representing the desired width and height of the images.
                            Default is (28, 28).

        Returns:
            A list of flattened numpy arrays with labels.

        Test:

        import pandas as pd
        import matplotlib.pyplot as plt

        dataload = ImageDataLoader()

        data = dataload.load_data(filepath, 0)
        data = pd.DataFrame(data)
        data.to_csv("images.csv", index=False)

        label = data.iloc[0, 1]
        pixels = data.iloc[0, 2:].values.astype(np.uint8)
        img = pixels.reshape((28, 28))

        plt.imshow(img, cmap='gray')
        plt.axis('off')
        plt.show()

        """

        images = []

        # Image file formats

        image_formats = ['jpg', 'jpeg', 'png', 'bmp', 'gif']

        for filename in os.listdir(path):

            if not any(filename.lower().endswith(img_format) for img_format in image_formats):
                continue

            img = Image.open(os.path.join(path, filename)) # load image
            img = img.convert('L')  # convert image to grayscale
            img = img.resize((resize[1], resize[0]))  # resize image
            img = np.array(img).flatten()  # flatten image
            images.append(np.concatenate(([filename, label], img)))

        return images

import pandas as pd
import matplotlib.pyplot as plt

dataload = ImageDataLoader()

neg_data = dataload.load_data(r"file_path", 0)
pos_data = dataload.load_data(r"file_path", 1)

neg_data[0]
pos_data[0]

neg_data_csv = pd.DataFrame(neg_data)
pos_data_csv = pd.DataFrame(pos_data)

neg_data_csv.to_csv("neg_data.csv", index=False)
pos_data_csv.to_csv("pos_data.csv", index=False)

label = pos_data_csv.iloc[0, 1]
pixels = pos_data_csv.iloc[0, 2:].values.astype(np.uint8)
img = pixels.reshape((28, 28))

plt.imshow(img, cmap='gray')
plt.axis('off')
plt.show()

data = pos_data + neg_data

np.random.shuffle(data)

images = np.array([i[2:] for i in data])
labels = np.array([i[1] for i in data])

# images = images.astype(np.uint8) / 255

x_train, x_test = images[:int(len(images) * 0.8)], images[int(len(images) * 0.8):]
y_train, y_test = labels[:int(len(images) * 0.8)], labels[int(len(images) * 0.8):]

len(x_train) + len(x_test)

len(y_train) + len(y_test)

train_data = np.column_stack((x_train, y_train))

pd.DataFrame(train_data).to_csv("train.csv", index=False)

pd.read_csv("train.csv").head()

test_data = np.column_stack((x_test, y_test))

pd.DataFrame(test_data).to_csv("test.csv", index=False)

pd.read_csv("test.csv").head()


