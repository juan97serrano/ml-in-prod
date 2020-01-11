"""Main module fr training."""
from .model import build_model
from tensorflow.keras.datasets import mnist
from tensorflow.keras import losses,metrics,optimizers
from tensorflow.keras.utils import to_categorical


import argparse
import logging.config

LOGGER = logging.getLogger()

def _transform_x(x):
    return x / 255.0

def _transform_y(y):
    return to_categorical(y)

def train_and_evaluate(epochs,batch_size):
    """Train and evaluate the model."""
    model = build_model()


    # Download data
    train_data, test_data = mnist.load_data()
    x_train, y_train = train_data
    x_test,y_test = test_data

    # Transform data
    x_train_transf = _transform_x(x_train)
    x_test_transf = _transform_x(x_test)
    y_train_transf = _transform_y(y_train)
    y_test_transf = _transform_y(y_test)

    model.compile(optimizer=optimizers.Adam(),
                  loss=losses.categorical_crossentropy,
                  metrics = [metrics.categorical_accuracy])
    model.fit(x_train_transf,
              y_train_transf,
              epochs=epochs,
              batch_size=batch_size)
    myloss, myacc = model.evaluate(x_test_transf,y_train_transf)
    LOGGER.info('Test loss: %.4f' % myacc)
    LOGGER.info('Test accuracy: %.4f' % myacc)


if '__main__' == __name__:
    # TODO: Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs',type=int,required=True)
    parser.add_argument('--batch_size', type=int, required=True)

    args = parser.parse_args()
    epochs = args.epochs
    batch_size = args.batch_size

    train_and_evaluate(epochs,batch_size)

