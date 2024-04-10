import logging
from abc import ABC, abstractmethod
from sklearn.linear_model import LinearRegression

class Model(ABC):
    @abstractmethod
    def train(self, x_train, y_train):
        pass


class LinearRegressionModel(Model):
    def train(self, x_train, y_train, **kwargs):

        try:
            rag = LinearRegression(**kwargs)
            rag.fit(x_train, y_train)
            logging.info("moddel training complete")
            return rag
        except Exception as e:
            logging.error("error in training model: {}".format(e))
            raise e
