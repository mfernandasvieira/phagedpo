import os
import pickle
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")


class SplitData:
    def __init__(self, folder, **dataset):
        self.X = None
        self.y = None
        self.__location__ = os.path.realpath(os.path.join(os.getcwd(), folder))
        for key, value in dataset.items():
            with open(os.path.join(self.__location__, str(value)), 'rb') as p:
                self.data = pickle.load(p)

        self.data.to_csv('data.csv')

    def train_test_split(self, stratify=True, test_size=0.3, random_state=42):
        print(self.data)
        self.data = self.data.drop('Annotation', axis=1)
        self.X = self.data.drop('DPO', axis=1)
        self.y = self.data['DPO']

        if stratify:
            X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=test_size, stratify=self.y,
                                                                shuffle=True, random_state=random_state)
        elif not stratify:
            X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=test_size, shuffle=True,
                                                                random_state=random_state)

        return X_train, X_test, y_train, y_test


