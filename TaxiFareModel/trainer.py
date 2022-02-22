# imports
from tkinter.tix import COLUMN
from TaxiFareModel.data import get_data, clean_data
from TaxiFareModel.encoders import DistanceTransformer, TimeFeaturesEncoder
from TaxiFareModel.utils import compute_rmse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
import pandas as pd

class Trainer():
    def __init__(self, X, y, **kwargs):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.y = y
        self.X = X
        self.test_size = kwargs.get('test_size', 0.1)
        self.X_train = ()
        self.X_test = ()
        self.y_train = ()
        self.y_test = ()
        self.hold_out_verif = 0

    def hold_out(self, X, y):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=self.test_size)
        self.hold_out_verif = 1
        return self.X_train, self.X_test, self.y_train, self.y_test 

    def set_pipeline(self):
        '''returns a pipelined model'''
        dist_pipe = Pipeline([
            ('dist_trans', DistanceTransformer()),
            ('stdscaler', StandardScaler())
        ])
        time_pipe = Pipeline([
            ('time_enc', TimeFeaturesEncoder('pickup_datetime')),
            ('ohe', OneHotEncoder(handle_unknown='ignore'))
        ])
        preproc_pipe = ColumnTransformer([
            ('distance', dist_pipe, ["pickup_latitude", "pickup_longitude", 'dropoff_latitude', 'dropoff_longitude']),
            ('time', time_pipe, ['pickup_datetime'])
        ], remainder="drop")
        self.pipeline = Pipeline([
            ('preproc', preproc_pipe),
            ('linear_model', LinearRegression())
        ])
        return self.pipeline

    def run(self):
        '''returns a trained pipelined model'''
        if self.hold_out_verif == 0:
            self.pipeline = self.set_pipeline().fit(self.X, self.y)
        else:
            self.pipeline = self.set_pipeline().fit(self.X_train, self.y_train)


    def evaluate(self, X_test, y_test):
        '''returns the value of the RMSE'''
        y_pred = self.pipeline.predict(X_test)
        rmse = compute_rmse(y_pred, y_test)
        return rmse

if __name__ == "__main__":
    # get data & clean data
    df = get_data()
    df = clean_data(df)
    # set X and y
    y = df.pop('fare_amount')
    X = df
    trainer = Trainer(X,y, test_size=0.1)
    # hold out
    X_train, X_test, y_train, y_test = trainer.hold_out(X, y)
    # train
    trainer.run()
    # evaluate
    res = trainer.evaluate(X_test, y_test)
    print(res)
