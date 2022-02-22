# imports
from re import L
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
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from memoized_property import memoized_property
import mlflow
from  mlflow.tracking import MlflowClient
import joblib

MLFLOW_URI = "https://mlflow.lewagon.co/"

class Trainer():

    def __init__(self, X, y, **kwargs):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.MLFLOW_URI = "https://mlflow.lewagon.co/"
        self.pipeline = None
        self.y = y
        self.X = X
        self.test_size = kwargs.get('test_size', 0.1)
        self.X_train = ()
        self.X_test = ()
        self.y_train = ()
        self.y_test = ()
        self.hold_out_verif = 0
        self.experiment_name = "[FR] [Nantes] [AlbanKv] TaxiFareModel v1.0"

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
            ('model', LinearRegression()) #'LinearRegression', LinearRegression()
        ])
        return self.pipeline

    def run(self):
        '''returns a trained pipelined model'''
        if self.hold_out_verif == 0:
            self.pipeline = self.set_pipeline().fit(self.X, self.y)
        else:
            self.pipeline = self.set_pipeline().fit(self.X_train, self.y_train)
        experiment_id = trainer.mlflow_experiment_id
        print(f"experiment URL: https://mlflow.lewagon.co/#/experiments/{experiment_id}")

    def evaluate(self, X_test, y_test):
        '''returns the value of the RMSE'''
        y_pred = self.pipeline.predict(X_test)
        rmse = compute_rmse(y_pred, y_test)
        return rmse

    def save_model(self):
        """ Save the trained model into a model.joblib file """
        joblib.dump(self.pipeline, 'model.joblib')
        self.mlflow_log_param('metric', LinearRegression())
        self.mlflow_log_metric('rmse', rmse)
        pass

    @memoized_property
    def mlflow_client(self):
        mlflow.set_tracking_uri(self.MLFLOW_URI)
        return MlflowClient()

    @memoized_property
    def mlflow_experiment_id(self):
        try:
            return self.mlflow_client.create_experiment(self.experiment_name)
        except BaseException:
            return self.mlflow_client.get_experiment_by_name(self.experiment_name).experiment_id

    @memoized_property
    def mlflow_run(self):
        return self.mlflow_client.create_run(self.mlflow_experiment_id)

    def mlflow_log_param(self, key, value):
        self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

    def mlflow_log_metric(self, key, value):
        self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)


if __name__ == "__main__":
    # get data & clean data
    # for model in [LinearRegression(), RandomForestRegressor()]:
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
    # save as .joblib
    trainer.save_model()
    print(res)
