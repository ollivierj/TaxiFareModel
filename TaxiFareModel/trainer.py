import joblib
import mlflow
from memoized_property import memoized_property
from mlflow.tracking import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from xgboost import XGBRegressor

from TaxiFareModel.data import get_data, clean_data
from TaxiFareModel.encoders import DistanceTransformer, TimeFeaturesEncoder, DirectionTransformer, \
    DistanceToCenterTransformer
from TaxiFareModel.utils import compute_rmse

MLFLOW_URI = 'https://mlflow.lewagon.co/'


class Trainer():

    def __init__(self, X, y, model,
                 preprocessors=dict(dist_pipe=True, dist_to_center_pipe=True, dir_pipe=True, time_pipe=True)
                 ):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y
        self.experiment_name = '[FR] [Nantes] [ollivierj] TaxiFareModel 1.0.0'
        self.model = model
        self.preprocessors = preprocessors

    def set_pipeline(self):
        '''returns a pipelined model'''
        preprocs = []
        if (self.preprocessors.get('dist_pipe', False)):
            preprocs.append(
                ('distance', Pipeline([
                    ('dist_trans', DistanceTransformer()),
                    ('stdscaler', StandardScaler())
                ]), ["pickup_latitude", "pickup_longitude", 'dropoff_latitude', 'dropoff_longitude'])
            )

        if (self.preprocessors.get('dist_to_center_pipe', False)):
            preprocs.append(
                ('distance_to_center', Pipeline([
                    ('dist_to_center_trans', DistanceToCenterTransformer()),
                    ('stdscaler', StandardScaler())
                ]), ['pickup_latitude', 'pickup_longitude'])
            )

        if (self.preprocessors.get('dir_pipe', False)):
            preprocs.append(
                ('direction', Pipeline([
                    ('dir_trans', DirectionTransformer()),
                    ('stdscaler', StandardScaler())
                ]), ["pickup_latitude", "pickup_longitude", 'dropoff_latitude', 'dropoff_longitude'])
            )

        if (self.preprocessors.get('time_pipe', False)):
            preprocs.append(
                ('time', Pipeline([
                    ('time_enc', TimeFeaturesEncoder('pickup_datetime')),
                    ('ohe', OneHotEncoder(handle_unknown='ignore'))
                ]), ['pickup_datetime'])
            )

        preproc_pipe = ColumnTransformer(preprocs, remainder = "drop")

        self.pipeline = Pipeline([
            ('preproc', preproc_pipe),
            ('linear_model', self.model)
        ])
        self.mlflow_log_param('features', self.preprocessors)
        self.mlflow_log_param('estimator', type(self.model).__name__)
        return self

    def run(self):
        """set and train the pipeline"""
        self.set_pipeline().pipeline.fit(self.X, self.y)

    def evaluate(self, X_test, y_test):
        '''returns the value of the RMSE'''
        y_pred = self.pipeline.predict(X_test)
        rmse = compute_rmse(y_pred, y_test)
        self.mlflow_log_metric('rmse', rmse)
        return rmse

    @memoized_property
    def mlflow_client(self):
        mlflow.set_tracking_uri(MLFLOW_URI)
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

    def save_model(self):
        """ Save the trained model into a model.joblib file """
        joblib.dump(self.pipeline, 'model.joblib')


if __name__ == "__main__":
    def run(X_train, y_train, model, preprocessors):
        trainer = Trainer(X_train, y_train, model, preprocessors)
        trainer.run()
        # evaluate
        trainer.evaluate(X_val, y_val)
        # save model
        trainer.save_model()

    # clean data
    data = get_data()
    data = clean_data(data)

    # set X and y
    y = data.pop("fare_amount")
    X = data
    X.drop(columns=['key'], inplace=True)

    # hold out
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15)

    # models
    xgboost_model = XGBRegressor(max_depth=10, n_estimators=100, learning_rate=0.1)
    gbr_model = GradientBoostingRegressor(random_state=0)
    ridge_model = Ridge()
    lasso_model = Lasso()
    rfg_model = RandomForestRegressor(max_depth=2, random_state=0)

    print(X.columns)

    for model in [xgboost_model, gbr_model, ridge_model, lasso_model, ridge_model]:
        # train
        run(X_train, y_train, model, dict(dist_pipe=True, dist_to_center_pipe=False, dir_pipe=True, time_pipe=True))
        # best
        run(X_train, y_train, model, dict(dist_pipe=True, dist_to_center_pipe=True, dir_pipe=True, time_pipe=True))
        run(X_train, y_train, model, dict(dist_pipe=True, dist_to_center_pipe=False, dir_pipe=False, time_pipe=True))
        run(X_train, y_train, model, dict(dist_pipe=True, dist_to_center_pipe=True, dir_pipe=False, time_pipe=True))
        run(X_train, y_train, model, dict(dist_pipe=True, dist_to_center_pipe=False, dir_pipe=False, time_pipe=True))
        run(X_train, y_train, model, dict(dist_pipe=True, dist_to_center_pipe=True, dir_pipe=False, time_pipe=False))
        run(X_train, y_train, model, dict(dist_pipe=False, dist_to_center_pipe=False, dir_pipe=False, time_pipe=True))
        run(X_train, y_train, model, dict(dist_pipe=False, dist_to_center_pipe=True, dir_pipe=False, time_pipe=False))


