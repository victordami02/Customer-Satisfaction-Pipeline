from zenml import pipeline
from steps.ingest_data import ingest_df
from steps.clean_data import clean_data
from steps.evaluation import evaluate_model
from steps.model_train import train_model

@pipeline(enable_cache=True)

def train_pipeline(datapath: str):
    df = ingest_df(datapath)
    x_train, x_test, y_train, y_test = clean_data(df)
    model = train_model(x_train, x_test, y_train, y_test)
    r2_score, rmse =  evaluate_model(model, x_test, y_test)

