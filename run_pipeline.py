from pipelines.training_pipeline import train_pipeline
from zenml.client import Client
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri

if __name__ == '__main__':
    print(Client().active_stack.experiment_tracker.get_tracking_uri())
    train_pipeline(datapath="/Users/mac/Documents/customer_satisfaction/data/olist_customers_dataset.csv")

#mlflow ui --backend-store-uri "file:/Users/mac/Library/Application Support/zenml/local_stores/86e9b2a5-f6e6-45b5-9ba5-63a0f13c2449/mlruns"
