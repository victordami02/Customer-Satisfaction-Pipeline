import logging

import pandas as pd
from zenml import step

class IngestData:
    """
    ingesting the data from the data_path

    """
    def __init__(self, data_path:str):
        """
        args:
        data_path: is path to the data
        """
        self.data_path = data_path
        

    def get_data(self):
        """
        ingesting the data from the data_path

        """
        logging.info(f"ingest_data from {self.data_path}")   
        return pd.read_csv(self.data_path)
    
@step
def ingest_df(data_path:str) -> pd.DataFrame:
    """
    ingesting the data from the data_path

    args:
    data_path: is path to the data
    returns:
    pd.DataFrame the data ingested

    """
    try:
        ingest_data = IngestData(data_path)
        df = ingest_data.get_data()
        return df
    except Exception as e:
        logging.error(f"error while ingest_data {e}")
        raise e
    