import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict

class DataPreprocessor(ABC):
    def __init__(self, data: pd.DataFrame):
        self.data = data

    @abstractmethod
    def extract_texts(self) -> Dict[str, str]:
        """
        Extract patient IDs and texts from the data and return a dictionary
        where keys are patient IDs and values are texts.

        :return: A dictionary mapping patient IDs to their corresponding texts.
        """
        pass

class DefaultDataPreprocessor(DataPreprocessor):
    def extract_texts(self) -> Dict[str, str]:
        """
        Assumes the input dataframe has columns 'sample_id' and 'text'.
        """
        return dict(zip(self.data['sample_id'], self.data['text']))


class CustomDataPreprocessor(DataPreprocessor):
    def __init__(self, data: pd.DataFrame, sample_id_column: str):
        """
        Initialize the CustomDataPreprocessor, ensuring that the data has a 'sample_id' column
        derived from the column specified by the user.

        :param data: The input dataframe containing patient data.
        :param sample_id_column: The name of the column in the dataframe that should be treated as 'sample_id'.
        """
        super().__init__(data)
        self.sample_id_column = sample_id_column

    def extract_texts(self) -> Dict[str, str]:
        """
        Extract texts using 'clinicalDataTranslated' if available; otherwise, use 'clinicalData'.
        Return a dictionary with 'sample_id' as keys and 'text' as values.
        """
        self.data = self.data.rename(columns={self.sample_id_column: 'sample_id'})
        self.data['text'] = self.data.apply(
            lambda row: row['clinicalDataTranslated'] if pd.notna(row['clinicalDataTranslated']) else row[
                'clinicalData'],
            axis=1
        )
        return dict(zip(self.data['sample_id'], self.data['text']))
