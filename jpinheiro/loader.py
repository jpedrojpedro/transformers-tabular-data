import sklearn.datasets as datasets
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path


class Loader:
    def __init__(self):
        """
        Initialize loader parameters
        """
        self.model_name = 'distilbert-base-uncased-finetuned-sst-2-english'
        self.df = None

    def join_columns(self, row):
        final = []
        for col in self.df.columns[:-1]:
            aux = [col, str(row[col])]
            final.append(' '.join(aux))
        return ', '.join(final)

    def load_sklearn_dataset(self):
        """
        Load dataset from sklearn
        """
        ds = datasets.load_iris()
        # ds = datasets.load_wine()

        self.df = pd.DataFrame(data=ds.data, columns=ds.feature_names)
        self.df['target'] = ds.target

    def load_local_dataset(self):
        """
        Load dataset from local folder
        """
        dataset_folder = Path(__file__).parent / "datasets"
        ds = pd.read_csv(dataset_folder / "car.csv")
        print(len(ds))

        self.df = pd.DataFrame(data=ds.data, columns=ds.feature_names)
        self.df['target'] = ds.target

    def data_to_text(self):
        """
        Transform tabular data to text, row-wise
        """
        self.load_sklearn_dataset()
        # self.load_local_dataset()
        self.df['text'] = self.df.apply(self.join_columns, axis=1)
        df_text = self.df[['text', 'target']].copy()

        print(df_text['text'][0])
        return df_text

    def split_sets(self):
        """
        Split data into train and test sets
        """
        data = self.data_to_text()
        X_train, X_test, y_train, y_test = train_test_split(
            data['text'],
            data['target'],
            test_size=0.2,
            random_state=42,
            stratify=data['target']
        )
        return X_train, X_test, y_train, y_test

    def load_data(self):
        """
        Load data in final state
        """
        return self.split_sets()
