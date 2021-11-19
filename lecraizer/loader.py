import sklearn.datasets as datasets
from sklearn.model_selection import train_test_split
import pandas as pd


class Loader:
    def __init__(self):
        ''' 
        Initialize loader parameters
        '''
        self.model_name = 'distilbert-base-uncased-finetuned-sst-2-english'
        

    def join_columns(self, row):
        final = []
        for col in self.df.columns[:-1]:
            aux = []
            aux.append(col)
            aux.append(str(row[col]))
            final.append(' '.join(aux))
        return ', '.join(final)
    
    
    def load_sklearn_dataset(self):
        '''
        Load dataset from sklearn
        '''
        self.dataset = datasets.load_iris()
#         self.dataset = datasets.load_wine()
        
        df = pd.DataFrame(data=self.dataset.data, columns=self.dataset.feature_names)
        df['target'] = self.dataset.target
        return df
    
    
    def load_local_dataset(self):
        '''
        Load dataset from local folder
        '''
        self.dataset = pd.read_csv('../text/datasets/car.csv')
        print(len(self.dataset))
        
        
        df = pd.DataFrame(data=self.dataset.data, columns=self.dataset.feature_names)
        df['target'] = self.dataset.target
        return df


    
    def data_to_text(self):
        '''
        Transform tabular data to text, row-wise
        '''
        self.df = self.load_sklearn_dataset()
#         self.df = self.load_local_dataset()
        self.df['text'] = self.df.apply(self.join_columns, axis=1)
        df_text = self.df[['text', 'target']].copy()
        
        print(df_text['text'][0])
        return df_text

    
    def split_sets(self):
        ''' 
        Split data into train and test sets
        '''
        data = self.data_to_text()
        X_train, X_test, y_train, y_test = train_test_split(data['text'], data['target'],
                                                    test_size=0.2,
                                                    random_state=42,
                                                    stratify=data['target'])
        return X_train, X_test, y_train, y_test

    
    def load_data(self):
        '''
        Load data in final state
        '''
        return self.split_sets()