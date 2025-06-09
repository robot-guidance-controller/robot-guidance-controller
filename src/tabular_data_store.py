import pandas as pd
import pickle

class TabularDataStore:
    def __init__(self, columns = None, column_names = None):
        if columns is None and column_names is None:
            raise ValueError('Must specify at least number of columns or column names.')
        
        if column_names is None:
            column_names = []

        duplicate_column_names = [name for name in set(column_names) if column_names.count(name) > 1]
        if duplicate_column_names:
            raise ValueError(f'Duplicate column names found: {", ".join(duplicate_column_names)}.')

        if columns is not None:
            for i in range(columns):
                if i >= len(column_names):
                    column_names.append(f'column_{i}')

            column_names = column_names[:columns]

        self.column_names = column_names

        self._table = {column_name:[] for column_name in column_names}

    def append_row(self, row_data):
        if len(row_data) != len(self.column_names):
            raise ValueError("Row data does not match number of columns.")
        
        for column_name, column_value in zip(self.column_names, row_data):
            self._table[column_name].append(column_value)

    def to_pandas(self):
        return pd.DataFrame(self._table, columns=self.column_names)
    
    def to_pickle(self, filepath: str):
        data = {
            'column_names': self.column_names,
            'table': self._table,
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)

    @classmethod
    def from_pickle(cls, filepath: str):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        instance = cls(column_names=data['column_names'])
        instance._table = data['table']
        return instance