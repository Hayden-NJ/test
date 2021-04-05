import os
import pandas as pd

# try, except, else, finally
def load_data(path):
    if os.path.exists(path):
        try:
            df = pd.read_csv(path, encoding='utf8', header=0, sep=',')
        except BaseException as e:
            print(e, 'type error, pandas can not open {}\n'.format(path))
        else:
            if 'label' in df.columns:
                return df.drop(['label'], axis=1).values, df['label'].values
            else:
                raise Exception('need label column')
        finally:
            pass
    else:
        raise Exception('file {} not exists'.format(path))
