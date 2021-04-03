import os
import pandas as pd


def load_data(path):
    if os.path.exists(path):
        try:
            df = pd.read_csv(path, encoding='utf8', header=0, sep=',')
        except BaseException as e:
            print(e, '文件类型错误，pandas无法打开{}\n'.format(path))
        else:
            if 'label' in df.columns:
                return df.drop(['label'], axis=1).values, df['label'].values
            else:
                raise Exception('读入的数据没有label列')
        finally:
            pass
    else:
        raise Exception('传入的文件路径{}不存在'.format(path))
