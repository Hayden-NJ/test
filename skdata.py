import pandas as pd
def skdata(name):
    if name=='boston':
        from sklearn.datasets import load_boston as loading
    elif name=='cancer':
        from sklearn.datasets import load_breast_cancer as loading
    elif name=='iris':
        from sklearn.datasets import load_iris as loading
    elif name=='wine':
        from sklearn.datasets import load_wine as loading
    else:
        print('choice: boston, cancer, iris, wine')
    data = loading(return_X_y=False)
    df = pd.DataFrame(data['data'], columns=data['feature_names'])
    df['label'] = data['target']
    return df