def describer(df):
    import pandas as pd
    from collections import Counter
    ct = df.count()
    nan = df.isnull().sum()
    other = df.shape[0] - ct - nan
    maxima = df.max()
    minima = df.min()
    unique = df.apply(lambda x: len(set(x)))
    mode1 = df.apply(lambda x: Counter(x).most_common(1)[0][1])
    types = {}
    for i in list(df):
        try:
            df[i].map(float)
        except:
            types[i] = 'string'
        else:
            types[i] = 'float'
    types = pd.Series(types)
    columns = ['ct', 'nan', 'other', 'maxima', 'minima', 'unique', 'mode1', 'types', ]
    res = {}
    for i in columns:
        res[i] = eval(i)
    result = pd.DataFrame(res)
    return result
    