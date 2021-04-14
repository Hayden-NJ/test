def describegene(df):
    """nan contains none"""
    import time
    import pandas as pd
    from collections import Counter
    t0 = time.time()

    res0 = {}
    ct = df.count()
    nan = df.isnull().sum()
    other = df.shape[0] - ct - nan
    maxima = df.max()
    minima = df.min()
    types = {}
    for i in list(df):
        try:
            df[i].map(float)
        except:
            types[i] = 'string'
        else:
            types[i] = 'float'
    types = pd.Series(types)
    for name in ['ct', 'nan', 'other', 'maxima', 'minima', 'types' ]:
        res0[name] = eval(name)
    t1 = time.time()
    print(f'get res0 done, wall time {round(t1-t0)} sec')
    
    res1 = {}
    items = {'emptystr':'','none':None,'zero':0,}
    for item in items:
        res1[item] = df.applymap(lambda x:1 if x==items[item] else 0).sum()
    t2 = time.time()
    print(f'get res0 done, wall time {round(t2-t1)} sec')
    
    res2 = {}
    res2_ = []
    for col in list(df):
        count = df[col].map(str).value_counts().sort_values(ascending=False)
        if len(count)>=2:
            res2_.append([count.shape[0],count.iloc[0],count.index[0],count.iloc[1],count.index[1]])
        else:
            res2_.append([count.shape[0],count.iloc[0],count.index[0],np.nan,np.nan])
    res2_ = list(zip(*res2_))
    for k,v in enumerate(['unique','mode1','mod1','mode2','mod2',]):
        res2[v] = pd.Series(res2_[k],index=list(df))
    t3 = time.time()
    print(f'get res0 done, wall time {round(t3-t2)} sec')
    
    res = {}
    res.update(res0)
    res.update(res1)
    res.update(res2)
    result = pd.DataFrame(res)
    reindex = ['emptystr','none','zero','types','nan','other','ct','unique','mode1','mod1','mode2','mod2','maxima','minima',]
    reindex.extend(set(list(result)) - set(reindex))
    result = result.reindex(columns=reindex)
    return result

