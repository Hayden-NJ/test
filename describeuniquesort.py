def describeuniquesort(df, topn=3, tailn=3, columns=None, sorttype='mix'):
    """
    观察dataframe每列数据，去重排序，前后几个展示
    对列不是纯数字的，有mix,float,str三种模式。
    目前看sr.unique可对任何np.nan进行合并
    """
    import numpy as np
    if columns:
        df = df.filter(items=columns)
    else:
        columns = list(df)
    row_index = [f'top{i}' for i in range(1,topn+1)] + [f'tail{i}' for i in range(1,tailn+1)]
    res_df = pd.DataFrame(np.zeros([len(row_index),len(columns)]),columns=columns,index=row_index).applymap(lambda x: np.nan)
    for i in columns:
        col_unique = df[i].unique()
        _ = sum(map(lambda x: not isinstance(x,(float,int)),col_unique))
        if _:
            if sorttype=='mix':
                col_unique = np.array(list(map(str,col_unique)))
            elif sorttype=='float':
                col_unique = list(filter(lambda x: isinstance(x,(float,int)),sr1.values))
            elif sorttype=='str':
                col_unique = list(filter(lambda x: isinstance(x,(str)),sr1.values))
        col_unique.sort()
        col_uni = np.hstack((np.hstack((col_unique[:topn], [np.nan]*(topn-len(col_unique)))),np.hstack(([np.nan]*(tailn-len(col_unique)), col_unique[-tailn:]))))
        res_df[i] = col_uni
    return res_df
