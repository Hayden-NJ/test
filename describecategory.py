def describecategory(df,topn=3, tailn=3, columns=None, ):
    if columns:
        df = df.filter(items=columns)
    else:
        columns = list(df)
    res_columns = []
    for i in columns:
        res_columns.append(i)
        res_columns.append(f'{i}_')
    res_index = ['count','unique'] + [f'top{i}' for i in range(1,topn+1)] + [f'tail{i}' for i in range(1,tailn+1)]
    res_df = pd.DataFrame(np.zeros([len(res_index),len(res_columns)]),columns=res_columns,index=res_index).applymap(lambda x: np.nan)
    for i in columns:
        counts_sorted = df[i].value_counts().sort_values(ascending=False)
        counts_df = pd.DataFrame(pd.concat([counts_sorted[:topn] , counts_sorted[-tailn:]],axis=0)).reset_index()
        counts_df.columns = [i,f'{i}_']
        counts_df.index = [f'top{i}' for i in range(1,counts_sorted[:topn].count()+1)] + [f'tail{i}' for i in range(1,counts_sorted[-tailn:].count()+1)]
        for loc in [(i,j) for i in counts_df.index.values.tolist() for j in counts_df.columns.values.tolist()]:
            res_df.loc[loc[0],loc[1]] = counts_df.loc[loc[0],loc[1]]
        res_df.loc['count',i] = df[i].count()
        res_df.loc['unique',i] = df[i].unique().shape[0]
    return res_df
    