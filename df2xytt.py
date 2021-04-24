def df2xytt(df, label='label', columns=None, balance=True, trainpercent=0.5):
    if Counter(df.index).most_common()[0][1]>1:
        print('error: index duplicated')
    
    labelkinds = df[label].unique()
    if len(labelkinds)!=2:
        print('warning: label kinds not equal 2')
        
    dfseries = []
    for i in labelkinds:
        dfseries.append(df[df[label]==i])
    
    if balance == True:
        minimum = np.inf
        for subdf in dfseries:
            if len(subdf)<minimum:
                minimum=len(subdf)
        if trainpercent>1:
            trainsize = trainpercent
        else:
            trainsize = int(minimum*trainpercent)
        testsize = minimum - trainsize
    elif balance == False:
        pass

    indexes = []
    for subdf in dfseries:
        subindex = list(subdf.index)
        random.shuffle(subindex)
        indexes.append(pd.Index(subindex[:trainsize]))
        indexes.append(pd.Index(subindex[trainsize:minimum]))
        indexes.append(pd.Index(subindex[minimum:]))
    
    res = []
    for s in range(3):
        dfcct = []
        for t in range(2):
            dfcct.append(df.loc[indexes[s+3*t]])
        res.append(pd.concat(dfcct,axis=0))
      
    return res
