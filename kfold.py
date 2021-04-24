def kfold(df,label='label',kf=5,):
    totalindex = df.index
    totalsplit = []
    for label_value in df[label].unique():
        subdf = df[df[label]==label_value]
        indexlist = list(subdf.index)
        random.shuffle(indexlist)
        subindex = pd.Index(indexlist)
        indexsplit = []
        size = math.floor(len(subindex)/kf)
        for k in range(kf):
            indexsplit.append(subindex[k*size:(k+1)*size])
        totalsplit.append(indexsplit)
    batchs = []
    for k_ in range(kf):
        batch = pd.Index([])
        for indexes in totalsplit:
            batch = batch.append(indexes[k_])
        batchs.append(batch)
    dfs = []
    for batch in batchs:
        train=df.loc[totalindex.difference(batch)]
        test=df.loc[batch]
        xtrain = train.drop([label],axis=1)
        ytrain = train[[label]]
        xtest = test.drop([label],axis=1)
        ytest = test[[label]]
        dfs.append([xtrain,ytrain,xtest,ytest])
#     for i in dfs:
#         for j in i:
#             print(j.shape)
    return dfs
