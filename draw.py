def draw(df,columns):
    n = len(columns)
    fig,axes = plt.subplots(n,1,sharex=False,sharey=False,figsize=(16,n*4))
    for i in range(n):
        sns.histplot(data=df,x=columns[i],ax=axes[i])
        