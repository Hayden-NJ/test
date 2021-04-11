form collections import Coutner
"""Counter when np.nan here"""
def Counterpp(data):
    datact = []
    for i in data:
        if isinstance(i, str):
            datact.append(i)
        elif np.isnan(i):
            datact.append(np.nan)
        else:
            datact.append(i)
    return Counter(datact)

