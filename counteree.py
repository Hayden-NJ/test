from collections import Counter
"""Counter when np.nan here"""
def counteree(data):
    datact = []
    for i in data:
        if isinstance(i, str):
            datact.append(i)
        elif np.isnan(i):
            datact.append(np.nan)
        else:
            datact.append(i)
    return Counter(datact)

