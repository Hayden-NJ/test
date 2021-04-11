def dida(func):
    def wrapper(*args,**kw):
        t1 = time.time()
        print(f'start time {t1}')
        result = func(*args,**kw)
        t2 = time.time()
        print(f'start time {t2}')
        print(f'{round(t2-t1)} sec')
        return result
    return wrapper

