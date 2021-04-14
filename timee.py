def timee(func):
    def wrapper(*args,**kw):
        t1 = time.time()
        print(f'{time.ctime(t1)}')
        result = func(*args,**kw)
        t2 = time.time()
        print(f'{time.ctime(t2)}')
        print(f'{round(t2-t1)} sec')
        return result
    return wrapper
