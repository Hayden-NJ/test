class tick():
    def __init__(self):
        self.num = 0
    def plain(self,i=None,step=1, total=None):
        import time
        if not i:
            i=self.num
        if not total:
            total = '?'
        if self.num%step==0:
            print(f'{time.ctime(time.time())}: {i}/{total}')
        self.num+=1
    def timebins(self):
        pass
        