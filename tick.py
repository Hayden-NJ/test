class Tick():
    def __init__(self, start=None, aim=None):
        if not start:
            self.num = 0
        else:
            self.num = start
        if not aim:
            self.aim = '?'
        else:
            self.aim = aim
        
    def __call__(self,step=1,):
        if self.num%step==0:
            print(f'{time.ctime(time.time())}: {self.num}/{self.aim}')
        self.num+=1
        
    def timebins(self):
        pass
        