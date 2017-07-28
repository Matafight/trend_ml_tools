import time
import os



class log_class():
    model_path= ''
    cur_time = None
    def __init__(self,model_name,data_name):
        if not os.path.exists('./log'):
            os.mkdir('./log')
        n_path = os.path.join('./log',data_name)
        if not os.path.exists(n_path):
            os.mkdir(n_path)

        cur_time = time.strftime("%Y-%m-%d-%H-%M",time.localtime())
        self.cur_time = cur_time
        path = os.path.join(n_path,model_name+'_'+cur_time+'.txt')
        with open(path,'a') as fh:
            fh.write(cur_time+'\n')
        self.model_path=path
    def get_time(self):
        return self.cur_time
    def add(self,info,ifdict = 0):
        if ifdict == 0: 
            with open(self.model_path,'a') as fh:
                fh.write(info+'\n')
        else:
            with open(self.model_path,'a') as fh:
                for item in info:
                    fh.write(item+':')
                    fh.write(str(info[item])+' ')

if __name__ == '__main__':
    mylog = log_class('testmodel','data')
    mylog.add('hello')
    params = {'id1':1,'id2':'aaa','id3':'cc'}
    mylog.add(params,1)

