import tensorflow as tf 
import logging
import numpy as np 
#from pylab import mpl  
from model import ratio as M
import matplotlib.pyplot as plt
import sys#, getopt
import xlwt 
from utils import fen
import config as C
import time    
import os
import PIL.Image as I
import glob
import random
from openpyxl import Workbook
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.metrics import mean_absolute_error

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
#mpl.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['font.sans-serif']=['Times New Roman']
#plt.rcParams['font.sans-serif']=['SimHei']    # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False    # 用来正常显示负号




class the_net():
    def __init__(self,num_h,num_md):
        self.save_path='ckpt\\'
        self.mse=[]
        self.learning_rate = C.learning_rate
        self.global_steps = tf.Variable(0,trainable=False)
        
        self.sess=tf.Session(config=tf.ConfigProto(
            inter_op_parallelism_threads=1, 
            intra_op_parallelism_threads=1,
        ))
        self.build_net(num_h,num_d)
        self.build_opt()
        self.saver=tf.train.Saver(max_to_keep=1)
        self.initialize()
    
    def build_net(self,num_h,num_d):
        h=num_h
        d=num_d
        
        self.y = M.neural_network(h,d)
        
        self.alue = tf.squeeze(self.y.value,axis=-1)


        self.chazhi=self.y.exact_y-self.alue
        self.loss=tf.reduce_mean(tf.pow(self.chazhi,2))
        print(self.loss,'self.loss1')
    def build_opt(self):
        self.learning_rate = tf.train.exponential_decay(self.learning_rate, \
                                self.global_steps, C.decay_step, C.decay_rate , staircase=True)

        self.opt=tf.train.AdamOptimizer(learning_rate=self.learning_rate)\
                .minimize(self.loss,global_step=self.global_steps)
        
    def initialize(self):             
        ckpt=tf.train.latest_checkpoint(self.save_path)     #自动找到最近保存的变量文件
        if ckpt!=None:                                      #不等于
            self.saver.restore(self.sess,ckpt)              #打开ckpt路径
        else:
            self.sess.run(tf.global_variables_initializer()) #
    def train(self,save_weight,save_path,num_h,num_d):
        data_train,label_train,test_data,test_label = fen()
        print(data_train.shape,'data_train.shape')
        print(label_train.shape,'label_train.shape')
        
        
        data_train=tf.constant(data_train,shape=(2175,3),dtype=tf.float64)
        label_train=tf.constant(label_train,shape=(2175,1),dtype=tf.float64)
       
        datasat=tf.data.Dataset.from_tensor_slices((data_train,label_train)).shuffle(buffer_size=200).batch(C.batchsize)
       
        print('开始训练')
        element = datasat.make_one_shot_iterator().get_next()
        for epoch in range(C.tal_epoch):
             
            with tf.Session() as sess:
                try:
                    while True:
                        x=sess.run(element)
                        train_data=(x[0])
                        train_label=(x[1])
                        
                        train_data=np.array(train_data)
                        train_label=np.array(train_label)
                        loss,_,gs,l_r=self.sess.run([self.loss,self.opt,self.global_steps,self.learning_rate],feed_dict={self.y.exact_y:train_label,self.y.input:train_data})
                 
                except tf.errors.OutOfRangeError:
                    print("")

            predict=self.sess.run(self.alue,\
                    feed_dict={self.y.exact_y:test_label,self.y.input:test_data})
            R2 = r2_score(test_label, predict)
            
            
            MSE = mean_squared_error(test_label,predict)
           
            MAE = mean_absolute_error(test_label,predict)
            if loss<min(loss_list):
                loss_list.append(loss)
            if MAE<min(mae_list):
                mae_list.append(MAE)
                mae_epoch_list.append(epoch)
                
                work_book = xlwt.Workbook(encoding = "UTF-8" )#获取一个workbook，然后设置中文编码集
                sheet = work_book.add_sheet('predict',cell_overwrite_ok=True)   #这个是在workbook里面生成一页
                
                            
                col=('predict','result','MAE','MSE','R2','loss')
                for b in range(0,6):
                    sheet.write(0,b,col[b])
                    
                    for i in range(0,len(predict)):
                        data = predict[i]
                        sheet.write(i+1,0,data)
                    sheet.write(1,2,MAE)
                    sheet.write(1,3,MSE)
                    sheet.write(1,4,R2)
                    sheet.write(1,5,min(loss_list))
                               
                filename1=('./result/{}-{}').format(num_h,num_d)
                if os.path.isdir(filename1) == False:
                        os.mkdir(filename1)
                filename=filename1+'/1.csv'
                
                work_book.save(filename)                #把workbook保存到文件里
                '''
                mae_pic_path=('./mae_test_data/{}-{}-{}-0.01-128/').format(string,num_h,num_d)
                weight_path=('./mae_weight/{}-{}-{}-0.01-128/').format(string,num_h,num_d)
                if os.path.isdir(weight_path) == False:
                        os.mkdir(weight_path)
                self.saver.save(self.sess, weight_path+"/{}check.ckpt".format(epoch))
                if os.path.isdir(mae_pic_path) == False:
                        os.mkdir(mae_pic_path)
                self.test(loss,epoch,mae_pic_path)
                '''
            if MSE<min(MSE_list):
                MSE_list.append(MSE)
                mse_epoch_list.append(epoch)
            
            if R2>max(R2_list):
                R2_list.append(R2)
                epoch_list.append(epoch)
                #if os.path.isdir(save_weight) == False:
                    #os.mkdir(save_weight)
            
                #self.saver.save(self.sess, save_weight+"/{}check.ckpt".format(epoch))
                #self.test(loss,epoch,save_path)
                
            #self.trainable_vars = tf.trainable_variables()
            #self.num_trainable_params = np.sum([np.prod(var.get_shape().as_list()) for var in self.trainable_vars])
            
            #print(f"Total Trainable Parameters: {self.num_trainable_params}")    
            #exit()
            print("[{}]loss:{:.5f}---[{}]:best_R2:{:.3f}---MSE:{:.8f}---[{}]:best_MSE:{:.3f}--MAE:{:.8f}--[{}]:best_MAE:{}--{}-{}".format(epoch,loss,max(epoch_list),max(R2_list),MSE,max(mse_epoch_list),min(MSE_list),MAE,max(mae_epoch_list),min(mae_list),num_h,num_d))
               
              
    def test(self,loss,epoch,save_path):
        
        _,_,test_data,test_label= fen()    
        print('test the dataset and num of the dataset：{}'.format(len(test_label)))
        predict=self.sess.run(self.alue,\
                    feed_dict={self.y.exact_y:test_label,self.y.input:test_data})
    
        MSE = mean_squared_error(test_label,predict)
        MAE = mean_absolute_error(test_label,predict)
        
        self.plot(save_path,predict,test_label,loss,epoch,MSE,MAE)
        
    def plot(self,save_path,predict,value,loss,epoch,mse,mae):
        plt.figure()

        plt.plot(value)
        plt.plot(predict)
        string1=('mae:{}  mse:{}').format(mae,mse)
        plt.title(string1)
        if os.path.isdir(save_path) == False:
            os.mkdir(save_path)
        plt.legend(["value","predict","mse is %s"%mse,'*mae is %s'%mae])
        plt.savefig(save_path+"%s.jpg"%epoch)
        plt.close()
        plt.show()
       
if __name__=="__main__":
    global loss_list
    global epoch_list
    global mae_epoch_list
    global mae_list
    global R2_list
    global string
    global mse_epoch_list
    global MSE_list
    
    mse_epoch_list=[0]
    R2_list=[0]
    MSE_list=[100]
    loss_list = [100]
    epoch_list = [0]
    
    mae_epoch_list = [0]
  
    num_h=C.num_h
    num_d=C.num_d
    if C.csv==1:
        string = 'NTU'
    if C.csv==2:
        string = 'water'
    if C.csv==3:
        string = 'all'
    if C.csv==4:
        string = 'NTU_8_2'
        mae_list = [0.62]
    if C.csv==5:
        string = 'water_8_2'
    if C.csv==6:
        mae_list = [10]
        string = 'all_8_2'
        
    save_path=('./test_data/{}-{}-{}-0.01-128/').format(string,num_h,num_d)
    save_weight=('./weight/{}-{}-{}-0.01-128/').format(string,num_h,num_d)
    
    main_net=the_net(num_h,num_d)        
    main_net.train(save_weight,save_path,num_h,num_d)

    