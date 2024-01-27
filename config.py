#地址
csv=6
path = ('./data/{}.csv').format(csv)


#学习率
learning_rate = 0.001#0.1
decay_step =1000   #每多少步学习率衰减一次
decay_rate = 1 #每次衰减原来的多少倍


#训练轮次等
tal_epoch = 1000000#总训练轮次
batchsize = 128#每次放多少个数据进入训练[8]
step_test = 100       #每多少步保存，测试一次


#准确率的判断
jingdu = 0.01          #精度

#输入的数据
input_num = 3
output_num = 1

#ratio的设置14-11 3-2
num_h =10#bg
num_d =11#small
#save_path=('./test_data/{}-{}/').format(num_h,num_d)
#save_weight=('./weight/{}-{}/').format(num_h,num_d)
