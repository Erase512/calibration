import pandas as pd
import numpy as np
import config as C



def fen(path = C.path):
    
    xlsfile=pd.read_csv(path,encoding='gb18030')
    data=np.array(xlsfile)
    all_data = data
    
    if C.csv==1:
        train_x = all_data[0:1583,0:3]
        test_x = all_data[1584:1759,0:3]
        
        train_y= all_data[0:1583,4:]
        test_y= all_data[1584:1759,4:]
        
    elif C.csv==2:
        train_x = all_data[0:863,0:3]
        test_x = all_data[864:959,0:3]
        
        train_y= all_data[0:863,4:]
        test_y= all_data[864:959,4:]
        
    elif C.csv==3:
        train_x = all_data[0:2447,0:3]
        test_x = all_data[2448:2719,0:3]
        
        train_y= all_data[0:2447,4:]
        test_y= all_data[2448:2719,4:]
        
    elif C.csv==4:
        train_x = all_data[0:1407,0:3]
        test_x = all_data[1408:1759,0:3]
        
        train_y= all_data[0:1407,4:]
        test_y= all_data[1408:1759,4:]
    elif C.csv==5:
        train_x = all_data[0:767,0:3]
        test_x = all_data[768:959,0:3]
        
        train_y= all_data[0:767,4:]
        test_y= all_data[768:959,4:]
    elif C.csv==6:
        train_x = all_data[0:2175,0:3]
        test_x = all_data[2175:2719,0:3]
        
        train_y= all_data[0:2175,4:5]
        test_y= all_data[2175:2720,4:5]

    return np.array(train_x), np.array(train_y), np.array(test_x), np.array(test_y)



# def fen(path = C.path,fenbu=C.xun_ce_rate):
    # df = pd.read_excel(path)
    # data=df.values
    # print(data.shape,'data')
    # data = np.array(data)
    # xunlian_x = []
    # xunlian_y = []
    
    # ceshi_x = []
    # ceshi_y = []
    
    # ceshi_num = int(data.shape[0]*fenbu)
    # print(ceshi_num,'ceshi_num1')
    
    # for xun in data[:data.shape[0] - ceshi_num]:
        # xunlian_x.append([float(xun[0]),float(xun[1]),float(xun[2])])
        # xunlian_y.append(float(xun[-1]))

    # for ce in data[data.shape[0] - ceshi_num:]:
        # ceshi_x.append([float(ce[0]),float(ce[1]),float(ce[2])])
        # ceshi_y.append(float(ce[-1]))

    # print(np.array(xunlian_x).shape,'shape1')
    # print(np.array(xunlian_y).shape,'shape2')
    # print(np.array(ceshi_x).shape,'shape3')
    # print(np.array(ceshi_y).shape,'shape4')   
    
    # return np.array(xunlian_x), np.array(xunlian_y), np.array(ceshi_x), np.array(ceshi_y)
    
if __name__ == "__main__":
    
    df = pd.read_excel()
    
    data=df.values
    
    data = np.array(data)
    print(data.shape)
    xunlian_x,xunlian_y,ceshi_x,ceshi_y = fen(data)

    oo = 0
    for j,z in zip(xunlian_x,xunlian_y):
        oo += 1
        print(j,"-=-=-=-=-=",z)
        
        if oo == 12:
            exit()
        
    print(xunlian_y)