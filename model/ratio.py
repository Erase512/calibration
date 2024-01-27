import tensorflow as tf 
import logging
import numpy as np 
#from pylab import mpl  
import model
import matplotlib.pyplot as plt
import sys#, getopt
import config as C
import time    
import os
import PIL.Image as I
import glob



class neural_network:
    def __init__(self,num_h,num_d):
        self.weight_initialization =  tf.contrib.layers.xavier_initializer()
        self.input_num = C.input_num
        self.output_num = C.output_num
        self.construct_input()
        
        self.sess=tf.Session(config=tf.ConfigProto(
            inter_op_parallelism_threads=1, 
            intra_op_parallelism_threads=1,
        ))
        self.build_value_ra(num_h,num_d)        
        
        
        
    def construct_input(self):
        self.input = tf.placeholder(tf.float64,[None, self.input_num])
        self.exact_y = tf.placeholder(tf.float64,[None,self.output_num])
        print(self.exact_y,'self.exact_y')
        
    def build_value_ra(self,num_h,num_d):
        print("开始计算value，即NET(x)")
        #num_h=num_h
        #num_d=num_d
        print("计算wi和bi")
        
        one_h=[]
        for i in range(1,num_h+1):
            one_hi= tf.get_variable('weight_h'+str(i) , 
                                shape=[self.input_num,self.output_num], 
                                initializer=self.weight_initialization, 
                                dtype=tf.float64)
            one_h.append(one_hi)
        
        one_d=[]
        for i in range(1,num_d+1):
            one_di= tf.get_variable('weight_d'+str(i) ,
                                shape=[self.input_num,self.output_num],
                                initializer=self.weight_initialization,
                                dtype=tf.float64)
            one_d.append(one_di)    
        
        two_h=[]
        for j in range(1,num_h+1):
            two_hj = tf.get_variable('bias_h' +str(j) , 
                                shape=[self.output_num], 
                                initializer=self.weight_initialization, 
                                dtype=tf.float64)
            two_h.append(two_hj)                                
        
        two_d=[]
        for j in range(1,num_d+1):
            two_dj = tf.get_variable('bias_d' +str(j) ,
                                shape=[self.output_num],
                                initializer=self.weight_initialization,
                                dtype=tf.float64)
            two_d.append(two_dj)
        #================================
       
        #===================================
        cm_h=[]
        
        for i in range(num_h):
            
            
            ci=tf.add(tf.matmul(self.input,one_h[i]), two_h[i])
       
            print(ci,'======',i,'ci')
            
            cm_h.append(ci)
            
    
        cm_d=[]
        for i in range(num_d):
            ci=tf.add(tf.matmul(self.input, one_d[i]), two_d[i])
            cm_d.append(ci)
    
            

        b_hhh = tf.get_variable('bias_hhh',
                               shape=[1],
                               initializer=tf.random_normal_initializer(),
                               dtype=tf.float64)

        d1=0.0
        d2=0.0
        for i in range(num_h):
            if i==0:
                d1=cm_h[0]
            else:
                d1=tf.multiply(d1,cm_h[i])

        for i in range(num_d):
            if i==0:
                d2=cm_d[0]
            else:
                d2=tf.multiply(d2,cm_d[i])
        aaa = tf.div(d1, d2)
        print(aaa,'111')
        #aaa = tf.case(aaa,tf.float64)
        #self.value=tf.add(aaa,b_hhh)
        self.value=aaa

        