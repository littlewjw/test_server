# -*- coding: utf-8 -*-
"""
Created on Sat Sep 16 11:35:50 2017

@author: WHU
"""
import random
import numpy as np
import tensorflow as tf
def read_txtdata(filename):
   count=0
   f = open(filename,"r")  
   lines = f.readlines()#读取全部内容  
   f.close() 
   sensordata=[]
   driverlabel=[]
   driveremotion=[]
   drivermotion=[]
   for line in lines: 
       if count==0:
           #名字和情绪标签
           line=line.split()
           if(len(line)>0):
               if line[0]=='zyh':
                   driverlabel.append(0)
                   driverlabel.append(0)
               else:
                   driverlabel.append(0)
                   driverlabel.append(1)
           count=count+1
       elif count==1:
           #动作序列
           for i in range(len(line)-1):
                   drivermotion.append(float(line[i]))
           count=count+1
       else:
           if count==151:
               line=line.split() 
               for i in range(len(line)):
                   sensordata.append(float(line[i]))
               #sensordata.append(line.reshape(1,9))
               count=0
           else:
               line=line.split()
               for i in range(len(line)):
                   sensordata.append(float(line[i]))
               count=count+1
   ar1=np.array(sensordata)
   driverlabel1=np.array(driverlabel)
   sensordata1=ar1.reshape(-1,1350)
   driverlabel1=driverlabel1.reshape(-1,2)
   ar=tf.constant(sensordata1)
   driverlabel=tf.constant(driverlabel1)
   driverlabel=tf.reshape(driverlabel,[-1,2])
   sensordata=tf.reshape(ar,[-1,150,3,3])
   return (sensordata1,driverlabel1)
def read_txtdata2(filename):
   count=0
   f = open(filename,"r")  
   lines = f.readlines()#读取全部内容  
   f.close() 
   sensordata=[]
   driverlabel=[]
   drivermotion=[]
   for line in lines: 
       if count==0:
           #名字和情绪标签
           line=line.split()
           if(len(line)>0):
               if line[0]=='zyh':
                   driverlabel.append(1)
                   driverlabel.append(0)
               else:
                   driverlabel.append(0)
                   driverlabel.append(1)
               #drivermotion.append(int(line[1]))
           count=count+1
       elif count==1:
           #动作序列
           for i in range(len(line)-1):
                   drivermotion.append(float(line[i]))
           count=count+1
       else:
           if count==151:
               line=line.split()
               for i in range(len(line)):
                   sensordata.append(float(line[i]))
               #sensordata.append(line.reshape(1,9))
               count=0
           else:
               line=line.split()
               for i in range(len(line)):
                   sensordata.append(float(line[i]))
               count=count+1
   ar1=np.array(sensordata)
   drivermotion1=np.array(drivermotion)
   driverlabel1=np.array(driverlabel)
   sensordata1=ar1.reshape(-1,1350)
   driverlabel1=driverlabel1.reshape(-1,2)
   drivermotion=drivermotion1.reshape(-1,6)
   ar=tf.constant(sensordata1)
   driverlabel=tf.constant(driverlabel1)
   driverlabel=tf.reshape(driverlabel,[-1,2])
   sensordata=tf.reshape(ar,[-1,150,3,3])
   return (sensordata1,driverlabel1,drivermotion)
def randomarray(x,y,z,size):
    r1 = random.sample(range(1,len(x)), size)
    result1=[]
    result2=[]
    result3=[]
    for i in r1:
        result1=np.concatenate((result1,x[i]))
        result2=np.concatenate((result2,y[i]))
        result3=np.concatenate((result3,z[i]))
    return(result1.reshape(-1,1350),result2.reshape(-1,2),result3.reshape(-1,6))
if __name__ == '__main__':
  filename="1.txt"
  filename1="test.txt"
  (traindata,trainlabel,drivermotion)=read_txtdata2(filename)
  (traindata,trainlabel,drivermotion)=randomarray(traindata,trainlabel,drivermotion)