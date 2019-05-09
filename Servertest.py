
import tensorflow as tf
import read_txtdata
import time
from http.server import HTTPServer,BaseHTTPRequestHandler     
import io,shutil,urllib,json
import matplotlib.pyplot as plt 
import numpy as np
class MyHttpHandler(BaseHTTPRequestHandler):
    
    #Paramters:
    driving_operation_width = 150
    driving_operation_size = 1350  # 5 * 30 * 9 = 1350  5 seconds 30HZ 9 sensors
    driver_number = 2 # is the driver or others
    
    conv1_core_width = 2
    conv1_core_feature = 64
    
    conv2_core_width = 3
    conv2_core_feature = 72
    
    conv3_core_width = 2
    conv3_core_feature = 24
    
    Max_pool1_height = 1;
    Max_pool1_width = 3;
    Max_pool1_strides = 3;
    
    Max_pool2_height = 1;
    Max_pool2_width = 2;
    Max_pool2_strides = 2;
    
    fc1_nodes = 10
    fc2_nodes = 5
    
    GradientDescentrate = 1e-4
    max_training_round = 2000
    batchsize = 100
    
    #***********************************************************************
    
    
    #sess = tf.InteractiveSession()
    start=time.clock()
    
    x = tf.placeholder("float", [None, driving_operation_size])
    W = tf.Variable(tf.zeros([driving_operation_size,driver_number]))
    b = tf.Variable(tf.zeros([driver_number]))
    y = tf.nn.softmax(tf.matmul(x,W) + b)
    y_ = tf.placeholder("float", [None,driver_number])
    xxx = tf.placeholder("float", [None, 6])
    xxx1=tf.reshape(x, [-1,6])
    
    def weight_variable(shape):
      initial = tf.truncated_normal(shape, stddev=0.1)
      return tf.Variable(initial)
    
    def bias_variable(shape):
      initial = tf.constant(0.1, shape=shape)
      return tf.Variable(initial)
      
    def conv2d(x, W):
      return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    
    
    W_conv1 = weight_variable([conv1_core_width, 3, 3, conv1_core_feature])
    b_conv1 = bias_variable([conv1_core_feature])
    
    x_image = tf.reshape(x, [-1,driving_operation_width,3,3])
    
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, Max_pool1_height, Max_pool1_width, 1],strides=[1, Max_pool1_strides, Max_pool1_strides, 1], padding='SAME')
    
    W_conv2 = weight_variable([conv2_core_width, 1, conv1_core_feature, conv2_core_feature])
    b_conv2 = bias_variable([conv2_core_feature])
    
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, Max_pool2_height, Max_pool2_width, 1],strides=[1, Max_pool2_strides, Max_pool2_strides, 1], padding='SAME')
    
    W_conv3 = weight_variable([conv3_core_width, 1, conv2_core_feature, conv3_core_feature])
    b_conv3 = bias_variable([conv3_core_feature])
    
    h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
    
    W_fc1 = weight_variable([int((driving_operation_width/Max_pool1_strides * 3 * conv3_core_feature)/Max_pool2_strides+6),fc1_nodes])
    b_fc1 = bias_variable([fc1_nodes])
    
    h_pool2_flat = tf.reshape(h_pool2, [-1,int( (driving_operation_width/Max_pool1_strides)/Max_pool2_strides * 3 * conv3_core_feature)])
    h_pool2_flat=tf.concat([h_pool2_flat, xxx],1)
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    
    W_fc2 = weight_variable([fc1_nodes, fc2_nodes])
    b_fc2 = bias_variable([fc2_nodes])
    
    h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)
    
    W_fc3 = weight_variable([fc2_nodes, driver_number])
    b_fc3 = bias_variable([driver_number])
    
    keep_prob = tf.placeholder("float")
    h_fc3_drop = tf.nn.dropout(h_fc2, keep_prob)
    
    y_conv=tf.nn.softmax(tf.matmul(h_fc3_drop, W_fc3) + b_fc3)
    saver = tf.train.import_meta_graph('./model.ckpt.meta')
    cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
    corrects=tf.argmax(y_conv,1)
    #train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    train_step = tf.train.GradientDescentOptimizer(GradientDescentrate).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
   
                                                                                                                     
    def das():
        print("1")
    def do_POST(self):
        length = int(self.headers['Content-Length'])
        readdata = self.rfile.read(length).decode('utf-8')
        post_data = json.loads(readdata)
        # You now have a dictionary of the post data
        print(post_data)
        result=-1
        identity=[]
        
        if(post_data['Type']==1):
            for currenti in range(1,post_data['OperationSize']+1):
                x1=range(0,150) 
                ax=[]
                ay=[]
                az=[]
                gx=[]
                gy=[]
                gz=[]
                ox=[]
                oy=[]
                oz=[]
                drivermotion=[]
                for i in range(1,151):
                   ax.append(float(post_data['OperationInfo'][str(currenti)]['OperationData'][str(i)][0]))
                   ax.append(float(post_data['OperationInfo'][str(currenti)]['OperationData'][str(i)][1]))
                   ax.append(float(post_data['OperationInfo'][str(currenti)]['OperationData'][str(i)][2]-9.8))
                   ax.append(float(post_data['OperationInfo'][str(currenti)]['OperationData'][str(i)][3]))
                   ax.append(float(post_data['OperationInfo'][str(currenti)]['OperationData'][str(i)][4]))
                   ax.append(float(post_data['OperationInfo'][str(currenti)]['OperationData'][str(i)][5]))
                   ax.append(float(post_data['OperationInfo'][str(currenti)]['OperationData'][str(i)][6]))
                   ax.append(float(post_data['OperationInfo'][str(currenti)]['OperationData'][str(i)][7]))
                   ax.append(float(post_data['OperationInfo'][str(currenti)]['OperationData'][str(i)][8]))
                ar1=np.array(ax)
                
                for i in range(len(post_data['OperationInfo'][str(currenti)]['Operation'])):
                       drivermotion.append(post_data['OperationInfo'][str(currenti)]['Operation'][i])
                drivermotion1=np.array(drivermotion)
                sensordata1=ar1.reshape(-1,1350)
                drivermotion=drivermotion1.reshape(-1,6)
                #print(driverlabel1) 
                #print(sensordata1)
                with tf.Session() as sess:
                    sess.run(tf.global_variables_initializer())
                    self.saver.restore(sess, "./model.ckpt")
                    result=self.corrects.eval(feed_dict={self.x:sensordata1,self.xxx:drivermotion,self.keep_prob:1.0}).tolist()[0]
                    print (result)
                    identity.append(result)
            #画图代码
        
        result={}
        result["OperationSize"]=str(post_data['OperationSize'])
        OperationInfo=[]
        for currenti in range(1,post_data['OperationSize']+1):
            OperationInfo.append(post_data['OperationInfo'][str(currenti)]['Starttime'])
            OperationInfo.append(post_data['OperationInfo'][str(currenti)]['Endtime'])
            OperationInfo.append(identity[currenti-1])
            
        result["OperationInfo"]=OperationInfo
        data = json.dumps(result)
        #data = json.dumps({'Identity':1})
        enc="UTF-8"  
        encoded = ''.join(data).encode(enc)  
        f = io.BytesIO()  
        f.write(encoded)  
        f.seek(0)  
        self.send_response(200)  
        self.send_header("Content-type", "text/html; charset=%s" % enc)  
        self.send_header("Content-Length", str(len(encoded)))  
        self.end_headers()  
        shutil.copyfileobj(f,self.wfile)

    
httpd=HTTPServer(('',9601),MyHttpHandler)     
print("Server started on 127.0.0.1,port 9601......")     
httpd.serve_forever() 







    
