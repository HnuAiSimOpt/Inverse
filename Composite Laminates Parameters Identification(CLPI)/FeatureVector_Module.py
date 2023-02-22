from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import pyqtSignal, QObject, QThread
from PyQt5.QtWidgets import QProgressBar, QMessageBox, QSizePolicy, QGridLayout, QDialog, QHeaderView, QMenu
from PyQt5.QtWidgets import QApplication, QPushButton, QVBoxLayout, QMainWindow, QWidget, QTabWidget, QTableView
from PyQt5.QtGui import QFont, QStandardItem, QStandardItemModel

import sys, time, os
import numpy as np 
import pandas as pd
import tensorflow as tf 
import scipy.io as scio
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

class AutoEncoder_Model(QThread):
    handle = -1
    ae_loss_signal = pyqtSignal(list, list)
    ae_accu_signal = pyqtSignal(list, list)
    ae_prog_signal = pyqtSignal(int)
    ae_time_signal = pyqtSignal(str)
    ae_pointer_signal = pyqtSignal(str, int)
    ae_feat_vectors_signal = pyqtSignal(list)
    
    def __init__(self, path, sample, dimension_feature, architecture, batch_size, ae_learning_rate, ae_training_ratio, max_iteration, output, inverse_objective):
        super(AutoEncoder_Model, self).__init__()
        self.architecture = architecture  
        self.architecture_type = 'AutoEncoder'
        self.Response   = sample['Response']
        self.Response_ori = self.Response
        self.min_Response = np.min(self.Response, 0)
        self.max_Response = np.max(self.Response, 0)
        self.Response   = (self.Response-self.min_Response)/((self.max_Response-self.min_Response))
        self.dimension_feature   = dimension_feature
        self.max_iteration = max_iteration
        self.output        = output
        self.batch_size     = batch_size
        self.ae_learning_rate = ae_learning_rate
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-08
        self.ae_training_ratio = ae_training_ratio

        self.train_Response,self.test_Response=train_test_split(self.Response,test_size=1-self.ae_training_ratio,random_state=42,shuffle=True)

        self.model_path = path + '\\' + self.architecture_type + '_Model'
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)
        self.inverse_objective = inverse_objective
        self.flag = 1
        self.keep_prob      = 0.6
        self.thread_on = True
        self.flag = 0

    def Print_HyperParameters(self):
        print("training data ratio ----------->", self.ae_training_ratio)
        print("feature dimension   ----------->", self.dimension_feature)
        print("training epoch      ----------->", self.max_iteration)
        print("learning rate       ----------->", self.ae_learning_rate)
        print("training batch      ----------->", self.batch_size)
        print("output step         ----------->", self.output)
        

    def add_layer(self, x, size, n_layer):
        in_dim = size[0]
        xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
        layer_name = 'layer%s' % n_layer
        with tf.name_scope(layer_name):
            Weights = tf.Variable(tf.random.normal(shape=size, stddev=xavier_stddev), name='W')
            bias    = tf.Variable(tf.zeros(shape=size[1]),name='b')
            x = tf.matmul(x, Weights) + bias
            x = self.lrelu(tf.layers.batch_normalization(x))
            return x

    def del_file(self, path_Model):
        if not os.listdir(path_Model): 
            print('No existing Models!')
            pass
        else:
            for i in os.listdir(path_Model) :
                file_data = path_Model + "/" + i
                if os.path.isfile(file_data) == True:
                    os.remove(file_data)
                else:
                    del_file(file_data)

    def Encoder(self, x, hidden):
        x = self.add_layer(x, size=[self.Response.shape[1],hidden[0]], n_layer=1)
        print(x)
        for q in range(len(hidden)-1):
            x = self.add_layer(x, size=[hidden[q],hidden[q+1]], n_layer=q+2)
            print(x)
        x = self.add_layer(x, size=[hidden[-1],self.dimension_feature], n_layer=len(hidden)+1)
        print(x)
        return x

    def Decoder(self, x, hidden):
        symmetry_hidden = sorted(hidden)
        x = self.add_layer(x, size=[self.dimension_feature,symmetry_hidden[0]], n_layer=len(symmetry_hidden)+2)
        print(x)

        for q in range(len(symmetry_hidden)-1):
            x = self.add_layer(x, size=[symmetry_hidden[q],symmetry_hidden[q+1]], n_layer=q+3+len(symmetry_hidden))
            print(x)
        x = self.add_layer(x, size=[symmetry_hidden[-1],self.Response.shape[1]], n_layer=len(hidden)+1+len(symmetry_hidden)+1)
        print(x)
        return x
    
    def next_batch(self, num, labels, U):
        '''
        Return a total of `num` random samples and labels. 
        '''
        num = self.batch_size
        idx = np.arange(0, len(labels))
        np.random.shuffle(idx)    
        idx = idx[:self.batch_size]    
        U_shuffle = [U[i] for i in idx]
        label_shuffle = [labels[i] for i in idx]
        return np.asarray(U_shuffle), np.asarray(label_shuffle)
    
    def lrelu(self, x):
        """ 
        Activation function. 
        """
        return tf.maximum(x, tf.multiply(x, 0.2))

    def xavier_init(self, size):
        """initilization"""
        in_dim = size[0]
        xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
        return tf.random.normal(shape=size, stddev=xavier_stddev)

    def R_squared(self, Prediction, Observed):

        R_s_1 = r2_score(Observed, Prediction)

        return R_s_1

    def stop(self):
        self.flag = 0

    def get_feature_vectors(self, all_sample):

        num1 = np.load(self.model_path + '/Best_AE_Model_index.npy')
        str1_AE = self.model_path + '/AE-model.ckpt-' + str(num1) + '.meta'
        str2_AE = self.model_path + '/AE-model.ckpt-' + str(num1)
        tf.reset_default_graph()
        with tf.Session() as sess:       
            saver = tf.train.import_meta_graph(str1_AE)
            saver.restore(sess,tf.train.latest_checkpoint(self.model_path))
            graph = tf.get_default_graph()
            Disp_input = graph.get_operation_by_name('Disp_input').outputs[0]
            feat_vect  = tf.get_collection('feat_vect')[0]
            Disp_Pred  = tf.get_collection('Disp_Pred')[0]
            Sample_Feature_Vectors = sess.run([feat_vect], feed_dict={Disp_input: all_sample})
            np.save(self.model_path + '/Sample_Feature_Vectors.npy', Sample_Feature_Vectors)

        return Sample_Feature_Vectors

    def neural_network_train(self):
        try:
            self.handle = ctypes.windll.kernel32.OpenThread(
                win32con.PROCESS_ALL_ACCESS, False, int(QThread.currentThreadId()))
        except Exception as e:
            print('get thread handle failed', e)

        print('Present thread id of AutoEncoder: ', int(QThread.currentThreadId())) 
        self.thread_name = 'ae_thread'
        self.thread_pointer = int(QThread.currentThreadId())
        print(self.thread_name, self.thread_pointer,'\n')
        self.ae_pointer_signal.emit(self.thread_name, self.thread_pointer)

        self.Disp_input = tf.placeholder("float", [None, self.Response.shape[1]], name='Disp_input')
        self.del_file(self.model_path)
        
        # save training results
        record = open(self.model_path + '/' + 'Results.txt', 'a+')
        record.write(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        record.write('\n')
        record.write('Dimension of feature vector is ' + str(self.dimension_feature))
        record.write('\n')

        max_accu = 0
        feat_vect = self.Encoder(self.Disp_input, self.architecture)
        tf.add_to_collection('feat_vect', feat_vect)
        Disp_Pred = self.Decoder(feat_vect, self.architecture)
        tf.add_to_collection('Disp_Pred', Disp_Pred)

        loss_ae = tf.reduce_mean(tf.pow(self.Disp_input-Disp_Pred, 2))
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            optimizer_vae = tf.train.AdamOptimizer(self.ae_learning_rate, self.beta1, self.beta2, self.epsilon).minimize(loss_ae)
        
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)
        saver = tf.train.Saver(max_to_keep=1)
        self.ae_step = []
        self.ae_loss = []
        self.ae_accu = []
        self.test_loss = []

        ff = open(self.model_path + '/' + 'trainingProcess.txt', 'w+')
        self.beginningtime = time.time()

        for i in range(1, self.max_iteration+1):
            if self.thread_on is True:
                batch_x, _ = self.next_batch(self.batch_size, self.train_Response, self.train_Response)
                _, self.l = sess.run([optimizer_vae, loss_ae], feed_dict={self.Disp_input: batch_x})

                # test error
                self.Feat_ae, self.Re_ae = sess.run([feat_vect, Disp_Pred], feed_dict={self.Disp_input: self.test_Response})
                self.Disp_AE = self.Re_ae 
                self.R_square_test = self.R_squared(self.Disp_AE, self.test_Response) 
                test_loss = np.mean(np.power(self.Disp_AE - self.test_Response, 2))

                if i % self.output == 0 or i==1:
                    if self.R_square_test > max_accu:
                        max_accu = self.R_square_test
                        best_acc_iter = i
                        saver.save(sess, self.model_path + '/AE-model.ckpt', global_step=i)
                        np.save(self.model_path + '/Best_AE_Model_index.npy',i)
                        
                    print('Step %i: Minibatch Loss: %f Test_R2: %f test_loss: %f ' % (i, self.l, self.R_square_test, test_loss))

                    # save model training index
                    self.ae_step.append(i)
                    self.ae_loss.append(round(self.l, 8))
                    self.ae_accu.append(round(self.R_square_test, 6))
                    self.test_loss.append(round(test_loss, 8))
                    # emit signal
                    self.ae_loss_signal.emit(self.ae_step, self.ae_loss)    
                    self.ae_accu_signal.emit(self.ae_step, self.ae_accu)
                self.ae_prog_signal.emit(i)

            else:
                while self.flag:
                    a = 0

        self.endingtime = time.time()
        scio.savemat(self.model_path + "\\" + "ae_report_loss.mat",{"iteration": np.array([self.ae_step]).T, "train_loss": np.array([self.ae_loss]).T, "test_loss": np.array([self.test_loss]).T}) 
        ff.write(str(self.ae_step))
        ff.write("\n")
        ff.write(str(self.ae_loss))
        ff.write("\n")
        ff.write(str(self.test_loss))
        ff.close()
        record.write('Best Accuracy is ' + str(round(max_accu,6)) + ' at iterataion ' + str(best_acc_iter))
        record.write('\n')
        self.str_takingtime = 'Extracting Feature Vector Process Time Cost: ' + str(round(self.endingtime-self.beginningtime, 4))+' ms '
        self.ae_time_signal.emit(self.str_takingtime)
        Sample_Feature_Vectors = self.get_feature_vectors(self.Response)
        record.write('Time consuming is ' + str(round(self.endingtime-self.beginningtime, 4)))
        record.write('\n')
        np.savetxt(self.model_path + '/' + 'Sample_Feature_Vectors.txt', np.squeeze(Sample_Feature_Vectors))

    def run(self):
        while self.thread_on:
            self.neural_network_train()
            self.thread_on = False


if __name__ == '__main__':

    case_num = 2

    if case_num == 1:
        mat_1 = scio.loadmat(r'E:\Inverse Problem\simply supported beam\samples_parameters.mat')
        Parameters = mat_1['New_Sample_scaled'] 
        mat_2 = scio.loadmat(r'E:\Inverse Problem\simply supported beam\samples_fields.mat')
        Response = mat_2['samples_fields']  
        mat_3 = scio.loadmat(r'E:\Inverse Problem\simply supported beam\observed_field.mat')
        inverse_objective = mat_3['observed_field']
        sample = {"Parameters": Parameters, "Response": Response}
        path_store= r'E:\Inverse Problem\simply supported beam\SSB\ershen'
        path = r'E:\Inverse Problem\simply supported beam\SSB'
        aenet = AutoEncoder_Model(path = path, sample=sample, dimension_feature=6, 
                                    architecture = [512,256,128,64,32],batch_size=16, 
                                    ae_learning_rate=0.0001, ae_training_ratio=0.7, 
                                    max_iteration = 50000, output = 10, inverse_objective =inverse_objective)
        aenet.run()
        
    else:
        mat_1 = scio.loadmat(r'E:\Inverse Problem\clamped clamped beam\samples_parameters.mat')
        Parameters = mat_1['New_Sample_scaled'] 
        mat_2 = scio.loadmat(r'E:\Inverse Problem\clamped clamped beam\samples_fields.mat')
        Response = mat_2['samples_fields']  
        mat_3 = scio.loadmat(r'E:\Inverse Problem\clamped clamped beam\observed_field.mat')
        inverse_objective = mat_3['observed_field']
        sample = {"Parameters": Parameters, "Response": Response}
        path_store= r'E:\Inverse Problem\clamped clamped beam\CCB'
        path = r'E:\Inverse Problem\clamped clamped beam\CCB'
        aenet = AutoEncoder_Model(path = path, sample=sample, dimension_feature=7, 
                                    architecture = [512,256,128,64,32],batch_size=16, 
                                    ae_learning_rate=0.0001, ae_training_ratio=0.8, 
                                    max_iteration = 50000, output = 10, inverse_objective =inverse_objective )
        aenet.run()

