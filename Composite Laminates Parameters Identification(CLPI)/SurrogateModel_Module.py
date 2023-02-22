import sys, time, os
import numpy as np 
import tensorflow.compat.v1 as tf 
import scipy.io as scio
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

class BPNN_Model(QThread):
    handle = -1
    bp_loss_signal = pyqtSignal(list, list)
    bp_accu_signal = pyqtSignal(list, list)
    bp_prog_signal = pyqtSignal(int)
    bp_time_signal = pyqtSignal(str)
    bp_pointer_signal = pyqtSignal(str, int)

    def __init__(self, path_feature_vector, path_store, sample, architecture, learning_rate, batch_size, training_ratio, dim_feature, max_iteration, output):
        super(BPNN_Model, self).__init__()
        self.thread_on = True
        self.flag = 0
        self.architecture   = architecture
        self.architecture_type = 'BP'
        self.learning_rate  = learning_rate
        self.batch_size     = batch_size
        self.training_ratio = training_ratio
        self.dim_feature    = dim_feature
        self.max_iteration  = max_iteration
        self.output         = output
        self.beta1          = 0.9
        self.beta2          = 0.999
        self.epsilon        = 1e-08

        self.parameters = sample['Parameters']
        print(self.parameters)
        print(self.parameters.shape)

        self.parameters = (self.parameters-np.min(self.parameters, 0))/(np.max(self.parameters, 0)-np.min(self.parameters, 0))
        print(self.parameters)
        print(self.parameters.shape)

        self.ae_model_path = path_feature_vector

        self.model_path = path_store + "/NN_Decoder_Model"
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)

        feature_vector_file = self.ae_model_path + 'Sample_Feature_Vectors.npy'
        self.keep_prob = 0.3
        self.feature_vectors = np.squeeze(np.load(feature_vector_file))
        self.train_para, self.test_para, self.train_fect, self.test_fect = train_test_split(self.parameters, self.feature_vectors,
                                                                                            test_size=1-self.training_ratio,
                                                                                            random_state=42,
                                                                                            shuffle=True)


        self.training_size  = int(self.parameters.shape[0]*self.training_ratio)
        
    def R_squared(self, Prediction, Observed):
        Prediction = np.squeeze(np.array(Prediction))
        R_s = r2_score(Observed, Prediction)
        if R_s < 0:
            R_s = 0
        return R_s

    def add_layer(self, x, size, n_layer, flag):
        in_dim = size[0]
        xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
        layer_name = 'layer%s' % n_layer
        if flag == 1:
            with tf.name_scope(layer_name):
                Weights = tf.Variable(tf.random.normal(shape=size, stddev=xavier_stddev), name='W')
                bias    = tf.Variable(tf.zeros(shape=size[1]),name='b')
                x = tf.matmul(x, Weights) + bias
                x = self.lrelu(tf.layers.batch_normalization(x))
                return x
        else:
            with tf.name_scope(layer_name):
                Weights = tf.Variable(tf.random.normal(shape=size, stddev=xavier_stddev), name='W')
                bias    = tf.Variable(tf.zeros(shape=size[1]),name='b')
                x = tf.matmul(x, Weights) + bias
                return x


    def forward_net(self, x, hidden):
        x = self.add_layer(x, size=[self.parameters.shape[1],hidden[0]], n_layer=1, flag=1)
        print(x)

        for q in range(len(hidden)-1):
            x = self.add_layer(x, size=[hidden[q],hidden[q+1]], n_layer=q+2, flag=1)
            print(x)

        x = self.add_layer(x, size=[hidden[-1],self.dim_feature], n_layer=len(hidden)+1, flag=0)
        print(x)
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
        return np.asarray(label_shuffle), np.asarray(U_shuffle)
    
    def lrelu(self, x):
        """ 
        Activation function. 
        """
        return tf.maximum(x, tf.multiply(x, 0.2))


    def NN_Decoder_Train(self):
        self.del_file(self.model_path)
        record = open(self.model_path + '/' + 'Results.txt', 'a+')
        record.write('Dimension of feature vector is ' + str(self.dim_feature))
        record.write('\n')

        ff = open(self.model_path + '/' + 'BPNN training', 'w')

        self.para_input = tf.placeholder("float", [None, self.parameters.shape[1]], name='para_input')
        self.feat_input = tf.placeholder("float", [None, self.dim_feature], name='feat_input')

        feat_pred = self.forward_net(self.para_input, self.architecture)
        tf.add_to_collection('feat_pred', feat_pred)

        loss_para2enc = tf.reduce_sum(tf.pow(self.feat_input-feat_pred, 2))
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            optimizer_para2enc = tf.train.AdamOptimizer(self.learning_rate, self.beta1, self.beta2, self.epsilon).minimize(loss_para2enc)
            
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)
        bp_saver = tf.train.Saver(max_to_keep=1)
        self.step = []
        self.loss = []
        self.accu = []
        self.test_loss = []
        max_accu  = 0

        self.beginningtime_NN_Decoder = time.time()
        print('\n\n','NN-Decoder trianing is processing!')
        for i in range(1, self.max_iteration+1):

            if self.thread_on is True:
                
                batch_x, batch_y = self.next_batch(self.batch_size, self.train_para, self.train_fect)
                if self.dim_feature == 1:
                    batch_y = batch_y.reshape(-1,1)
                    
                _, self.l1 = sess.run([optimizer_para2enc, loss_para2enc], feed_dict={self.para_input: batch_x, self.feat_input: batch_y})

                # test 
                self.feature_vectors_prediction = sess.run([feat_pred], feed_dict={self.para_input: self.test_para})
                self.R_square_test = self.R_squared(self.feature_vectors_prediction, self.test_fect)

                test_loss = np.mean(np.power(self.feature_vectors_prediction - self.test_fect,2))
                if i % self.output == 0 or i == 1:
                    if self.R_square_test > max_accu:
                        max_accu = self.R_square_test
                        best_acc_iter = i
                        bp_saver.save(sess, self.model_path + '/BP-model.ckpt', global_step=i) 
                        np.save(self.model_path + '/Best_NN_Decoder_Model_index.npy',i)

                    print('Step %i: Minibatch Loss: %f  NN-Decoder R2: %f ' % (i, self.l1, self.R_square_test))
                    self.step.append(i)
                    self.loss.append(round(self.l1, 8))
                    self.accu.append(round(self.R_square_test, 6))
                    self.test_loss.append(round(test_loss,8))
                    self.bp_loss_signal.emit(self.step, self.loss)    
                    self.bp_accu_signal.emit(self.step, self.accu)

                self.bp_prog_signal.emit(i)
            else:
                while self.flag:
                    a = 0
                    
        self.endingtime_NN_Decoder = time.time()
        record.write('Best Accuracy is ' + str(round(max_accu, 6)) + ' at iterataion ' + str(best_acc_iter))
        record.write('\n')
        self.str_takingtime_NN_Decoder = 'cost time '+str(round(self.endingtime_NN_Decoder-self.beginningtime_NN_Decoder, 4))+' ms '
        self.bp_time_signal.emit(self.str_takingtime_NN_Decoder)
        record.write('Time consuming is ' + str(round(self.endingtime_NN_Decoder-self.beginningtime_NN_Decoder, 4)))
        record.write('\n')
        ff.write(str(self.step))
        ff.write('\n')
        ff.write(str(self.loss))
        ff.write('\n')
        ff.write(str(self.test_loss))
        ff.close()

        scio.savemat(self.model_path + "\\" + "bpnn_report_loss.mat", {"iteration": np.array([self.step]).T, "train_loss": np.array([self.loss]).T, "test_loss": np.array([self.test_loss]).T})
 
    def run(self):
        while self.thread_on:
            self.NN_Decoder_Train()
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
        path_store= r'E:\Inverse Problem\simply supported beam\SSB'
        BP = BPNN_Model(path_feature_vector = r'E:\Inverse Problem\simply supported beam\SSB\AutoEncoder_Model/', 
                        path_store = path_store, sample=sample, 
                        architecture = [512,256,128,64,64], 
                        dim_feature=6, batch_size=10, 
                        learning_rate=0.001, training_ratio=0.8, 
                        max_iteration = 500000, output = 100)
        BP.run()
        
    else:
        mat_1 = scio.loadmat(r'E:\Inverse Problem\clamped clamped beam\samples_parameters.mat')
        Parameters = mat_1['New_Sample_scaled'] 
        mat_2 = scio.loadmat(r'E:\Inverse Problem\clamped clamped beam\samples_fields.mat')
        Response = mat_2['samples_fields']  
        mat_3 = scio.loadmat(r'E:\Inverse Problem\clamped clamped beam\observed_field.mat')
        inverse_objective = mat_3['observed_field']
        sample = {"Parameters": Parameters, "Response": Response}
        path_store= r'E:\Inverse Problem\clamped clamped beam\CCB'
        BP = BPNN_Model(path_feature_vector = r'E:\Inverse Problem\clamped clamped beam\CCB\AutoEncoder_Model\\', 
                        path_store = path_store, sample=sample, 
                        architecture = [512,256,128], 
                        dim_feature=7, batch_size=10, 
                        learning_rate=0.001, training_ratio=0.7, 
                        max_iteration = 500000, output = 100)
        BP.run()


