import numpy as np
import theano
import theano.tensor as T
from theano.tensor.nnet import conv2d
from theano.tensor.signal import downsample
from sklearn.preprocessing import StandardScaler
import sys
import glob
import random

#load data
X_files = sorted(glob.glob('../preprocessing/X_*'))
y_files = sorted(glob.glob('../preprocessing/y_*'))

X = []
y = []
for i in range(len(X_files)):
    X.append(np.load(X_files[i]))
    y.append(np.load(y_files[i]))

#standard scaler
scaler = StandardScaler()
X_temp = np.array(X)
X_flattened = scaler.fit_transform(X_temp.reshape(X_temp.shape[0],X_temp.shape[1]*X_temp.shape[2]))
X = X_flattened.reshape(X_temp.shape[0],X_temp.shape[1],X_temp.shape[2])

#pad by 16 to allow for testing of edge pixels
X = np.pad(X,((0,0),(20,20),(20,20)),'constant')
y = np.pad(y,((0,0),(20,20),(20,20)),'constant')

#test/train split
X_train = X[2:]
y_train = np.array(y[2:])
X_test = X[:2]
y_test = np.array(y[:2])

nonzeros_train = np.nonzero(y_train)
zeros_train = np.where(y_train[:,20:532,20:532]==0)
for i in zeros_train[1:]:
    i += 20
nonzero_coords_train = zip(nonzeros_train[0],nonzeros_train[1],nonzeros_train[2])
zero_coords_train = zip(zeros_train[0],zeros_train[1],zeros_train[2])

nonzeros_test = np.nonzero(y_test)
zeros_test = np.where(y_test[:,20:532,20:532]==0)
for i in zeros_test[1:]:
    i += 20
nonzero_coords_test = zip(nonzeros_test[0],nonzeros_test[1],nonzeros_test[2])
zero_coords_test = zip(zeros_test[0],zeros_test[1],zeros_test[2])

#conv net settings
convolutional_layers = 6
feature_maps = [1,40,40,40,80,80,80]
filter_shapes = [(5,5),(5,5),(3,3),(3,3),(3,3),(3,3)]
feedforward_layers = 1
feedforward_nodes = [2000]
classes = 1

class convolutional_layer(object):
    def __init__(self, input, output_maps, input_maps, filter_height, filter_width, maxpool=None):
        self.input = input
        self.w = theano.shared(self.ortho_weights(output_maps,input_maps,filter_height,filter_width),borrow=True)
        self.b = theano.shared(np.zeros((output_maps,), dtype=theano.config.floatX),borrow=True)
        self.conv_out = conv2d(input=self.input, filters=self.w, border_mode='half')
        if maxpool:
            self.conv_out = downsample.max_pool_2d(self.conv_out, ds=maxpool, ignore_border=True)
        self.output = T.nnet.elu(self.conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))
    def ortho_weights(self,chan_out,chan_in,filter_h,filter_w):
        bound = np.sqrt(6./(chan_in*filter_h*filter_w + chan_out*filter_h*filter_w))
        W = np.random.random((chan_out, chan_in * filter_h * filter_w))
        u, s, v = np.linalg.svd(W,full_matrices=False)
        if u.shape[0] != u.shape[1]:
            W = u.reshape((chan_out, chan_in, filter_h, filter_w))
        else:
            W = v.reshape((chan_out, chan_in, filter_h, filter_w))
        return W.astype(theano.config.floatX)
    def get_params(self):
        return self.w,self.b

class feedforward_layer(object):
    def __init__(self,input,features,nodes):
        self.input = input
        self.bound = np.sqrt(1.5/(features+nodes))
        self.w = theano.shared(self.ortho_weights(features,nodes),borrow=True)
        self.b = theano.shared(np.zeros((nodes,), dtype=theano.config.floatX),borrow=True)
        self.output = T.nnet.sigmoid(-T.dot(self.input,self.w)-self.b)
    def ortho_weights(self,fan_in,fan_out):
        bound = np.sqrt(2./(fan_in+fan_out))
        W = np.random.randn(fan_in,fan_out)*bound
        u, s, v = np.linalg.svd(W,full_matrices=False)
        if u.shape[0] != u.shape[1]:
            W = u
        else:
            W = v
        return W.astype(theano.config.floatX)
    def get_params(self):
        return self.w,self.b        

class neural_network(object):
    def __init__(self,convolutional_layers,feature_maps,filter_shapes,feedforward_layers,feedforward_nodes,classes):
        self.input = T.tensor4()        
        self.convolutional_layers = []
        self.convolutional_layers.append(convolutional_layer(self.input,feature_maps[1],feature_maps[0],filter_shapes[0][0],filter_shapes[0][1]))
        for i in range(1,convolutional_layers):
            if i==3:
                self.convolutional_layers.append(convolutional_layer(self.convolutional_layers[i-1].output,feature_maps[i+1],feature_maps[i],filter_shapes[i][0],filter_shapes[i][1],maxpool=(2,2)))
            else:
                self.convolutional_layers.append(convolutional_layer(self.convolutional_layers[i-1].output,feature_maps[i+1],feature_maps[i],filter_shapes[i][0],filter_shapes[i][1]))
        self.feedforward_layers = []
        self.feedforward_layers.append(feedforward_layer(self.convolutional_layers[-1].output.flatten(2),32000,feedforward_nodes[0]))
        for i in range(1,feedforward_layers):
            self.feedforward_layers.append(feedforward_layer(self.feedforward_layers[i-1].output,feedforward_nodes[i-1],feedforward_nodes[i]))
        self.output_layer = feedforward_layer(self.feedforward_layers[-1].output,feedforward_nodes[-1],classes)
        self.params = []
        for l in self.convolutional_layers + self.feedforward_layers:
            self.params.extend(l.get_params())
        self.params.extend(self.output_layer.get_params())
        self.target = T.matrix()
        self.output = self.output_layer.output
        self.cost = -self.target*T.log(self.output)-(1-self.target)*T.log(1-self.output)
        self.cost = self.cost.mean()
        self.updates = self.adam(self.cost, self.params)
        self.propogate = theano.function([self.input,self.target],self.cost,updates=self.updates,allow_input_downcast=True)
        self.classify = theano.function([self.input],self.output,allow_input_downcast=True)
        
    def adam(self, cost, params, lr=0.0002, b1=0.1, b2=0.01, e=1e-8):
        updates = []
        grads = T.grad(cost, params)
        self.i = theano.shared(np.float32(0.))
        i_t = self.i + 1.
        fix1 = 1. - (1. - b1)**i_t
        fix2 = 1. - (1. - b2)**i_t
        lr_t = lr * (T.sqrt(fix2) / fix1)
        for p, g in zip(params, grads):
            self.m = theano.shared(p.get_value() * 0.)
            self.v = theano.shared(p.get_value() * 0.)
            m_t = (b1 * g) + ((1. - b1) * self.m)
            v_t = (b2 * T.sqr(g)) + ((1. - b2) * self.v)
            g_t = m_t / (T.sqrt(v_t) + e)
            p_t = p - (lr_t * g_t)
            updates.append((self.m, m_t))
            updates.append((self.v, v_t))
            updates.append((p, p_t))
        updates.append((self.i, i_t))
        return updates
        
    def train(self,X,y,batch_size):
        input = []
        target = []
        for i in range(batch_size):
            if random.random() < .5:
                (a,b,c) = random.choice(nonzero_coords_train)
                input.append(X[a,b-20:b+20,c-20:c+20])
                target.append(y[a,b,c])
            else:
                (a,b,c) = random.choice(zero_coords_train)
                input.append(X[a,b-20:b+20,c-20:c+20])
                target.append(y[a,b,c])
        input = np.array(input).reshape(batch_size,1,40,40)
        target = np.array(target).reshape(len(target),1)
        return self.propogate(input,target) 
        
    def predict(self,X,y,batch_size):
        input = []
        target = []
        for i in range(batch_size):
            if random.random() < .5:
                (a,b,c) = random.choice(nonzero_coords_test)
                input.append(X[a,b-20:b+20,c-20:c+20])
                target.append(y[a,b,c])
            else:
                (a,b,c) = random.choice(zero_coords_test)
                input.append(X[a,b-20:b+20,c-20:c+20])
                target.append(y[a,b,c])
        input = np.array(input).reshape(batch_size,1,40,40)
        target = np.array(target).reshape(len(target),1)
        prediction = self.classify(input)
        label = np.around(prediction)
        error = 1-float(np.sum(label==target))/len(label)
        return prediction,error

#train
print "building neural network"
nn = neural_network(convolutional_layers,feature_maps,filter_shapes,feedforward_layers,feedforward_nodes,classes)

batch_size = 100

for i in range(25000):
    cost = nn.train(X_train,y_train,batch_size)
    sys.stdout.write("step %i loss: %f \r" % (i+1, cost))
    sys.stdout.flush()
    if (i+1)%100 == 0:
        pred,error = nn.predict(X_test,y_test,batch_size)
        print "test error at iteration %i: %.4f" % (i+1,error)