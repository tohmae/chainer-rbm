#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
 Restricted Boltzmann Machine (RBM)
 References :
   - Y. Bengio, P. Lamblin, D. Popovici, H. Larochelle: Greedy Layer-Wise
   Training of Deep Networks, Advances in Neural Information Processing
   Systems 19, 2007
   - DeepLearningTutorials
   https://github.com/lisa-lab/DeepLearningTutorials
"""

import os,sys
import argparse
import time

import numpy as np

import chainer
from chainer import computational_graph
from chainer import cuda,Variable
import chainer.links as L
import chainer.optimizers as O

import pickle
import gzip

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')

parser.add_argument('--test', '-t', choices=['simple', 'mnist'],
                    default='mnist',
                    help='test type ("simple", "mnist")')


''' GPUで計算する時は、cupy = numpy + GPUで処理する。 '''
args = parser.parse_args()
if args.gpu >= 0:
    cuda.check_cuda_available()
xp = cuda.cupy if args.gpu >= 0 else np

xp.seterr(all='ignore')

def sigmoid(x):
    return 1. / (1 + xp.exp(-x))

class RBM(object):
    """
    chainerで使用する場合、link.linearが出力層*入力層でデータを扱うので
    RBM.W.Tをchainerの初期値で使用する必要あり!!!
    n_visbile:可視層の次元
    n_hidden:隠れ層の次元
    W:重み行列(事前学習に必要)
    hbias:隠れ層のbias(事前学習に必要)
    vbaias:可視層のbias
    vbinaryflag: 可視層が0,1か実数なのかで場合分け。 初期層は実数を想定しているが、中間層は0,1しか考えていない。
    lr:学習係数
    k:CDの回数
    sample_flag:確率分布を元にデータをサンプリング(0,1)
    pcd_flag: CD or PCD
    lamda: 重さ減衰
    mu: モーメンタム
    chainerで h = l(v)とかけるように一般的なRBMに出てくる重み行列を転置している。
    """
    def __init__(self, n_visible=2, n_hidden=3, \
        W=None, hbias=None, vbias=None, xp_rng=None, \
        vbinaryflag=0,lr=0.1,k=1,sample_flag=1,pcd_flag=0,lamda=0.0,mu=0.0):

        self.n_visible = n_visible  # num of units in visible (input) layer
        self.n_hidden = n_hidden    # num of units in hidden layer
        self.vbinaryflag = vbinaryflag
        self.lr = lr
        self.k = k
        self.sample_flag = sample_flag
        self.pcd_flag = pcd_flag
        self.lamda = lamda
        self.mu = mu

        if xp_rng is None:
            xp_rng = xp.random.RandomState(1234)


        if W is None:
            a = 1. / n_visible
            initial_W = xp.array(xp_rng.uniform(  # initialize W uniformly
                low=-a,
                high=a,
                size=(n_visible, n_hidden)))

            W = initial_W

        if hbias is None:
            hbias = xp.zeros(n_hidden)  # initialize h bias 0

        if vbias is None:
            vbias = xp.zeros(n_visible)  # initialize v bias 0


        self.xp_rng = xp_rng
        self.W = W
        self.hbias = hbias
        self.vbias = vbias

        self.deltaW = xp.zeros(shape=(n_visible,n_hidden))
        self.deltavbias = xp.zeros(n_visible)
        self.deltahbias = xp.zeros(n_hidden)

        # self.params = [self.W, self.hbias, self.vbias]

    def backward(self,v_data,v_prev_data=None):
        lr = self.lr
        k = self.k

        if self.pcd_flag == 0:
            v_prev_data = v_data
        elif self.pcd_flag ==1 and v_prev_data is None:
            v_prev_data = v_data

        h_data = self.propup(v_data)
        nv_data,nh_data = self.contrastive_divergence(v_prev_data,k)

        batchsize = v_data.shape[0]
        deltaW = (xp.dot(v_data.T, h_data) - xp.dot(nv_data.T, nh_data)) / batchsize
        deltavbias = xp.mean(v_data - nv_data, axis=0)
        deltahbias = xp.mean(h_data - nh_data, axis=0)

        self.W += lr * (deltaW - self.lamda * self.W) + self.mu * deltaW
        self.vbias += lr * deltavbias + self.mu * deltavbias
        self.hbias += lr * deltahbias + self.mu * deltahbias

        self.deltaW = deltaW
        self.deltavbias = deltavbias
        self.deltahbias = deltahbias

        '''
        self.deltaW = xp.dot(v_data.T, h_data) - xp.dot(nv_data.T, nh_data)
        self.deltavbias = xp.mean(v_data - nv_data, axis=0)
        self.deltahbias = xp.mean(h_data - nh_data, axis=0)
        '''
    def contrastive_divergence(self,v_data,k):
        ''' CD-k '''
        ph_mean, ph_sample = self.sample_h_given_v(v_data)
        chain_start = ph_sample
        for step in range(k):
            if step == 0:
                nv_means, nv_samples,\
                nh_means, nh_samples = self.gibbs_hvh(chain_start)
            else:
                nv_means, nv_samples,\
                nh_means, nh_samples = self.gibbs_hvh(nh_samples)
        nv_data = nv_samples
        nh_data = nh_means

        return [nv_data,nh_data]


    def sample_h_given_v(self, v0_sample):
        '''
        v0_sample → h1_sampleをgibbs sampling
        [0,1]の値をrandom作成して、popupで計算した1となる確率と比較
        h1_mean[i] >  p[i] なら h1_sample[i] = 1
        h1_mean[i] <= p[i] なら h1_sample[i] = 0
        '''
        sample_flag = self.sample_flag
        h1_mean = self.propup(v0_sample)
        if sample_flag == 0:
            return [h1_mean,h1_mean]
        h1_sample = self.xp_rng.binomial(size=h1_mean.shape,   # discrete: binomial
                                       n=1,
                                       p=h1_mean)
        return [h1_mean, h1_sample]

    def sample_v_given_h(self, h0_sample):
        '''
        h0_sample → v1_sampleをgibbs sampling
        可視層が[0,1]の場合
        [0,1]の値をrandom値 p[i]を作成して、popdownで計算した1となる確率と比較
        v1_mean[i] >  p[i] なら v1_sample[i] = 1
        v1_mean[i] <= p[i] なら v1_sample[i] = 0
        可視層が実数の場合
        v1_sample[i]は平均v1_mean[i]、分散1の正規分布となるので
        平均0、分散1の正規分布からrandom値 u[i]を作成して、popdownで計算した値を追加
        v1_sample[i] = v1_mean[i] + u[i]
        '''
        sample_flag = self.sample_flag
        v1_mean = self.propdown(h0_sample)
        if sample_flag == 0:
            return [v1_mean,v1_mean]
        if self.vbinaryflag ==0:
            v1_sample = self.xp_rng.binomial(size=v1_mean.shape,   # discrete: binomial
                                                n=1,
                                                p=v1_mean)
        else:
            batch_number = h0_sample.shape[0]
            v1_sample = v1_mean + self.xp_rng.randn(batch_number,self.n_visible)
        return [v1_mean, v1_sample]

    def propup(self, v):
        '''
        可視層データから隠れ層の確率分布を計算
        隠れ層は[0,1]なので1となる確率を返す。
        入力値visは必ずxp(np)
        '''
        pre_sigmoid_activation = xp.dot(v, self.W) + self.hbias
        return sigmoid(pre_sigmoid_activation)

    def propdown(self, h):
        '''
        隠れ層データから可視層の確率分布を計算
        可視層が[0,1]の時は1となる確率を返す。
        可視層が実数の時は正規分布の平均を返す。分散は1固定。
        入力値hidは必ずxp(np)
        '''
        vbinaryflag = self.vbinaryflag
        if vbinaryflag == 0:
            pre_sigmoid_activation = xp.dot(h, self.W.T) + self.vbias
            v_mean = sigmoid(pre_sigmoid_activation)
        else:
            v_mean = xp.dot(h, self.W.T) + self.vbias
        return v_mean

    def gibbs_hvh(self, h0_sample):
        ''' h->v->hをgibbs sampling '''
        v1_mean, v1_sample = self.sample_v_given_h(h0_sample)
        h1_mean, h1_sample = self.sample_h_given_v(v1_sample)
        return [v1_mean, v1_sample,
                h1_mean, h1_sample]

    def get_reconstruction_cross_entropy(self,v_test):
        pre_sigmoid_activation_h = xp.dot(v_test, self.W) + self.hbias
        sigmoid_activation_h = sigmoid(pre_sigmoid_activation_h)

        pre_sigmoid_activation_v = xp.dot(sigmoid_activation_h, self.W.T) + self.vbias
        sigmoid_activation_v = sigmoid(pre_sigmoid_activation_v)

        cross_entropy =  - xp.mean(
            xp.sum(v_test * xp.log(sigmoid_activation_v) +
            (1 - v_test) * xp.log(1 - sigmoid_activation_v),
                      axis=1))

        return cross_entropy

    def reconstruct(self, v):
        h = sigmoid(xp.dot(v, self.W) + self.hbias)
        reconstructed_v = sigmoid(xp.dot(h, self.W.T) + self.vbias)
        return reconstructed_v

    def get_cost(self,v_test):
        k = self.k
        h_test,vh_test,nh_test = self.contrastive_divergence(v_test,k)
        loss = self.free_energy(v_test) - self.free_energy(vh_test)
        return loss

    def free_energy(self,v_test):
        ''' Function to compute the free energy '''
        wx_b = xp.dot(v_test, self.W) + self.hbias
        if self.vbinaryflag== 0:
            vbias_term = xp.sum(xp.dot(v_test, self.vbias.T))
        else:
            v_ = v_test - self.vbias
            vbias_term = xp.sum(0.5 * v_ * v_)

        hidden_term = xp.sum(xp.log(1 + xp.exp(wx_b)))

        return -hidden_term - vbias_term

def test_rbm(training_epochs=1000):
    l_W = xp.zeros(shape=(training_epochs,6,2))
    l_vbias = xp.zeros(shape=(training_epochs,6))
    l_hbias = xp.zeros(shape=(training_epochs,2))
    data = xp.array([[1,1,1,0,0,0],
                        [1,0,1,0,0,0],
                        [1,1,1,0,0,0],
                        [0,0,1,1,1,0],
                        [0,0,1,1,0,0],
                        [0,0,1,1,1,0]])
    rng = xp.random.RandomState(123)

    # construct RBM
#    rbm = RBM(n_visible=6, n_hidden=2, xp_rng=rng,vbinaryflag=0,lr=0.1,k=1,sample_flag=1)
    rbm = RBM(n_visible=6, n_hidden=2, xp_rng=rng,vbinaryflag=0,lr=0.1,k=1,sample_flag=1,pcd_flag=1)
#    rbm = RBM(n_visible=6, n_hidden=2, xp_rng=rng,vbinaryflag=1,lr=0.1,k=1,sample_flag=1)
    # train

    for epoch in range(training_epochs):
        print("epoch=" + str(epoch))
        v_data = data
        rbm.backward(v_data,v_data)
        '''
        print("    W=" + str(rbm.W))
        print("    vbias=" + str(rbm.vbias))
        print("    hbias=" + str(rbm.hbias))
        '''
#        cross_entry = rbm.get_reconstruction_cross_entropy(v_data)
#        loss = rbm.get_cost(v_data)
#        print("    cross_entry", cross_entry)
#        print("    loss", loss)
        l_W[epoch] = rbm.W
        l_vbias[epoch] = rbm.vbias
        l_hbias[epoch] = rbm.hbias
    result = {}
    result["W"] = l_W
    result["vbias"] = l_vbias
    result["hbias"] = l_hbias

    with open("result.test.pkl","wb") as w_f:
        pickle.dump(result,w_f)

    # test
    v = xp.array([[0, 0, 0, 1, 1, 0],
                     [1, 1, 0, 0, 0, 0]])

    print(rbm.reconstruct(v))

def test_mnist():
    # Load the dataset
    dataset = 'mnist.pkl.gz'
    f = gzip.open(dataset, 'rb')
    train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
    f.close()

    xp.set_printoptions(linewidth=1000,threshold=1000,precision=3)
    x_train,y_train = train_set

    x_train = xp.where(x_train>0.5,1,0)

    N = y_train.size
#    batchsize = 100
    batchsize = 10
    n_epoch = 50
    n_visible= 784
    n_hidden = 500

    l_W = xp.zeros(shape=(n_epoch,n_visible,n_hidden))
    l_vbias = xp.zeros(shape=(n_epoch,n_visible))
    l_hbias = xp.zeros(shape=(n_epoch,n_hidden))

    k = 1
    # construct RBM
    rng = xp.random.RandomState(123)
    rbm = RBM(n_visible=n_visible, n_hidden=n_hidden,xp_rng=rng,vbinaryflag=0,lr=0.01,k=k,sample_flag=1,pcd_flag=0,lamda=0,mu=0.5)

    v_prev_data = None
    for epoch in range(0,n_epoch):
        print('epoch', epoch+1)
        sys.stdout.flush()
        begin_time = time.time()
        perm = xp.random.permutation(N)
        for i in range(0,N,batchsize):
            v_data = xp.asarray(x_train[perm[i:i+batchsize]])
            rbm.backward(v_data,v_prev_data)
            v_prev_data = v_data
#        cross_entry = rbm.get_reconstruction_cross_entropy(x_train)
#        loss = rbm.get_cost(x_train)
#        print("    cross_entry", cross_entry)
#        print("    loss", loss)
        l_W[epoch] = rbm.W
        l_vbias[epoch] = rbm.vbias
        l_hbias[epoch] = rbm.hbias
        end_time = time.time()
        duration = end_time - begin_time
        print('    {:.2f} sec'.format(duration))
        sys.stdout.flush()

    result = {}
    result["W"] = l_W
    result["vbias"] = l_vbias
    result["hbias"] = l_hbias

    with open("result.pkl","wb") as w_f:
        pickle.dump(result,w_f)



if __name__ == "__main__":
    if args.test == 'simple':
        test_rbm()
    elif args.test == 'mnist':
        test_mnist()
