# -*- coding: utf-8 -*-

import os,sys
import argparse
import time

import numpy as np

import chainer
from chainer import computational_graph
from chainer import cuda, Variable

import chainer.functions as F
import chainer.links as L
import chainer.optimizers as O

xp = np

def set_gpuflag(gpu):
    global xp 
    xp = cuda.cupy if gpu >= 0 else np
    print("RBM:GPU=" + str(gpu))


import pickle
import gzip

def sigmoid(x):
    return 1. / (1 + xp.exp(-x))

class RBM(chainer.Chain):
    '''
    n_visbile:可視層の次元
    n_hidden:隠れ層の次元
    l.W:重み行列
    l.b:hbias
    l.a:vbaias
    binomail: 可視層が0,1か実数なのかで場合分け。 初期層は実数を想定しているが、中間層は0,1しか考えていない。
    chainerで h = l(v)とかけるように一般的なRBMに出てくる重み行列を転置している。
    '''
    def __init__(self,n_visible,n_hidden,binomial=1,k=1,pcd_flag=0):
        super(RBM,self).__init__(
            l=L.Linear(n_visible,n_hidden),
            # chainer 1.5のmanualにこの使い方は非推奨とあったのでadd_param使用。
#            a=L.Parameter(xp.zeros(n_visible,dtype=xp.float32)),
        )
        self.l.add_param("a",(n_visible),dtype=xp.float32)

        ''' l.aを初期化、chainerの使用で初期化しないと学習しない？ '''
        self.l.a.data.fill(0)

        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.binomial = binomial
        self.k = k
        self.pcd_flag = pcd_flag

    def __call__(self,v_data,v_prev_data=None):

        batch_size = v_data.shape[0]

        if self.pcd_flag == 0:
            v_prev_data = v_data
        elif self.pcd_flag ==1 and v_prev_data is None:
            v_prev_data = v_data
        vh_data = self.constrastive_divergence(v_prev_data,self.k)
        v = Variable(v_data)
        vh = Variable(vh_data.astype(np.float32))
        '''
        http://deeplearning.net/tutorial/rbm.htmlの式(5)から
        誤差関数(loss)は(可視層のデータのfree energy)と(可視層のデータのCD-kのfree energy)の差となる。
        loss = (self.free_energy(v) - self.free_energy(vh)) / batch_size
        v->vhへの写像は学習に入れないため、CD-kの処理完了後にVariable化
        '''
        loss = (self.free_energy(v) - self.free_energy(vh)) / batch_size
        return loss

    def free_energy(self,v):
        ''' Function to compute the free energy '''
        '''
        入力値vは必ずVariable化
        本来行単位でSUMを取ってから列単位(バッチ数)でSUMを取るべき処理だけど
        結局SUMを取るので一括でSUM
        '''
        batch_size = v.data.shape[0]
        n_visible = self.n_visible
        if self.binomial == 1:
            '''
            可視層が[0,1]のとき
            vbias_term = -1 * SUM((a(i) * v(i))
            '''
            vbias_term = F.sum(F.matmul(v,self.l.a))
        else:
            '''
            可視層が実数のとき
            vbias_term = -0.5 * SUM((v(i)-a(i)) * (v(i)-a(i)))
            chainerではバッチ数*入力層で扱うため、各行からbiasを引くため
            無理やり m＊nを使用
            '''
            m = Variable(xp.ones((batch_size,1),dtype=xp.float32))
            n = F.reshape(self.l.a,(1,n_visible))
            v_ = v - F.matmul(m,n)
            vbias_term =  - F.sum(0.5 * v_ * v_)

        wx_b = self.l(v)
        hidden_term = F.sum(F.log(1+F.exp(wx_b)))
        return -vbias_term-hidden_term

    def propup(self,vis):
        '''
        可視層データから隠れ層の確率分布を計算
        隠れ層は[0,1]なので1となる確率を返す。
        入力値visは必ずxp(np)
        '''
        pre_sigmoid_activation = xp.dot(vis,self.l.W.data.T) + self.l.b.data
        return sigmoid(pre_sigmoid_activation)

    def propdown(self,hid):
        '''
        隠れ層データから可視層の確率分布を計算
        可視層が[0,1]の時は1となる確率を返す。
        可視層が実数の時は正規分布の平均を返す。分散は1固定。
        入力値hidは必ずxp(np)
        '''
        if self.binomial == 1:
            pre_sigmoid_activation = xp.dot(hid,self.l.W.data) + self.l.a.data
            v_mean = sigmoid(pre_sigmoid_activation)
        else:
            v_mean = xp.dot(hid,self.l.W.data) + self.l.a.data
        return v_mean

    def sample_h_given_v(self,v0_sample):
        '''
        v0_sample → h1_sampleをgibbs sampling
        [0,1]の値をrandom作成して、popupで計算した1となる確率と比較
        h1_mean[i] >  p[i] なら h1_sample[i] = 1
        h1_mean[i] <= p[i] なら h1_sample[i] = 0
        '''
        h1_mean = self.propup(v0_sample)
        if xp == np:
            h1_sample = xp.random.binomial(size=h1_mean.shape,n=1,p=h1_mean)
        else:
            z = xp.random.random(size=h1_mean.shape,dtype=xp.float32)
            h1_sample = xp.where(h1_mean>z,1,0)
        return h1_mean,h1_sample

    def sample_v_given_h(self,h0_sample):
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


        v1_mean = self.propdown(h0_sample)
        if self.binomial == 1:
            if xp == np:
                v1_sample = xp.random.binomial(size=v1_mean.shape,n=1,p=v1_mean)
            else:
                z = xp.random.random(size=v1_mean.shape,dtype=xp.float32)
                v1_sample = xp.where(v1_mean>z,1,0)
        else:
            batch_number = h0_sample.shape[0]
            v1_sample = v1_mean + xp.random.randn(batch_number,self.n_visible)
        return v1_mean,v1_sample

    def gibbs_hvh(self,h0_sample):
        ''' h->v->hをgibbs sampling '''
        v1_mean,v1_sample = self.sample_v_given_h(h0_sample)
        h1_mean,h1_sample = self.sample_h_given_v(v1_sample)
        return v1_mean,v1_sample,h1_mean,h1_sample

    def gibbs_vhv(self,v0_sample):
        ''' v->h->vをgibbs sampling '''
        h1_mean,h1_sample = self.sample_h_given_v(v0_sample)
        v1_mean,v1_sample = self.sample_v_given_h(h1_sample)
        return h1_mean,h1_sample,v1_mean,v1_sample

    def constrastive_divergence(self,v0_sample,k=1):
        ''' CD-kの処理、GPU化させるにはcupy必要 '''
        vh_sample = v0_sample
        for step in range(k):
            ph_mean,ph_sample,vh_mean,vh_sample = self.gibbs_vhv(vh_sample)
        return vh_sample

    def reconstruct(self, v):
        h = sigmoid(xp.dot(v,self.l.W.data.T) + self.l.b.data)
        reconstructed_v = sigmoid(xp.dot(h,self.l.W.data) + self.l.a.data)
        return reconstructed_v

def test_rbm(training_epochs=1000):

    binomial = 1 
    model = RBM(6,3,binomial=binomial)
#    print("W="  + str(model.l.W.data))
#    print("a="  + str(model.l.a.data))
#    print("b="  + str(model.l.b.data))
    data = xp.array([[1,1,1,0,0,0],
                        [1,0,1,0,0,0],
                        [1,1,1,0,0,0],
                        [0,0,1,1,1,0],
                        [0,0,1,1,0,0],
                        [0,0,1,1,1,0]]
                      ,dtype=xp.float32)

#    optimizer = O.Adam()
#    optimizer = O.SGD(lr=0.1)
    optimizer = O.MomentumSGD(lr=0.01, momentum=0.5)
    optimizer.setup(model)

    for i in range(training_epochs):
        model.zerograds()
        loss = model(data)
        loss.backward()
        if i == 0:
            dotfile = 'graph.dot'
            with open(dotfile, 'w') as o:
                g = computational_graph.build_computational_graph(
                        (loss, ), remove_split=True)
                o.write(g.dump())
            print('graph generated')

        optimizer.update()

    # test
    print("W",model.l.W.data)
    print("b",model.l.b.data)
    print("a",model.l.a.data)
    v = xp.array([[0, 0, 0, 1, 1, 0],
                     [1, 1, 0, 0, 0, 0]])

    print(model.reconstruct(v))

def test_mnist():
    # Load the dataset
    dataset = 'mnist.pkl.gz'
    f = gzip.open(dataset, 'rb')
    train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
    f.close()

    np.set_printoptions(linewidth=1000,threshold=1000,precision=3)
    x_train,y_train = train_set

    if args.binomial == 1:
        x_train = np.where(x_train>0.5,1,0)
    x_train = x_train.astype(xp.float32)

    N = y_train.size
    batchsize = 10
#    batchsize = 10
    n_epoch = 50
#    n_epoch = 1
    n_visible= 784
    n_hidden = 500

    l_W = np.zeros(shape=(n_epoch,n_visible,n_hidden))
    l_vbias = np.zeros(shape=(n_epoch,n_visible))
    l_hbias = np.zeros(shape=(n_epoch,n_hidden))

    pcd_flag = args.pcd
    k = args.kcd
    binomial = args.binomial

    print("gpu=",args.gpu)
    print("pcd_flag=",pcd_flag)
    print("k=",k)
    print("binomial=",binomial)

    if binomial == 0:
        avg_mean = np.mean(x_train,axis=0)
        avg_std = np.std(x_train,axis=0)
        avg_std2 = np.where(avg_std<0.4,0.4,avg_std)

        print("before:max=",np.max(x_train))
        print("before:min=",np.min(x_train))
        x_train = (x_train - avg_mean) / avg_std2
        print("after:max=",np.max(x_train))
        print("afrer:min=",np.min(x_train))

    # construct RBM
    model = RBM(n_visible=n_visible, n_hidden=n_hidden,binomial=binomial,k=k,pcd_flag=pcd_flag)

    if args.gpu >= 0:
        model.to_gpu()

    optimizer = O.MomentumSGD(lr=0.01, momentum=0.5)
    optimizer.setup(model)

    v_prev_data = None
    for epoch in range(0,n_epoch):
        print('epoch', epoch+1)
        sys.stdout.flush()
        begin_time = time.time()
        perm = np.random.permutation(N)
        for i in range(0,N,batchsize):
            v_data = xp.asarray(x_train[perm[i:i+batchsize]])
            model.zerograds()
            loss = model(v_data,v_prev_data)
            loss.backward()
            optimizer.update()
            v_prev_data = v_data
            if i == 0 and epoch==0:
                dotfile = 'graph.dot'
                with open(dotfile, 'w') as o:
                    g = computational_graph.build_computational_graph(
                            (loss, ), remove_split=True)
                    o.write(g.dump())
                print('graph generated')
#        print(model.l.a.data)
        if args.gpu >= 0:
            model.to_cpu()
        l_W[epoch] = model.l.W.data.T
        l_vbias[epoch] = model.l.a.data
        l_hbias[epoch] = model.l.b.data
        if args.gpu >= 0:
            model.to_gpu()
        end_time = time.time()
        duration = end_time - begin_time
        print('    {:.2f} sec'.format(duration))
        sys.stdout.flush()

    result = {}
    result["W"] = l_W
    result["vbias"] = l_vbias
    result["hbias"] = l_hbias

    if binomial == 0:
        result["avg_mean"] = avg_mean
        result["avg_std"] = avg_std
        result["avg_std2"] = avg_std2

    if pcd_flag == 1:
        subname = "pcd" + str(k) + ".binomial" + str(binomial)
    else:
        subname = "cd" + str(k) + ".binomial" + str(binomial)
    result_file_name = "result.chainer." + subname + ".pkl"

    with open(result_file_name,"wb") as w_f:
        pickle.dump(result,w_f)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--test', '-t', choices=['simple', 'mnist'],
                    default='mnist',
                    help='test type ("simple", "mnist")')
    parser.add_argument('--pcd', '-p', default=1, type=int,
                    help='pcd_flag')
    parser.add_argument('--kcd', '-k', default=1, type=int,
                    help='cd-k')
    parser.add_argument('--binomial', '-b', default=1, type=int,
                    help='binomial')

    args = parser.parse_args()
    ''' GPUで計算する時は、cupy = numpy + GPUで処理する。 '''
    if args.gpu >= 0:
        cuda.check_cuda_available()
        cuda.get_device(args.gpu).use()
    xp = cuda.cupy if args.gpu >= 0 else np

    parser = argparse.ArgumentParser()
    if args.test == 'simple':
        test_rbm()
    elif args.test == 'mnist':
        test_mnist()


