import numpy as np
import tensorflow as tf
from argparse import ArgumentParser

import model_began as md
from layers import LpLoss
from image_dataloader import ImageDataLoaderPrefetch, ImageTransformer
from test_gan_util import gen_seed
DTYPE = tf.float32

class SolverBEGAN(object):
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.gamma = gamma

    def setup(self, args):
        self.gen = md.GeneratorBEGAN(args.cdim, args.bsize, args.ichn, args.isize, args.idepth, args.repeat,
                args.chn, args.ksize, args.stride, filler, deconv=args.deconv)
        self.dis = md.DiscriminatorBEGAN(args.cdim, args.bsize, args.ichn, args.isize, args.idepth, args.repeat,
                args.chn, args.ksize, args.stride, args.ochn, args.osize, args.odepth,
                filler, deconv=args.deconv)
        
        # rand vector
        rand_shape = self.gen.inputs[0].shape.as_list()
        self.z = tf.placeholder(DTYPE, shape=rand_shape, name='z')
        # real data
        image_shape = self.dis.inputs[0].shape.as_list()
        self.x_real = tf.placeholder(DTYPE, shape=image_shape, name='x_real')
        # control factor and learning rate
        self.k = tf.placeholder(DTTYE, name='k')
        self.lr = tf.placeholder(DTYPE, name='lr')
        # generated data
        self.x_fake = self.gen.run(self.z)[0]
        # reconstructed data
        _, self.y_real = self.dis.run(self.x_real)
        _, self.y_fake = self.dis.run(self.x_fake)
        
        # loss
        self.loss_real, _ = LpLoss(self.x_real, self.y_real, p=args.lp)
        self.loss_fake, _ = LpLoss(self.x_fake, self.y_fake, p=args.lp)
        self.loss_dis = self.loss_real[0] - self.k*self.loss_fake[0]
        self.loss_gen = self.loss_fake[0]
        self.measure = self.loss_dis + tf.abs(self.gamma*self.loss_real[0] - self.loss_fake[0])

        # Optimizer
        self.opt_gen = tf.train.AdamOptimizer(self.lr, self.mm).minimize(self.loss_gen, var_list=self.gen.params)
        self.opt_dis = tf.train.AdamOptimizer(self.lr, self.mm).minimize(self.loss_dis, var_list=self.dis.params)

        # initialize
        self.sess.run(tf.global_variables_initializer())

        # saver
        self.saver = tf.train.Saver(max_to_keep=(self.max_to_keep))

        try:
            self.load(self.sess, self.saver, self.save_folder)
        except:
            self.saver.save(self.sess, self.model_name, write_meta_graph=True)

        # summary
        if self.flag:
            tf.summary.scalar('loss_gen', self.loss_gen)
            tf.summary.scalar('loss_dis', self.loss_dis)
            tf.summary.scalar('loss_real', self.loss_real[0])
            tf.summary.scalar('loss_fake', self.loss_fake[0])
            tf.summary.scalar('k', self.k)
            tf.summary.scalar('measure', self.measure)
            self.merged = tf.summary.merge_all()
            self.writer = tf.summary.FileWriter(self.project_dir, self.sess.graph)
        

def main(args):

    input_name = dis.inputs[0]
    input_shape = [args.bsize, args.isize, args.isize, args.ichn]
    seed = gen_seed()
    itf = ImageTransformer({input_name: input_shape}, seed=seed.next())
    idl = ImageDataLoaderPrefetch(args.qsize, args.imfd, args.imnm, seed=seed.next())
    idl.add_prefetch_process(input_name, input_shape)
    
    

def get_parser():
    ps = ArgumentParser()
    ps.add_argument('--imfd', type=str)
    ps.add_argument('--imnm', type=str)
    ps.add_argument('--cdim', type=int, default=128)
    ps.add_argument('--bsize', type=int, default=16)
    ps.add_argument('--ichn', type=int, default=3)
    ps.add_argument('--isize', type=int, default=64)
    ps.add_argument('--idepth', type=int, default=4)
    ps.add_argument('--ochn', type=int, default=-1)
    ps.add_argument('--osize', type=int, default=-1)
    ps.add_argument('--odepth', type=int, default=-1)
    ps.add_argument('--repeat', type=int, default=2)
    ps.add_argument('--chn', type=int, default=16)
    ps.add_argument('--ksize', type=int, default=3)
    ps.add_argument('--stride', type=int, default=2)
    ps.add_argument('--gstd', type=float, default=-1)
    ps.add_argument('--deconv', action='store_true', default=False)
    ps.add_argument('--qsize', type=int, default=4)
    ps.add_argument('--lp', type=int, default=1, choices=[1,2])
    return ps

if __name__ == '__main__':
    ps = get_parser()
    args = ps.parse_args()
    if args.ochn == -1:
        args.ochn = args.ichn
    if args.osize == -1:
        args.osize = args.isize
    if args.odepth == -1:
        args.odepth = args.idepth
    filler = ('msra', 0., 1.) if args.gstd == -1 else ('gaussian', 0., args.gstd)

    main(args)
