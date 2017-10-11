# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 13:54:38 2017

@author: shiwu_001
"""

import os.path as osp
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def gen_seed(n=10000000):
    i = 37
    while i < n:
        yield i
        i += 100
        
def get_plotable_data(data, input_format='HWC'):
    data = np.clip(data, 0, 255)
    if input_format == 'CHW':
        data = data.swapaxes(0,1).swapaxes(1,2)
    data = np.require(data, dtype=np.uint8)
    return data

def get_sgn(score, thres):
    return 'green' if score > thres else 'red'
    
def display_samples_encoder(samples, save_path,
                            stride=2, font_size=10, text_height=12, mid_gap=16,
                            font_name='/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf'):
    real_images = samples[0][0]
    real_images_rec = samples[0][1]
    real_scores = samples[0][2]
    gen_images = samples[1][0]
    gen_images_rec = samples[1][1]
    gen_scores = samples[1][2]
    mean = np.mean([real_scores, gen_scores])
    n_samples = real_images.shape[0]
    rows = int(np.sqrt(n_samples))
    cols = (n_samples + rows - 1)  / rows
    im_h = real_images.shape[2]
    im_w = real_images.shape[3]
    font = ImageFont.truetype(font_name, size=font_size)
    canvas = Image.new('RGB', ((im_w + stride)*cols*2*2 + mid_gap,
                               (im_h + text_height)*rows),
                               'white')
    draw = ImageDraw.Draw(canvas)
    for num in xrange(n_samples):
        # index of the grid to paste the images
        i = num / cols
        j = num - i*cols
        # real image
        im = Image.fromarray(get_plotable_data(real_images[num,...]))
        im_rec = Image.fromarray(get_plotable_data(real_images_rec[num,...]))
        score = real_scores[num]
        sgn = get_sgn(score, mean)
        x = (im_w + stride) * j*2
        y = (im_h + text_height) * i
        canvas.paste(im, (x,y))
        canvas.paste(im_rec, (x + (im_w+stride), y))
        draw.text((x, y + im_h), '%.4e' % score, font=font, fill=sgn)
        # generated sample
        im = Image.fromarray(get_plotable_data(gen_images[num,...]))
        im_rec = Image.fromarray(get_plotable_data(gen_images_rec[num,...]))
        score = gen_scores[num]
        sgn = get_sgn(score, mean)
        x = x + (im_w + stride)*cols*2 + mid_gap
#        y = (im_h + text_height) * i
        canvas.paste(im, (x,y))
        canvas.paste(im_rec, (x+(im_w+stride), y))
        draw.text((x, y + im_h), '%.4e' % score, font=font, fill=sgn)
    canvas.save(save_path, 'PNG')
    
def display_samples(samples, save_path, input_format='HWC', scores=None,
                    stride=2, font_size=10, text_height=12, mid_gap=16,
                    font_name='/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf'):
    real_images = samples[0]
    gen_images = samples[1]
    n_images_per_sample = len(real_images)
    n_samples = real_images[0].shape[0]
    rows = int(np.sqrt(n_samples))
    cols = (n_samples + rows - 1)  / rows
    input_format = input_format.upper()
    idx_h = input_format.find('H')
    idx_w = input_format.find('W')
    im_h = real_images[0].shape[idx_h]
    im_w = real_images[0].shape[idx_w]
    w_real = (im_w + stride)*cols*n_images_per_sample
    mean = 0.
    if scores is not None:
        mean = np.mean(scores)
    font = ImageFont.truetype(font_name, size=font_size)
    canvas = Image.new('RGB', (w_real*2 + mid_gap,
                               (im_h + text_height)*rows),
                               'white')
    draw = ImageDraw.Draw(canvas)
    for num in xrange(n_samples):
        # index of the grid to paste the images
        i = num / cols
        j = num - i*cols
        x_first = (im_w + stride) * j
        y = (im_h + text_height) * i
        for k in xrange(n_images_per_sample):
            x = x_first + (im_w + stride)*k
            # real image
            im = Image.fromarray(get_plotable_data(real_images[k][num,...], input_format))
            canvas.paste(im, (x,y))
            # generated sample
            im = Image.fromarray(get_plotable_data(gen_images[k][num,...], input_format))
            canvas.paste(im, (x + w_real + mid_gap, y))
            if scores is not None and k == 0:
                score = scores[0][num]
                sgn = get_sgn(score, mean)
                draw.text((x, y + im_h), '%.4e' % score, font=font, fill=sgn)
                score = scores[1][num]
                sgn = get_sgn(score, mean)
                draw.text((x + w_real + mid_gap, y + im_h), '%.4e' % score, font=font, fill=sgn)
    canvas.save(save_path, 'PNG')

def generate_test_samples(netg, netd, score_blob, n_samples=64):
#    g_output_blob = netg.output_blobs[0]
    d_input_blob = netd.input_blobs[0]
    real_images = list()
    real_scores = list()
    gen_images = list()
    gen_scores = list()
    count = 0
    batchsize = d_input_blob.shape[0]
    while count < n_samples:
        # generated data
        netg.load_data(None)
        netg.forward()
        gen_images.append(d_input_blob.gpu_data.get())
        netd.forward()
        gen_scores.append(score_blob.gpu_data.get())
        # real data
        netd.load_data(None)
        real_images.append(d_input_blob.gpu_data.get())
        netd.forward()
        real_scores.append(score_blob.gpu_data.get())
        count += batchsize
    real_images = np.concatenate(real_images)
    real_scores = np.concatenate(real_scores)
    gen_images = np.concatenate(gen_images)
    gen_scores = np.concatenate(gen_scores)
    return [(real_images[:n_samples,...], real_scores[:n_samples,...]),
            (gen_images[:n_samples,...], gen_scores[:n_samples,...])]

def generate_test_samples_encoder(netg, netd, rec_blob, score_blob, n_samples=64):
#    g_output_blob = netg.output_blobs[0]
    d_input_blob = netd.input_blobs[0]
    real_images = list()
    real_images_rec = list()
    real_scores = list()
    gen_images = list()
    gen_images_rec = list()
    gen_scores = list()
    count = 0
    batchsize = d_input_blob.shape[0]
    while count < n_samples:
        # generated data
        netg.load_data(None)
        netg.forward()
        gen_images.append(d_input_blob.gpu_data.get())
        netd.forward()
        gen_images_rec.append(rec_blob.gpu_data.get())
        gen_scores.append(score_blob.gpu_data.get())
        # real data
        netd.load_data(None)
        real_images.append(d_input_blob.gpu_data.get())
        netd.forward()
        real_images_rec.append(rec_blob.gpu_data.get())
        real_scores.append(score_blob.gpu_data.get())
        count += batchsize
    real_images = np.concatenate(real_images)
    real_images_rec = np.concatenate(real_images_rec)
    real_scores = np.concatenate(real_scores)
    gen_images = np.concatenate(gen_images)
    gen_images_rec = np.concatenate(gen_images_rec)
    gen_scores = np.concatenate(gen_scores)
    return [(real_images[:n_samples,...], real_images_rec[:n_samples,...], real_scores[:n_samples,...]),
            (gen_images[:n_samples,...], gen_images_rec[:n_samples,...], gen_scores[:n_samples,...])]
            
def snapshot(itr, nets, net_names, log_obj, log_name, folder='.'):
    for net, net_name in zip(nets, net_names):
        net.save(osp.join(folder, '%s_itr_%d.caffemodel' % (net_name, itr)))
    eval("np.savez('%s', %s)" % (osp.join(folder, log_name+'.npz'),
                   ','.join(["%s=log_obj['%s']" % (k,k) for k in log_obj.keys()])))
