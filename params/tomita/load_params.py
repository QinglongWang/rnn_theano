import os, sys
import argparse
import time
import numpy as np
from numpy import linalg as LA
# python main_tomita.py --data g3 --epoch 100 --batch 100 --test_batch 10 --rnn O2 --act sigmoid --nhid 10

parser = argparse.ArgumentParser(description='RNN trained on Tomita grammars')
parser.add_argument('--data', type=str, default='7', help='location of data')
parser.add_argument('--seed', type=int, default=1, help='random seed for initialize weights')
parser.add_argument('--rnn', type=str, default='SRN', help='rnn model')
parser.add_argument('--nhid', type=int, default=5, help='hidden dimension')

args = parser.parse_args()

def l2_norm(weight):
    return np.sqrt(np.sum(weight * weight))

def params_norms(path, mode, params):
    pp = np.load(path)
    if mode == 'SRN':
        tmp = [l2_norm(pp['SRN_U']), l2_norm(pp['SRN_V']), l2_norm(pp['SRN_B']),
               pp['SRN_U'].mean(), pp['SRN_V'].mean(), pp['SRN_B'].mean()]
    elif mode == 'MI':
        tmp = [l2_norm(pp['MI_U']), l2_norm(pp['MI_V']), l2_norm(pp['MI_B']),
               l2_norm(pp['MI_alpha']), l2_norm(pp['MI_beta1']), l2_norm(pp['MI_beta2']),
               pp['MI_U'].mean(), pp['MI_V'].mean(), pp['MI_B'].mean(),
               pp['MI_alpha'].mean(), pp['MI_beta1'].mean(), pp['MI_beta2'].mean()]
    elif mode == 'M':
        tmp = [l2_norm(pp['M_fx']), l2_norm(pp['M_fh']), l2_norm(pp['M_hf']), l2_norm(pp['M_hx']), l2_norm(pp['M_B']),
               pp['M_fx'].mean(), pp['M_fh'].mean(), pp['M_hf'].mean(), pp['M_hx'].mean(), pp['M_B'].mean()]
    elif mode == 'O2':
        tmp = [l2_norm(pp['O2_W'][0]), l2_norm(pp['O2_W'][1]), l2_norm(pp['O2_B']),
               pp['O2_W'][0].mean(), pp['O2_W'][1].mean(), pp['O2_B'].mean()]
    elif mode == 'UNI':
        tmp = [l2_norm(pp['UNI_W'][0]), l2_norm(pp['UNI_W'][1]), l2_norm(pp['UNI_U']), l2_norm(pp['UNI_V']), l2_norm(pp['UNI_B']),
               pp['UNI_W'][0].mean(), pp['UNI_W'][1].mean(), pp['UNI_U'].mean(), pp['UNI_V'].mean(), pp['UNI_B'].mean()]

    params.append(tmp)

    return params

def params_eigs(path):
    pp = np.load(path)
    w_eig = LA.eig(pp['UNI_W'].sum(axis=0))[0]
    w0_eig = LA.eig(pp['UNI_W'][0])[0]
    w1_eig = LA.eig(pp['UNI_W'][1])[0]
    v_eig = LA.eig(pp['UNI_V'])[0]
    return np.concatenate((w_eig, w0_eig, w1_eig, v_eig))

params_norm_log = []
for i in range(7):
    save_dir = ''.join(('./g', str(i+1), '_', args.rnn, '_h', str(args.nhid), '_seed', str(args.seed), '_params.npz'))
    params_norm_log = params_norms(save_dir, args.rnn, params_norm_log)


np.savetxt(''.join(('./' + args.rnn, '_params_norms.csv')), np.array(params_norm_log), delimiter=",", fmt='%1.5f')