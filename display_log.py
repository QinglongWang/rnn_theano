import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Display')
parser.add_argument('--gram', type=str, default='SL', help = 'SL:strick local, SP:strict piecewise')
parser.add_argument('--k', type=str, default='2', help = '2, 4 or 8')
parser.add_argument('--n', type=str, default='1k', help = 'number of strings')

args = parser.parse_args()

model_names = {'uni_t':0, 'o2_t':1, 'm':2, 'mi':3, 'srn_t':4, 'lstm':5, 'gru':6}
def read_file(filename, model_name, log_matrix):
    f_read = open(filename, 'r')
    lines = f_read.readlines()[-6:]
    val_line = lines[0]
    test_line = lines[3]

    log_matrix[model_names[model_name], 0] = float(val_line[val_line.index('Acc')+4 : val_line.index('F1')-1])
    log_matrix[model_names[model_name], 1] = float(test_line[test_line.index('Acc')+4 : test_line.index('F1')-1])

    return log_matrix


log_matrix = np.zeros((len(model_names), 2))

model_name = 'uni_t'
filename = 'log/' + args.gram + args.k + '/' + args.n + '/' + model_name + '_ep500_b100_h10.log'
log_matrix = read_file(filename, 'uni_t', log_matrix)

np.savetxt(''.join(('log/' + args.gram + args.k + '/' + args.n, '/acc.csv')), log_matrix, delimiter=",", fmt='%1.5f')