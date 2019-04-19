import numpy as np
import argparse
# python display_log.py --gram SL --k 2 --n 1k
parser = argparse.ArgumentParser(description='Display')
parser.add_argument('--gram', type=str, default='SL', help = 'SL:strick local, SP:strict piecewise')
parser.add_argument('--k', type=str, default='2', help = '2, 4 or 8')
parser.add_argument('--n', type=str, default='1k', help = 'number of strings')

args = parser.parse_args()

model_names = {'uni_t':0, 'o2_t':1, 'm':2, 'mi':3, 'srn_t':4, 'lstm':5, 'gru':6}
file_path = 'log/' + args.gram + args.k + '/' + args.n + '/'
file_names = {'uni_t': file_path + 'uni_t_ep100_b100_h28.log',
              'o2_t':  file_path + 'o2_t_ep100_b100_h31.log',
              'm':     file_path + 'm_ep100_b100_h42.log',
              'mi':    file_path + 'mi_ep100_b100_h60.log',
              'srn_t': file_path + 'srn_t_ep100_b100_h62.log',
              'lstm':  file_path + 'lstm_ep100_b100_h30.log',
              'gru':   file_path + 'gru_ep100_b100_h34.log'}
def read_file(filename, line_idx, log_matrix):
    f_read = open(filename, 'r')
    lines = f_read.readlines()[-6:]
    val_line = lines[0]
    test_line = lines[3]
    #print(val_line)
    #print(test_line)
    log_matrix[line_idx, 0] = float(val_line[val_line.index('Acc')+4 : val_line.index('F1')-1])
    log_matrix[line_idx, 1] = float(test_line[test_line.index('Acc')+4 : test_line.index('F1')-1])

    return log_matrix


log_matrix = np.zeros((len(model_names), 2))

for key, value in model_names.items():
    log_matrix = read_file(file_names[key], value, log_matrix)

np.savetxt(''.join(('log/' + args.gram + args.k + '/' + args.n, '/acc.csv')), log_matrix, delimiter=",", fmt='%1.5f')
