import os, sys
#os.environ["THEANO_FLAGS"]="floatX=float32,device=gpu1,gpuarray.preallocate=1,mode=FAST_RUN"
os.environ["THEANO_FLAGS"]="floatX=float32,device=cpu,gpuarray.preallocate=1"
import argparse
import time
#import numpy as np
from model import RNNModel_IMDB
from utils import *#unzip, update_model, load_params, save_hinit, load_data, get_minibatches_idx, perf_measure

#python main_rna.py --epoch 100 --batch 100 --test_batch 10 --rnn UNI --act sigmoid --nhid 10

parser = argparse.ArgumentParser(description='RNN trained on SST-2')
parser.add_argument('--epoch', type=int, default=50, help='epoch num')
parser.add_argument('--evaluate_loss_after', type=int, default=5, help='evaluate and print out results')
parser.add_argument('--early_stopping', type=int, default=20, help='Tolerance for early stopping (# of epochs).')
parser.add_argument('--batch', type=int, default=100, help='batch size')
parser.add_argument('--test_batch', type=int, default=-1, help='test batch_ ize')
parser.add_argument('--continue_train', action='store_true', default=False, help='continue train from a checkpoint')
parser.add_argument('--curriculum', action='store_true', default=False, help='curriculum train')
parser.add_argument('--seed', type=int, default=123, help='random seed for initialize weights')

parser.add_argument('--rnn', type=str, default='O2', help='rnn model')
parser.add_argument('--act', type=str, default='tanh', help='rnn model')
parser.add_argument('--ninp', type=int, default=100, help='embedding dimension')
parser.add_argument('--nhid', type=int, default=10, help='hidden dimension')

args = parser.parse_args()

if args.test_batch < 0:
    args.test_batch = args.batch

np.random.seed(args.seed)
###############################################################################
# Train the model
###############################################################################
def train(model, x, m, y, x_v, m_v, y_v, args, params_file, data_type='float32'):
    #emb = np.fliplr(np.eye(args.ntoken, dtype=data_type))
    #x = emb[x].reshape([x.shape[0], x.shape[1], args.ntoken])
    m = np.array(m, dtype=data_type)

    cost_val = []

    for epoch in range(args.epoch):
        #h_log = np.zeros((x.shape[0], x.shape[1], args.nhid), dtype=data_type)
        y_pred = np.zeros(shape=(y.shape[0],), dtype=data_type)
        kf = get_minibatches_idx(x.shape[0], args.batch, shuffle=False)
        total_cost = []
        start_time_epoch = time.time()

        for batch, sample_index in kf:
            cost = model.f_grad_shared(x[sample_index, :].T, m[sample_index, :].T, y[sample_index])
            y_pred[sample_index] = model.f_pred(x[sample_index, :].T, m[sample_index, :].T)

            #h = model.f_states(np.transpose(x[sample_index, :], (1, 0, 2)), m[sample_index, :].T)
            #h_log[sample_index] = np.transpose(h, (1, 0, 2))

            total_cost.append(cost)
            model.f_update(0.99)

        (precision, recall, accuracy, f1) = perf_measure(y_true=y, y_pred=y_pred, use_self=False)
        print("Epoch %d: Time: %.4f Cost: %.4f Pre: %.4f Re: %.4f Acc: %.4f F1: %.4f" %
              (epoch, time.time() - start_time_epoch, np.mean(total_cost), precision, recall, accuracy, f1))

        if (epoch % args.evaluate_loss_after == 0):
            print('\n')
            print('--------------------------------------------------------------------')
            precision, recall, accuracy, f1, this_cost_val = validate(model=model, x=x_v, m=m_v, y=y_v, args=args)
            print('--------------------------------------------------------------------\n')
            sys.stdout.flush()

            if args.early_stopping is not None:
                cost_val.append(this_cost_val)
                if epoch > args.early_stopping and cost_val[-1] > np.mean(cost_val[-(args.early_stopping+1):-1]) and accuracy >= 0.9:
                    #model_params = unzip(model.tparams)
                    #np.savez(params_file, history_errs=total_cost, **model_params)
                    print("Early stopping...")
                    break



    model_params = unzip(model.tparams)
    np.savez(params_file, history_errs=total_cost, **model_params)

    return model
###############################################################################
# Curriculum Train the model
###############################################################################
def curriculum_train(model, x, m, y, x_v, m_v, y_v, args, params_file, data_type='float32'):
    #emb = np.fliplr(np.eye(args.ntoken, dtype=data_type))
    #x = emb[x].reshape([x.shape[0], x.shape[1], args.ntoken])
    m = np.array(m, dtype=data_type)
    lengths = sorted(list(set(m.sum(axis=1))))
    length_epochs = 5

    cost_val = []

    for epoch in range(args.epoch):
        for l in lengths:
            l_idx = list(np.where(m.sum(axis=1) == l)[0])
            x_batch = x[l_idx]
            y_batch = y[l_idx]
            m_batch = m[l_idx]

            for l_epoch in range(length_epochs):
                y_pred_batch = np.zeros(shape=(y_batch.shape[0],), dtype=data_type)
                kf = get_minibatches_idx(x_batch.shape[0], args.batch, shuffle=False)
                total_cost = []
                for batch, sample_index in kf:
                    cost = model.f_grad_shared(x_batch[sample_index, :].T, m_batch[sample_index, :].T, y_batch[sample_index])
                    y_pred_batch[sample_index] = model.f_pred(x_batch[sample_index, :].T, m_batch[sample_index, :].T)
                    total_cost.append(cost)
                    model.f_update(0.99)

            (precision, recall, accuracy, f1) = perf_measure(y_true=y_batch, y_pred=y_pred_batch, use_self=False)
            print("Epoch:%d Len:%d Cost:%.4f Pre:%.4f Re:%.4f Acc:%.4f F1:%.4f" %
                  (epoch, l, np.mean(total_cost), precision, recall, accuracy, f1))

        if (epoch % args.evaluate_loss_after == 0):
            print('\n--------------------------------------------------------------------')
            precision, recall, accuracy, f1, this_cost_val = validate(model=model, x=x_v, m=m_v, y=y_v, args=args)
            cost_val.append(this_cost_val)
            print('--------------------------------------------------------------------\n')
            sys.stdout.flush()
            if epoch > args.early_stopping and cost_val[-1] > np.mean(cost_val[-(args.early_stopping+1):-1]) and accuracy >= 0.9:
                model_params = unzip(model.tparams)
                np.savez(params_file, history_errs=total_cost, **model_params)
                print("Early stopping...")
                return model


    model_params = unzip(model.tparams)
    np.savez(params_file, history_errs=total_cost, **model_params)

    return model

###############################################################################
# Validate the model
###############################################################################


def validate(model, x, m, y, args, data_type='float32'):
    #emb = np.fliplr(np.eye(args.ntoken, dtype=data_type))
    #x = emb[x].reshape([x.shape[0], x.shape[1], args.ntoken])
    m = np.array(m, dtype=data_type)

    y_pred = np.zeros(shape=(y.shape[0],), dtype=data_type)
    kf = get_minibatches_idx(x.shape[0], args.batch, shuffle=False)
    total_cost = []

    for batch, sample_index in kf:
        cost = model.f_grad_shared(x[sample_index, :].T, m[sample_index, :].T, y[sample_index])
        y_pred[sample_index] = model.f_pred(x[sample_index, :].T, m[sample_index, :].T)
        total_cost.append(cost)

    (precision, recall, accuracy, f1) = perf_measure(y_true=y, y_pred=y_pred, use_self=False)
    print("Eval results: Cost:%.4f Pre:%.4f Re:%.4f Acc:%.4f F1:%.4f" %
          (np.mean(total_cost), precision, recall, accuracy, f1))
    sys.stdout.flush()

    return precision, recall, accuracy, f1, np.mean(total_cost)


###############################################################################
# Test the model
###############################################################################


def test(model, model_train, x, m, y, args, data_type='float32'):
    #emb = np.fliplr(np.eye(args.ntoken, dtype=data_type))
    #x = emb[x].reshape([x.shape[0], x.shape[1], args.ntoken])
    m = np.array(m, dtype=data_type)

    if model_train:
        model = update_model(model_train, model)
    '''
    if args.rnn == 'lstm':
        h_log = np.zeros((x.shape[0], x.shape[1], 2*args.nhid), dtype=data_type)
    else:
        h_log = np.zeros((x.shape[0], x.shape[1], args.nhid), dtype=data_type)
    '''
    y_pred = np.zeros(shape=(y.shape[0],), dtype=data_type)
    kf = get_minibatches_idx(x.shape[0], args.test_batch, shuffle=False)

    start_time_epoch = time.time()

    for batch, sample_index in kf:
        y_pred[sample_index] = model.f_pred(x[sample_index, :].T, m[sample_index, :].T)

    print('--------------------------------------------------------------------')
    print("Test %d samples take time:%.4f" % (x.shape[0], time.time() - start_time_epoch))
    print('--------------------------------------------------------------------\n')
    sys.stdout.flush()

    #return precision, recall, accuracy, f1

###############################################################################
# Load data
###############################################################################
save_dir = ''.join(('./params/SST2/', args.rnn, '_h', str(args.nhid), '_seed', str(args.seed)))
params_file = ''.join((save_dir, '_params.npz'))
hinit_file = ''.join((save_dir, '_hinit.npz'))
train_val_test_file = ''.join(('./data/SST-2/sst2.train_val_test_data.npz'))
assert os.path.exists(train_val_test_file)

# load data first
print('Load data')
npzfile = np.load(train_val_test_file)
args.ntoken = npzfile['alphabet']


train_x = npzfile['train_x'].astype('int32')
train_m = npzfile['train_m']
train_y = npzfile['train_y'].astype('int32')
val_x = npzfile['val_x'].astype('int32')
val_m = npzfile['val_m']
val_y = npzfile['val_y'].astype('int32')
test_x = npzfile['test_x'].astype('int32')
test_m = npzfile['test_m']
test_y = npzfile['test_y'].astype('int32')

###############################################################################
# Build the model
###############################################################################
if args.ninp < 0:
    args.ninp = args.ntoken

model = RNNModel_IMDB(rnn_type=args.rnn, ntoken=args.ntoken, ninp=args.ninp, nhid=args.nhid,
                      nonlinearity=args.act, seed=args.seed, debug=False)
model_test = RNNModel_IMDB(rnn_type=args.rnn, ntoken=args.ntoken, ninp=args.ninp, nhid=args.nhid,
                           nonlinearity=args.act, seed=args.seed, debug=False)

total_params = sum([np.prod(x[1].shape) for x in model.params.items()])
print('RNN type:' + args.rnn + " Seed:" + str(args.seed))
print('Model total parameters: {}'.format(total_params))

try:
    if args.continue_train:
        model.params, h_init = load_params(params_file, hinit_file, model.params)
        model.reload_hidden(h_init, args.batch)
        model.init_tparams()
        model.build_model()
    else:
        model.init_hidden(args.batch, args.seed)
        save_hinit(model.h_init[0], hinit_file)
        model.build_model()

    model_test.reload_hidden(model.h_init[0], args.test_batch)
    model_test.build_model()

    # train the model
    if args.curriculum:
        model = curriculum_train(model=model, x=train_x, m=train_m, y=train_y,
                                 x_v=val_x, m_v=val_m, y_v=val_y,
                                 args=args, params_file=params_file)
    else:
        model = train(model=model, x=train_x, m=train_m, y=train_y,
                      x_v=val_x, m_v=val_m, y_v=val_y,
                      args=args, params_file=params_file)

    # evaluate the model with all data
    print('--------------------------------------------------------------------')
    test(model=model_test, model_train=model, x=test_x, m=test_m, y=test_y, args=args)


except KeyboardInterrupt:
    print('-' * 89)
    # evaluate the model with all data
    test(model=model_test, model_train=model, x=test_x, m=test_m, y=test_y, args=args)
    print('Exiting from training early')
