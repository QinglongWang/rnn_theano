python main.py --data SL/SL2/1k/ --epoch 500 --batch 100 --test_batch 10 --rnn UNI --act tanh --nhid 10 --seed 1 > log/SL2/1k/uni_t_ep500_b100_h10.log
python main.py --data SL/SL2/1k/ --epoch 500 --batch 100 --test_batch 10 --rnn O2 --act tanh --nhid 10 --seed 1 > log/SL2/1k/o2_t_ep500_b100_h10.log
python main.py --data SL/SL2/1k/ --epoch 500 --batch 100 --test_batch 10 --rnn M --nhid 10 --seed 1 > log/SL2/1k/m_ep500_b100_h10.log
python main.py --data SL/SL2/1k/ --epoch 500 --batch 100 --test_batch 10 --rnn MI --nhid 10 --seed 1 > log/SL2/1k/mi_ep500_b100_h10.log
python main.py --data SL/SL2/1k/ --epoch 500 --batch 100 --test_batch 10 --rnn SRN --act tanh --nhid 10 --seed 1 > log/SL2/1k/srn_t_ep500_b100_h10.log
python main.py --data SL/SL2/1k/ --epoch 500 --batch 100 --test_batch 10 --rnn LSTM --nhid 10 --seed 1 > log/SL2/1k/lstm_ep500_b100_h10.log
python main.py --data SL/SL2/1k/ --epoch 500 --batch 100 --test_batch 10 --rnn GRU --nhid 10 --seed 1 > log/SL2/1k/gru_ep500_b100_h10.log

#python main.py --data SL/SL4/1k/ --epoch 500 --batch 100 --test_batch 10 --rnn UNI --act tanh --nhid 10 --seed 1 > log/SL4/1k/uni_t_ep500_b100_h10.log
#python main.py --data SL/SL4/1k/ --epoch 500 --batch 100 --test_batch 10 --rnn O2 --act tanh --nhid 10 --seed 1 > log/SL4/1k/o2_t_ep500_b100_h10.log
#python main.py --data SL/SL4/1k/ --epoch 500 --batch 100 --test_batch 10 --rnn M --nhid 10 --seed 1 > log/SL4/1k/m_ep500_b100_h10.log
#python main.py --data SL/SL4/1k/ --epoch 500 --batch 100 --test_batch 10 --rnn MI --nhid 10 --seed 1 > log/SL4/1k/mi_ep500_b100_h10.log
#python main.py --data SL/SL4/1k/ --epoch 500 --batch 100 --test_batch 10 --rnn SRN --act tanh --nhid 10 --seed 1 > log/SL4/1k/srn_t_ep500_b100_h10.log
#python main.py --data SL/SL4/1k/ --epoch 500 --batch 100 --test_batch 10 --rnn LSTM --nhid 10 --seed 1 > log/SL4/1k/lstm_ep500_b100_h10.log
#python main.py --data SL/SL4/1k/ --epoch 500 --batch 100 --test_batch 10 --rnn GRU --nhid 10 --seed 1 > log/SL4/1k/gru_ep500_b100_h10.log

#python main.py --data SL/SL8/1k/ --epoch 500 --batch 100 --test_batch 10 --rnn UNI --act tanh --nhid 10 --seed 1 > log/SL8/1k/uni_t_ep500_b100_h10.log
#python main.py --data SL/SL8/1k/ --epoch 500 --batch 100 --test_batch 10 --rnn O2 --act tanh --nhid 10 --seed 1 > log/SL8/1k/o2_t_ep500_b100_h10.log
#python main.py --data SL/SL8/1k/ --epoch 500 --batch 100 --test_batch 10 --rnn M --nhid 10 --seed 1 > log/SL8/1k/m_ep500_b100_h10.log
#python main.py --data SL/SL8/1k/ --epoch 500 --batch 100 --test_batch 10 --rnn MI --nhid 10 --seed 1 > log/SL8/1k/mi_ep500_b100_h10.log
#python main.py --data SL/SL8/1k/ --epoch 500 --batch 100 --test_batch 10 --rnn SRN --act tanh --nhid 10 --seed 1 > log/SL8/1k/srn_t_ep500_b100_h10.log
#python main.py --data SL/SL8/1k/ --epoch 500 --batch 100 --test_batch 10 --rnn LSTM --nhid 10 --seed 1 > log/SL8/1k/lstm_ep500_b100_h10.log
#python main.py --data SL/SL8/1k/ --epoch 500 --batch 100 --test_batch 10 --rnn GRU --nhid 10 --seed 1 > log/SL8/1k/gru_ep500_b100_h10.log

#python main.py --data SP/SP2/1k/ --epoch 500 --batch 100 --test_batch 10 --rnn UNI --act tanh --nhid 10 --seed 1 > log/SP2/1k/uni_t_ep500_b100_h10.log
#python main.py --data SP/SP2/1k/ --epoch 500 --batch 100 --test_batch 10 --rnn O2 --act tanh --nhid 10 --seed 1 > log/SP2/1k/o2_t_ep500_b100_h10.log
#python main.py --data SP/SP2/1k/ --epoch 500 --batch 100 --test_batch 10 --rnn M --nhid 10 --seed 1 > log/SP2/1k/m_ep500_b100_h10.log
#python main.py --data SP/SP2/1k/ --epoch 500 --batch 100 --test_batch 10 --rnn MI --nhid 10 --seed 1 > log/SP2/1k/mi_ep500_b100_h10.log
#python main.py --data SP/SP2/1k/ --epoch 500 --batch 100 --test_batch 10 --rnn SRN --act tanh --nhid 10 --seed 1 > log/SP2/1k/srn_t_ep500_b100_h10.log
#python main.py --data SP/SP2/1k/ --epoch 500 --batch 100 --test_batch 10 --rnn LSTM --nhid 10 --seed 1 > log/SP2/1k/lstm_ep500_b100_h10.log
#python main.py --data SP/SP2/1k/ --epoch 500 --batch 100 --test_batch 10 --rnn GRU --nhid 10 --seed 1 > log/SP2/1k/gru_ep500_b100_h10.log

#python main.py --data SP/SP4/1k/ --epoch 500 --batch 100 --test_batch 10 --rnn UNI --act tanh --nhid 10 --seed 1 > log/SP4/1k/uni_t_ep500_b100_h10.log
#python main.py --data SP/SP4/1k/ --epoch 500 --batch 100 --test_batch 10 --rnn O2 --act tanh --nhid 10 --seed 1 > log/SP4/1k/o2_t_ep500_b100_h10.log
#python main.py --data SP/SP4/1k/ --epoch 500 --batch 100 --test_batch 10 --rnn M --nhid 10 --seed 1 > log/SP4/1k/m_ep500_b100_h10.log
#python main.py --data SP/SP4/1k/ --epoch 500 --batch 100 --test_batch 10 --rnn MI --nhid 10 --seed 1 > log/SP4/1k/mi_ep500_b100_h10.log
#python main.py --data SP/SP4/1k/ --epoch 500 --batch 100 --test_batch 10 --rnn SRN --act tanh --nhid 10 --seed 1 > log/SP4/1k/srn_t_ep500_b100_h10.log
#python main.py --data SP/SP4/1k/ --epoch 500 --batch 100 --test_batch 10 --rnn LSTM --nhid 10 --seed 1 > log/SP4/1k/lstm_ep500_b100_h10.log
#python main.py --data SP/SP4/1k/ --epoch 500 --batch 100 --test_batch 10 --rnn GRU --nhid 10 --seed 1 > log/SP4/1k/gru_ep500_b100_h10.log

#python main.py --data SP/SP8/1k/ --epoch 500 --batch 100 --test_batch 10 --rnn UNI --act tanh --nhid 10 --seed 1 > log/SP8/1k/uni_t_ep500_b100_h10.log
#python main.py --data SP/SP8/1k/ --epoch 500 --batch 100 --test_batch 10 --rnn O2 --act tanh --nhid 10 --seed 1 > log/SP8/1k/o2_t_ep500_b100_h10.log
#python main.py --data SP/SP8/1k/ --epoch 500 --batch 100 --test_batch 10 --rnn M --nhid 10 --seed 1 > log/SP8/1k/m_ep500_b100_h10.log
#python main.py --data SP/SP8/1k/ --epoch 500 --batch 100 --test_batch 10 --rnn MI --nhid 10 --seed 1 > log/SP8/1k/mi_ep500_b100_h10.log
#python main.py --data SP/SP8/1k/ --epoch 500 --batch 100 --test_batch 10 --rnn SRN --act tanh --nhid 10 --seed 1 > log/SP8/1k/srn_t_ep500_b100_h10.log
#python main.py --data SP/SP8/1k/ --epoch 500 --batch 100 --test_batch 10 --rnn LSTM --nhid 10 --seed 1 > log/SP8/1k/lstm_ep500_b100_h10.log
#python main.py --data SP/SP8/1k/ --epoch 500 --batch 100 --test_batch 10 --rnn GRU --nhid 10 --seed 1 > log/SP8/1k/gru_ep500_b100_h10.log