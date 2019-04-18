touch log/SL2/100k/uni_t_ep500_b100_h10.log
touch log/SL2/100k/o2_t_ep500_b100_h10.log
touch log/SL2/100k/m_ep500_b100_h10.log
touch log/SL2/100k/mi_ep500_b100_h10.log
touch log/SL2/100k/srn_t_ep500_b100_h10.log
touch log/SL2/100k/lstm_ep500_b100_h10.log
touch log/SL2/100k/gru_ep500_b100_h10.log

touch log/SL4/100k/uni_t_ep500_b100_h10.log
touch log/SL4/100k/o2_t_ep500_b100_h10.log
touch log/SL4/100k/m_ep500_b100_h10.log
touch log/SL4/100k/mi_ep500_b100_h10.log
touch log/SL4/100k/srn_t_ep500_b100_h10.log
touch log/SL4/100k/lstm_ep500_b100_h10.log
touch log/SL4/100k/gru_ep500_b100_h10.log

touch log/SL8/100k/uni_t_ep500_b100_h10.log
touch log/SL8/100k/o2_t_ep500_b100_h10.log
touch log/SL8/100k/m_ep500_b100_h10.log
touch log/SL8/100k/mi_ep500_b100_h10.log
touch log/SL8/100k/srn_t_ep500_b100_h10.log
touch log/SL8/100k/lstm_ep500_b100_h10.log
touch log/SL8/100k/gru_ep500_b100_h10.log

touch log/SP2/100k/uni_t_ep500_b100_h10.log
touch log/SP2/100k/o2_t_ep500_b100_h10.log
touch log/SP2/100k/m_ep500_b100_h10.log
touch log/SP2/100k/mi_ep500_b100_h10.log
touch log/SP2/100k/srn_t_ep500_b100_h10.log
touch log/SP2/100k/lstm_ep500_b100_h10.log
touch log/SP2/100k/gru_ep500_b100_h10.log

touch log/SP4/100k/uni_t_ep500_b100_h10.log
touch log/SP4/100k/o2_t_ep500_b100_h10.log
touch log/SP4/100k/m_ep500_b100_h10.log
touch log/SP4/100k/mi_ep500_b100_h10.log
touch log/SP4/100k/srn_t_ep500_b100_h10.log
touch log/SP4/100k/lstm_ep500_b100_h10.log
touch log/SP4/100k/gru_ep500_b100_h10.log

touch log/SP8/100k/uni_t_ep500_b100_h10.log
touch log/SP8/100k/o2_t_ep500_b100_h10.log
touch log/SP8/100k/m_ep500_b100_h10.log
touch log/SP8/100k/mi_ep500_b100_h10.log
touch log/SP8/100k/srn_t_ep500_b100_h10.log
touch log/SP8/100k/lstm_ep500_b100_h10.log
touch log/SP8/100k/gru_ep500_b100_h10.log

python main.py --data 'SL/SL2/100k/' --epoch 500 --batch 100 --test_batch 10 --rnn UNI --act tanh --nhid 10 --seed 1 > log/SL2/100k/uni_t_ep500_b100_h10.log
python main.py --data 'SL/SL2/100k/' --epoch 500 --batch 100 --test_batch 10 --rnn O2 --act tanh --nhid 10 --seed 1 > log/SL2/100k/o2_t_ep500_b100_h10.log
python main.py --data 'SL/SL2/100k/' --epoch 500 --batch 100 --test_batch 10 --rnn M --nhid 10 --seed 1 > log/SL2/100k/m_ep500_b100_h10.log
python main.py --data 'SL/SL2/100k/' --epoch 500 --batch 100 --test_batch 10 --rnn MI --nhid 10 --seed 1 > log/SL2/100k/mi_ep500_b100_h10.log
python main.py --data 'SL/SL2/100k/' --epoch 500 --batch 100 --test_batch 10 --rnn SRN --act tanh --nhid 10 --seed 1 > log/SL2/100k/srn_t_ep500_b100_h10.log
python main.py --data 'SL/SL2/100k/' --epoch 500 --batch 100 --test_batch 10 --rnn LSTM --nhid 10 --seed 1 > log/SL2/100k/lstm_ep500_b100_h10.log
python main.py --data 'SL/SL2/100k/' --epoch 500 --batch 100 --test_batch 10 --rnn GRU --nhid 10 --seed 1 > log/SL2/100k/gru_ep500_b100_h10.log

python main.py --data 'SL/SL4/100k/' --epoch 500 --batch 100 --test_batch 10 --rnn UNI --act tanh --nhid 10 --seed 1 > log/SL4/100k/uni_t_ep500_b100_h10.log
python main.py --data 'SL/SL4/100k/' --epoch 500 --batch 100 --test_batch 10 --rnn O2 --act tanh --nhid 10 --seed 1 > log/SL4/100k/o2_t_ep500_b100_h10.log
python main.py --data 'SL/SL4/100k/' --epoch 500 --batch 100 --test_batch 10 --rnn M --nhid 10 --seed 1 > log/SL4/100k/m_ep500_b100_h10.log
python main.py --data 'SL/SL4/100k/' --epoch 500 --batch 100 --test_batch 10 --rnn MI --nhid 10 --seed 1 > log/SL4/100k/mi_ep500_b100_h10.log
python main.py --data 'SL/SL4/100k/' --epoch 500 --batch 100 --test_batch 10 --rnn SRN --act tanh --nhid 10 --seed 1 > log/SL4/100k/srn_t_ep500_b100_h10.log
python main.py --data 'SL/SL4/100k/' --epoch 500 --batch 100 --test_batch 10 --rnn LSTM --nhid 10 --seed 1 > log/SL4/100k/lstm_ep500_b100_h10.log
python main.py --data 'SL/SL4/100k/' --epoch 500 --batch 100 --test_batch 10 --rnn GRU --nhid 10 --seed 1 > log/SL4/100k/gru_ep500_b100_h10.log

python main.py --data 'SL/SL8/100k/' --epoch 500 --batch 100 --test_batch 10 --rnn UNI --act tanh --nhid 10 --seed 1 > log/SL8/100k/uni_t_ep500_b100_h10.log
python main.py --data 'SL/SL8/100k/' --epoch 500 --batch 100 --test_batch 10 --rnn O2 --act tanh --nhid 10 --seed 1 > log/SL8/100k/o2_t_ep500_b100_h10.log
python main.py --data 'SL/SL8/100k/' --epoch 500 --batch 100 --test_batch 10 --rnn M --nhid 10 --seed 1 > log/SL8/100k/m_ep500_b100_h10.log
python main.py --data 'SL/SL8/100k/' --epoch 500 --batch 100 --test_batch 10 --rnn MI --nhid 10 --seed 1 > log/SL8/100k/mi_ep500_b100_h10.log
python main.py --data 'SL/SL8/100k/' --epoch 500 --batch 100 --test_batch 10 --rnn SRN --act tanh --nhid 10 --seed 1 > log/SL8/100k/srn_t_ep500_b100_h10.log
python main.py --data 'SL/SL8/100k/' --epoch 500 --batch 100 --test_batch 10 --rnn LSTM --nhid 10 --seed 1 > log/SL8/100k/lstm_ep500_b100_h10.log
python main.py --data 'SL/SL8/100k/' --epoch 500 --batch 100 --test_batch 10 --rnn GRU --nhid 10 --seed 1 > log/SL8/100k/gru_ep500_b100_h10.log

python main.py --data 'SP/SP2/100k/' --epoch 500 --batch 100 --test_batch 10 --rnn UNI --act tanh --nhid 10 --seed 1 > log/SP2/100k/uni_t_ep500_b100_h10.log
python main.py --data 'SP/SP2/100k/' --epoch 500 --batch 100 --test_batch 10 --rnn O2 --act tanh --nhid 10 --seed 1 > log/SP2/100k/o2_t_ep500_b100_h10.log
python main.py --data 'SP/SP2/100k/' --epoch 500 --batch 100 --test_batch 10 --rnn M --nhid 10 --seed 1 > log/SP2/100k/m_ep500_b100_h10.log
python main.py --data 'SP/SP2/100k/' --epoch 500 --batch 100 --test_batch 10 --rnn MI --nhid 10 --seed 1 > log/SP2/100k/mi_ep500_b100_h10.log
python main.py --data 'SP/SP2/100k/' --epoch 500 --batch 100 --test_batch 10 --rnn SRN --act tanh --nhid 10 --seed 1 > log/SP2/100k/srn_t_ep500_b100_h10.log
python main.py --data 'SP/SP2/100k/' --epoch 500 --batch 100 --test_batch 10 --rnn LSTM --nhid 10 --seed 1 > log/SP2/100k/lstm_ep500_b100_h10.log
python main.py --data 'SP/SP2/100k/' --epoch 500 --batch 100 --test_batch 10 --rnn GRU --nhid 10 --seed 1 > log/SP2/100k/gru_ep500_b100_h10.log

python main.py --data 'SP/SP4/100k/' --epoch 500 --batch 100 --test_batch 10 --rnn UNI --act tanh --nhid 10 --seed 1 > log/SP4/100k/uni_t_ep500_b100_h10.log
python main.py --data 'SP/SP4/100k/' --epoch 500 --batch 100 --test_batch 10 --rnn O2 --act tanh --nhid 10 --seed 1 > log/SP4/100k/o2_t_ep500_b100_h10.log
python main.py --data 'SP/SP4/100k/' --epoch 500 --batch 100 --test_batch 10 --rnn M --nhid 10 --seed 1 > log/SP4/100k/m_ep500_b100_h10.log
python main.py --data 'SP/SP4/100k/' --epoch 500 --batch 100 --test_batch 10 --rnn MI --nhid 10 --seed 1 > log/SP4/100k/mi_ep500_b100_h10.log
python main.py --data 'SP/SP4/100k/' --epoch 500 --batch 100 --test_batch 10 --rnn SRN --act tanh --nhid 10 --seed 1 > log/SP4/100k/srn_t_ep500_b100_h10.log
python main.py --data 'SP/SP4/100k/' --epoch 500 --batch 100 --test_batch 10 --rnn LSTM --nhid 10 --seed 1 > log/SP4/100k/lstm_ep500_b100_h10.log
python main.py --data 'SP/SP4/100k/' --epoch 500 --batch 100 --test_batch 10 --rnn GRU --nhid 10 --seed 1 > log/SP4/100k/gru_ep500_b100_h10.log

python main.py --data 'SP/SP8/100k/' --epoch 500 --batch 100 --test_batch 10 --rnn UNI --act tanh --nhid 10 --seed 1 > log/SP8/100k/uni_t_ep500_b100_h10.log
python main.py --data 'SP/SP8/100k/' --epoch 500 --batch 100 --test_batch 10 --rnn O2 --act tanh --nhid 10 --seed 1 > log/SP8/100k/o2_t_ep500_b100_h10.log
python main.py --data 'SP/SP8/100k/' --epoch 500 --batch 100 --test_batch 10 --rnn M --nhid 10 --seed 1 > log/SP8/100k/m_ep500_b100_h10.log
python main.py --data 'SP/SP8/100k/' --epoch 500 --batch 100 --test_batch 10 --rnn MI --nhid 10 --seed 1 > log/SP8/100k/mi_ep500_b100_h10.log
python main.py --data 'SP/SP8/100k/' --epoch 500 --batch 100 --test_batch 10 --rnn SRN --act tanh --nhid 10 --seed 1 > log/SP8/100k/srn_t_ep500_b100_h10.log
python main.py --data 'SP/SP8/100k/' --epoch 500 --batch 100 --test_batch 10 --rnn LSTM --nhid 10 --seed 1 > log/SP8/100k/lstm_ep500_b100_h10.log
python main.py --data 'SP/SP8/100k/' --epoch 500 --batch 100 --test_batch 10 --rnn GRU --nhid 10 --seed 1 > log/SP8/100k/gru_ep500_b100_h10.log