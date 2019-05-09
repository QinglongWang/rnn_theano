python main_slsp.py --data SL/SL4/100k/ --epoch 300 --batch 100 --test_batch 10 --rnn SRN --act tanh --nhid 30 --seed 1 > log/SL4/100k/srn_t_ep300_b100_h30.log
python main_slsp.py --data SL/SL4/100k/ --epoch 300 --batch 100 --test_batch 10 --rnn MI --nhid 28 --seed 1 > log/SL4/100k/mi_ep300_b100_h28.log
python main_slsp.py --data SL/SL4/100k/ --epoch 300 --batch 100 --test_batch 10 --rnn O2 --act tanh --nhid 21 --seed 1 > log/SL4/100k/o2_t_ep300_b100_h21.log
python main_slsp.py --data SL/SL4/100k/ --epoch 300 --batch 100 --test_batch 10 --rnn UNI --act tanh --nhid 16 --seed 1 > log/SL4/100k/uni_t_ep300_b100_h16.log
python main_slsp.py --data SL/SL4/100k/ --epoch 300 --batch 100 --test_batch 10 --rnn GRU --nhid 17 --seed 1 > log/SL4/100k/gru_ep300_b100_h17.log
python main_slsp.py --data SL/SL4/100k/ --epoch 300 --batch 100 --test_batch 10 --rnn LSTM --nhid 14 --seed 1 > log/SL4/100k/lstm_ep300_b100_h14.log

python main_slsp.py --data SL/SL8/100k/ --epoch 300 --batch 100 --test_batch 10 --rnn SRN --act tanh --nhid 30 --seed 1 > log/SL8/100k/srn_t_ep300_b100_h30.log
python main_slsp.py --data SL/SL8/100k/ --epoch 300 --batch 100 --test_batch 10 --rnn MI --nhid 28 --seed 1 > log/SL8/100k/mi_ep300_b100_h28.log
python main_slsp.py --data SL/SL8/100k/ --epoch 300 --batch 100 --test_batch 10 --rnn O2 --act tanh --nhid 21 --seed 1 > log/SL8/100k/o2_t_ep300_b100_h21.log
python main_slsp.py --data SL/SL8/100k/ --epoch 300 --batch 100 --test_batch 10 --rnn UNI --act tanh --nhid 16 --seed 1 > log/SL8/100k/uni_t_ep300_b100_h16.log
python main_slsp.py --data SL/SL8/100k/ --epoch 300 --batch 100 --test_batch 10 --rnn GRU --nhid 17 --seed 1 > log/SL8/100k/gru_ep300_b100_h17.log
python main_slsp.py --data SL/SL8/100k/ --epoch 300 --batch 100 --test_batch 10 --rnn LSTM --nhid 14 --seed 1 > log/SL8/100k/lstm_ep300_b100_h14.log

python main_slsp.py --data SP/SP4/100k/ --epoch 300 --batch 100 --test_batch 10 --rnn SRN --act tanh --nhid 30 --seed 1 > log/SP4/100k/srn_t_ep300_b100_h30.log
python main_slsp.py --data SP/SP4/100k/ --epoch 300 --batch 100 --test_batch 10 --rnn MI --nhid 28 --seed 1 > log/SP4/100k/mi_ep300_b100_h28.log
python main_slsp.py --data SP/SP4/100k/ --epoch 300 --batch 100 --test_batch 10 --rnn O2 --act tanh --nhid 21 --seed 1 > log/SP4/100k/o2_t_ep300_b100_h21.log
python main_slsp.py --data SP/SP4/100k/ --epoch 300 --batch 100 --test_batch 10 --rnn UNI --act tanh --nhid 16 --seed 1 > log/SP4/100k/uni_t_ep300_b100_h16.log
python main_slsp.py --data SP/SP4/100k/ --epoch 300 --batch 100 --test_batch 10 --rnn GRU --nhid 17 --seed 1 > log/SP4/100k/gru_ep300_b100_h17.log
python main_slsp.py --data SP/SP4/100k/ --epoch 300 --batch 100 --test_batch 10 --rnn LSTM --nhid 14 --seed 1 > log/SP4/100k/lstm_ep300_b100_h14.log

python main_slsp.py --data SP/SP8/100k/ --epoch 300 --batch 100 --test_batch 10 --rnn SRN --act tanh --nhid 30 --seed 1 > log/SP8/100k/srn_t_ep300_b100_h30.log
python main_slsp.py --data SP/SP8/100k/ --epoch 300 --batch 100 --test_batch 10 --rnn MI --nhid 28 --seed 1 > log/SP8/100k/mi_ep300_b100_h28.log
python main_slsp.py --data SP/SP8/100k/ --epoch 300 --batch 100 --test_batch 10 --rnn O2 --act tanh --nhid 21 --seed 1 > log/SP8/100k/o2_t_ep300_b100_h21.log
python main_slsp.py --data SP/SP8/100k/ --epoch 300 --batch 100 --test_batch 10 --rnn UNI --act tanh --nhid 16 --seed 1 > log/SP8/100k/uni_t_ep300_b100_h16.log
python main_slsp.py --data SP/SP8/100k/ --epoch 300 --batch 100 --test_batch 10 --rnn GRU --nhid 17 --seed 1 > log/SP8/100k/gru_ep300_b100_h17.log
python main_slsp.py --data SP/SP8/100k/ --epoch 300 --batch 100 --test_batch 10 --rnn LSTM --nhid 14 --seed 1 > log/SP8/100k/lstm_ep300_b100_h14.log


python main_slsp.py --data SL/SL4/100k/ --epoch 300 --batch 100 --test_batch 10 --rnn M --nhid 20 --seed 1 > log/SL4/100k/m_ep300_b100_h20.log
python main_slsp.py --data SL/SL8/100k/ --epoch 300 --batch 100 --test_batch 10 --rnn M --nhid 20 --seed 1 > log/SL8/100k/m_ep300_b100_h20.log
python main_slsp.py --data SP/SP4/100k/ --epoch 300 --batch 100 --test_batch 10 --rnn M --nhid 20 --seed 1 > log/SP4/100k/m_ep300_b100_h20.log
python main_slsp.py --data SP/SP8/100k/ --epoch 300 --batch 100 --test_batch 10 --rnn M --nhid 20 --seed 1 > log/SP8/100k/m_ep300_b100_h20.log
