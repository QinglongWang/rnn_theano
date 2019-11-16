import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import numpy as np

figure(num=None, figsize=(4, 2), dpi=80, facecolor='w', edgecolor='k')
plt.rcParams.update({'font.size': 20})
# unit area ellipse
"""
SRN		U	V	B			
MI		U	V	B	alpha	beta1	beta2
O2	W			B			
UNI	W	U	V	B			
"""

filled_markers = ('o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X')
models = ['SRN', 'MI', 'O2', 'UNI']
params_names = ['W', 'U', 'V', 'B', 'alpha', 'beta1', 'beta2']
colors = ['red','darkgreen','darkblue','black','darkviolet','saddlebrown','cyan']
model_color_codes = {'SRN':[1,2,3], 'MI':[1,2,3,4,5,6], 'O2':[0,3], 'UNI':[0,1,2,3]}


nhid = 5
seed = 1
alpha_value = 0.99
marker_size = 80
x = range(100)

m = 'UNI'
i = 0
for g in [3]:
    params_log_file = ''.join(('./g', str(g), '_', m, '_h', str(nhid), '_seed', str(seed), '_params_log_n05.npz'))
    log = np.load(params_log_file)['log']

    plt.figure(g+i*7)
>>>>>>> 64ac1afe2c22796b658d4bc480b7548182aba0c9

    for p in range(log.shape[1]):
        plt.plot(x, log[:,p], color=colors[model_color_codes[m][p]], alpha=alpha_value)

    '''
    if params_log_file[-7:-4] == 'log':
        legend_items = [params_names[j] for j in model_color_codes[m]]
        plt.legend(tuple(legend_items), scatterpoints=1, loc='best', ncol=3, fontsize=16)
    '''
    legend_items = [params_names[j] for j in model_color_codes[m]]
    plt.legend(tuple(legend_items), scatterpoints=1, loc='best', ncol=3, fontsize=16)


    plt.grid(True)
    plt.margins(0, 0)
    # show it
    #plt.show()

    figure_file = './img/g' + str(g) + '_' + m +'_params_n05.pdf'
    #plt.savefig(figure_file)
    plt.savefig(figure_file, bbox_inches='tight')
