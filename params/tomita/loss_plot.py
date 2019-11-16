import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import numpy as np

figure(num=None, figsize=(6, 3), dpi=80, facecolor='w', edgecolor='k')
plt.rcParams.update({'font.size': 20})
# unit area ellipse

"""
color_dict = {'SRN':0.1, 'MI':0.2, 'M':0.3, 'O2':0.4, 'UNI':0.5, 'GRU':0.6, 'LSTM':0.7}
color_names = {'SRN':'red', 'MI':'green', 'M':'blue', 'O2':'cyan', 'UNI':'magenta', 'GRU':'yellow', 'LSTM':'black' }
x = np.array([4,5,5,5,6,8,10,10,10,10,10,10,10,14,16,17,20,21,28,30,30,30,30,30,30,30,49,57,57,70,71,98,100,100,100,100,100,100,100])
y = np.array([0.0349,0.5473,0.9913,1,0.0349,0.5301,0.4981,0.603,1,1,1,0.2197,0.505,0.4808,1,0.5376,0.4852,1,0.5089,0.4852,1,0.5089,
                0.5376,0.4852,1,0.5089,0.4852,0.5376,1,0.5089,0.5376,0.4852,0.4852,0.5325,1,0.4852,0.9999,0.5493,0.5511])
m = ['LSTM','GRU','UNI','O2','M','MI','SRN','MI','O2','M','UNI','GRU','LSTM','LSTM','UNI','GRU','M','O2','MI','SRN','MI','O2','M',
     'UNI','GRU','LSTM','LSTM','UNI','GRU','M','O2','MI','SRN','MI','O2','M','UNI','GRU','LSTM']


s = np.ones(x.shape[0])*0.5#np.random.rand(x.shape[0])
s *= 10**2.

c = [color_names[i] for i in m]

fig, ax = plt.subplots()
ax.scatter(x, y, s, c, marker='o',alpha=0.7)
ax.legend()
plt.show()

aa = 1
"""

filled_markers = ('o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X')

SRN =  {'x':[10,30,100], 'p':[130,990,10300],
        'g1':[0.7273,0.5926,0.6061], 'g2':[0.8571,0.9963,0.8571], 'g7':[0.9534,0.9422,0.9294],
        'g3':[0.9146,0.8933,0.8857], 'g4':[0.9778,0.9711,0.9725],
        'g5':[0.4981,0.4852,0.6004], 'g6':[0.7074,0.7315,0.576],
        'c':'red', 'm':'x'}

MI =   {'x':[8,10,28,30,98,100], 'p':[112,160,952,1080,10192,10600],
        'g1':[1.0,1.0,1.0,1.0,1.0,1.0], 'g2':[1.0,1.0,1.0,1.0,1.0,1.0], 'g7':[1.0,1.0,1.0,1.0,1.0,1.0],
        'g3':[0.9683,0.9685,0.9685,0.9685,0.9685,0.9685], 'g4':[1.0,1.0,1.0,1.0,1.0,1.0],
        'g5':[0.5301,0.603,0.5089,0.5284,0.5517,0.5325], 'g6':[0.7839,0.7806,0.7386,0.713,0.6728,0.6779],

        'c':'darkgreen', 'm':'v'}

M =    {'x':[6,10,20,30,70,100], 'p':[102,250,900,1950,10150,20500],
        'g1':[1.0,1.0,1.0,1.0,1.0,1.0], 'g2':[1.0,1.0,1.0,1.0,1.0,1.0], 'g7':[1.0,1.0,1.0,1.0,1.0,1.0],
        'g3':[1.0,1.0,1.0,1.0,1.0,1.0], 'g4':[1.0,1.0,1.0,1.0,1.0,1.0],
        'g5':[0.5011,1,1,1,1,1], 'g6':[1.0,1.0,1.0,1.0,1.0,1.0],
        'c':'darkblue', 'm':'^'}

O2 =   {'x':[7,10,21,30,71,100], 'p':[105,210,903,1830,10153,20100],
        'g1':[1.0,1.0,1.0,1.0,1.0,1.0], 'g2':[1.0,1.0,1.0,1.0,1.0,1.0], 'g7':[0.9994,1,1,1,1,1],
        'g3':[1.0,1.0,1.0,1.0,1.0,1.0], 'g4':[1.0,1.0,1.0,1.0,1.0,1.0],
        'g5':[1.0,1.0,1.0,1.0,1.0,1.0], 'g6':[1.0,1.0,1.0,1.0,1.0,1.0],
        'c':'cyan', 'm':'s'}

UNI =  {'x':[5,10,16,30,57,100], 'p':[90,330,816,2790,9918,30300],
        'g1':[1.0,1.0,1.0,1.0,1.0,1.0], 'g2':[1,1,0.8571,1,1,1], 'g7':[0.9971,1,1,1,1,1],
        'g3':[0.9993,1,0.9998,1,1,1], 'g4':[0.9827,1,1,1,1,1],
        'g5':[0.9913,1,1,1,1,0.9999], 'g6':[1.0,1.0,1.0,1.0,1.0,0.9999],
        'c':'darkviolet', 'm':'*'}

GRU =  {'x':[5,10,17,30,57,100], 'p':[105,360,969,2880,10089,30600],
        'g1':[1.0,1.0,1.0,1.0,1.0,1.0], 'g2':[1,0.6154,1,1,1,1], 'g7':[1.0,1.0,1.0,1.0,1.0,1.0],
        'g3':[1.0,1.0,1.0,1.0,1.0,1.0], 'g4':[1.0,1.0,1.0,1.0,1.0,1.0],
        'g5':[0.5473,0.2197,0.5376,0.5466,0.5482,0.5493], 'g6':[0.303,0.4254,0.6634,0.8432,0.5312,0.8057],
        'c':'saddlebrown', 'm':'d'}

LSTM = {'x':[4,10,14,30,49,100], 'p':[96,480,896,3840,9996,40800],
        'g1':[1,1,1,1,1,1], 'g2':[1,1,1,1,1,1], 'g7':[1,0.9994,1,1,0.9994,0.9994],
        'g3':[0.9998,1,0.9998,0.9619,1,0.9803], 'g4':[1.0,1.0,1.0,1.0,1.0,1.0],
        'g5':[0.0349,0.505,0.4808,0.9974,0.542,0.5511], 'g6':[0.3015,0.9737,0.9444,0.9363,0.4939,0.5388],
        'c':'black', 'm':'o'}

gram = ['g1','g2','g3','g4','g5','g6','g7']
alpha_value = 0.7
marker_size = 160

for k, g in enumerate(gram):
    plt.figure(k)
    SRN_plt = plt.scatter(SRN['x'], SRN[g], s=marker_size, marker=SRN['m'], color=SRN['c'], alpha=alpha_value)
    MI_plt = plt.scatter(MI['x'], MI[g], s=marker_size, marker=MI['m'], color=MI['c'], alpha=alpha_value)
    M_plt  = plt.scatter(M['x'], M[g], s=marker_size, marker=M['m'], color=M['c'], alpha=alpha_value)
    O2_plt  = plt.scatter(O2['x'], O2[g], s=marker_size, marker=O2['m'], color=O2['c'], alpha=alpha_value)
    UNI_plt  = plt.scatter(UNI['x'], UNI[g], s=marker_size, marker=UNI['m'], color=UNI['c'], alpha=alpha_value)
    GRU_plt = plt.scatter(GRU['x'], GRU[g], s=marker_size, marker=GRU['m'], color=GRU['c'], alpha=alpha_value)
    LSTM_plt = plt.scatter(LSTM['x'], LSTM[g], s=marker_size, marker=LSTM['m'], color=LSTM['c'], alpha=alpha_value)
    if k == 0:
        location = 'center right'
        plt.legend((SRN_plt, MI_plt, M_plt, O2_plt, UNI_plt, GRU_plt, LSTM_plt),
                   ('SRN', 'MI', 'M', '2RNN', 'UNI', 'GRU', 'LSTM'),
                   scatterpoints=1, ncol=3, fontsize=17, bbox_to_anchor=(0.02, 0.7, 1., .102))#loc=locationa)
    '''
    elif k == 1:
        location = 'lower right'
    elif k == 2:
        location = 'center right'
    elif k == 3:
        location = 'lower right'
    elif k == 4:
        location = 'lower right'
    elif k == 5:
        location = 'lower right'
    elif k == 6:
        location = 'center right'     
    '''

    plt.grid(True)
    plt.margins(0, 0)
    # show it
    # #plt.show()

    figure_file = './img/' + g + '.pdf'
    #plt.savefig(figure_file)
    plt.savefig(figure_file, bbox_inches='tight')