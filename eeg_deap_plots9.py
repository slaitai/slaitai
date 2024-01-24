import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from collections import Counter
from ZML_cZML_entropy import *
import glob
import numpy as np
from scipy.signal import butter, lfilter, freqz
import matplotlib.pyplot as plt
import math

PlotCaptionFontSize = 16
PlotLineWidth = 1
PlotLineWidth2 = 1.25
PlotTickFontSize = 14
DoPlot = 1
DoPlotEEGTime = 0
DoPlotEEGInformationTopology = 1
DoSave = 0

first_run = 1

cb_red1 = (200/255,82/255,0/255)
red2 = (238/255,0/255,0/255)	#CD0000	RGB(205,0,0)
red3 = (205/255,0/255,0/255)	#CD0000	RGB(205,0,0)
red4 = (139/255,0/255,0/255)	#8B0000	RGB(139,0,0)
royalblue1 = (72/255,118/255,255/255)	#4876FF	RGB(72,118,255)
royalblue2 = (67/255,110/255,238/255)	#436EEE	RGB(67,110,238)
royalblue3 = (58/255,95/255,205/255)	#3A5FCD	RGB(58,95,205)
cb_dark_blue = (0/255,107/255,164/255)
cb_orange = (255/255, 128/255, 14/255)
cb_orange2 = (230/255, 97/255, 1/255)
cb_indigo = (94/255,60/255,153/255)
cb_brown = (166/255,97/255,26/255)

tan2 = (238/255,154/255,73/255)	#EE9A49	RGB(238,154,73)

cb_green1 = (128/255,205/255,193/255)
cb_green2 = (1/255,133/255,113/255)
seagreen4 = (46/255,139/255,87/255) 	#2E8B57	RGB(46,139,87)
springgreen3 = (0/255,139/255,69/255) 	#008B45	RGB(0,139,69)

sienna3 = (205/255,104/255,57/255) 	#CD6839	RGB(205,104,57)

slateblue =  (106/255,90/255,205/255)           #6A5ACD	RGB(106,90,205)
lightgoldenrod2 = (238/255,220/255,130/255) 	# #EEDC82	RGB(238,220,130)
salmon3 = (205/255,112/255,84/255)      	    #CD7054	RGB(205,112,84)

slategray1 = (198/255,226/255,255/255)   #C6E2FF	RGB(198,226,255)
slategray3 = (159/255,182/255,205/255)    	#9FB6CD	RGB(159,182,205)
springgreen3 = (0/255,139/255,69/255)	#008B45	RGB(0,139,69)

tomato1 = (255/255,99/255,71/255)	 #FF6347	RGB(255,99,71)
tomato3	= (205/255,79/255,57/255)    #CD4F39	RGB(205,79,57)
wheat = (245/255,222/255,179/255)	#F5DEB3	RGB(245,222,179)
wheat3	= (205/255,186/255,150/255) #CDBA96	RGB(205,186,150)
wheat4	= (139/255,126/255,102/255)	#8B7E66	RGB(139,126,102)



############################################################
#   loading data fn
############################################################
# participant_col = [
#     'Alcohol-Use', 'Coffee',
#     'Tea', 'Tobacco',
#     'Hours of sleep',]

# def load_participant_info(path):
#     participant_info = pd.read_csv(path)
#     participant_info = participant_info[participant_col]
#     return participant_info

def load_eeg_data(mat_files, max_participants):
    # files = os.listdir(folder_path)
    # mat_files = [file for file in files if file.endswith(".mat")]
    # mat_files_pattern = '*.mat'
    # mat_files = glob.glob(mat_files_pattern)

    eeg_data = []
    filecount = 0
    for i in range(0, len(mat_files)):
        if filecount < max_participants:
            filecount = filecount + 1
            mat_data = scipy.io.loadmat(mat_files[i])
            data = mat_data['data']
            labels = mat_data['labels']

            par_str = ''
            # j=0
            # for col in participant_col:
            #     cell_val = participant_info.loc[i, col]
            #     par_str += col + ":"  + str(cell_val) + "  "
            #     if j==2:
            #         par_str += '\n'
            #     j+=1
            eeg_data.append((data, labels, par_str))
            print("loaded raw data shape:", mat_files[i], data.shape)

    return eeg_data

def plot_eeg(channel_data, showing_sec, down_samp_factor, titlestr, figsize=(10, 6)):
    sample_channel = channel_data[::down_samp_factor]
    if showing_sec <= 63: 
        fig = plt.figure(figsize=figsize)
        ax = fig.add_axes([0.15, 0.17, 0.8, 0.73])
        ax.plot(np.arange(0, showing_sec, 1/128), sample_channel[:128*showing_sec],
                     marker='o',  # each marker will be rendered as a circle
                     markersize=0,  # marker size
                     markerfacecolor='red',  # marker facecolor
                     markeredgecolor='black',  # marker edgecolor
                     markeredgewidth=0.2,  # marker edge width
                     linestyle='solid',  # line style will be dash line
                     linewidth=PlotLineWidth,  # line width
                     alpha=0.9,
                     color=royalblue3)
        
        ax.set_xlabel("time (t) sec")
        ylabelstr = "amplitude"
        ax.set_ylabel(ylabelstr)
        #titlestr = "EEG signal plot" #pa:{:d}, ch:{:d}, tr:{:d}, wave:{:s}".format(par, chan, trial, band)
        ax.set_title(titlestr)
        plt.show()

############################################################
#   symbolisation fn
############################################################
def min_max_norm(data, print_detail = 0):
    min_val = min(data)
    max_val = max(data)
    norm_data= [(x - min_val) / (max_val - min_val) for x in data]
    if print_detail:
        print('normalized data mean: ', np.mean(norm_data))
        print('normalized data std: ', np.std(norm_data))
    return norm_data

def round_up(n, decimals=0):
    multiplier = 10 ** decimals
    return math.ceil(n * multiplier) / multiplier


def roundint(value, base=5):
    return int(value) - int(value) % int(base)


#----------------------------------------------------------
def symbolise_amplitude_states(norm_data, ngram):
    
    def build_word(vector):
        concatenated_strings = []
        vector_length = len(vector)
        
        for i in range(vector_length - (ngram-1)):
            interval = vector[i:i + (ngram)]
            concatenated_string = "".join(map(str, interval))
            concatenated_strings.append(concatenated_string)
        
        return concatenated_strings
    
    norm_mean = np.mean(norm_data)
    norm_std = np.std(norm_data)
    symbol = []
    for val in norm_data:
        if (1+norm_std)*norm_mean < val: 
            symbol.append('3')
        elif norm_mean < val <= (1+norm_std)*norm_mean:
            symbol.append('2')
        elif (1-norm_std)*norm_mean < val <= norm_mean:
            symbol.append('1')
        else:
            symbol.append('0')

    # if ngram == 1:
    #     output = symbol
    # elif ngram == 2:
    output = build_word(symbol)
    return output

#----------------------------------------------------------
# create dictionary to mapping words
alphabet_mapping = {f'{i}{j}': chr(ord('A') + i*4 + j) for i in range(4) for j in range(4)}

#----------------------------------------------------------
def symbol_mapping(word_list, symb_approach=1):
    symbols = []
    if symb_approach==0: 
        
        for word in word_list:
            symbols.append(alphabet_mapping[word])
    else:
        for word in word_list:
            if word in ['00', '11', '22', '33']:
                symbols.append("FL")
            elif word in ['01','12','23']:
                symbols.append("U1")
            elif word in ['02','13']:
                symbols.append("U2")
            elif word in ['03']:
                symbols.append("U3")
            elif word in ['30']:
                symbols.append("D3")
            elif word in ['20','31']:
                symbols.append("D2")
            elif word in ['10','21','32']:
                symbols.append("D1")
    return symbols

#----------------------------------------------------------
def gen_symbol_probability(symbol_list):
    symbol_frequency = Counter(symbol_list)
    dict_freq = {}
    for word, count in symbol_frequency.items():
        dict_freq[word] = count
        
    # Calculate the probability of each word
    total_frequency = sum(dict_freq.values())
    symbol_probabilities = {word: frequency / total_frequency for word, frequency in dict_freq.items()}
    # build word probability dictionary
    dict_symbol_prob = {}
    for word, probability in symbol_probabilities.items():
        dict_symbol_prob[word] = probability
        
    return dict_symbol_prob, symbol_probabilities

#----------------------------------------------------------
def most_frequent_symbol(symbols):

    word_counts = Counter(symbols)
    most_common_word = word_counts.most_common(1)
    return most_common_word[0][0] if most_common_word else None

#----------------------------------------------------------
def GetBlockInformation(y_alpha, eeg_block_n, M, ap, bp, cp):
    y_alpha_symbols_blockwise = []
    for ygroup in chunker(y_alpha, eeg_block_n):
        y_alpha_norm = min_max_norm(ygroup)
        y_alpha_symb = symbolise_amplitude_states(y_alpha_norm, ngram=1)
        y_alpha_symbols = y_alpha_symb  # symbol_mapping(y_alpha_symb, symb_approach=SYMB_METHOD)

        Nsym_alpha = len(y_alpha_symbols)
        selected_symbol = most_frequent_symbol(y_alpha_symbols)
        Hetheta, palpha, symbols_ranked_alpha = FastEntropy5_czml(y_alpha_symbols, Nsym_alpha, selected_symbol, M, ap,
                                                                  bp, cp, zml_model='zml', eta=0.2)
        Information_alpha = -np.log2(palpha)
        dominant_symbol_alpha = symbols_ranked_alpha[M-1]
        y_alpha_symbols_blockwise.append(dominant_symbol_alpha)

    return y_alpha_symbols_blockwise

#----------------------------------------------------------
# constants
SAMP_RATE_Hz = 128
DOWN_SAMP_FACTOR = 1 # 32
SAMP_RATE = SAMP_RATE_Hz/DOWN_SAMP_FACTOR
SYMB_METHOD = 0
TRIAL_NUMBER = 40
CHANNEL_NUMBER = 32
max_participants = 5
time_plot_secs = 20 # 7
eeg_block_secs = 1  # 0.5
eeg_block_n = int(eeg_block_secs*SAMP_RATE_Hz/DOWN_SAMP_FACTOR)
time_plot_n =  int(time_plot_secs*SAMP_RATE_Hz/DOWN_SAMP_FACTOR) # 2560  # calc no of samples for time to be display

# pre-calculated model paramaters
if SYMB_METHOD == 0:
    M = 16
elif SYMB_METHOD == 1:
    M = 7
    
ap = 0.0075
bp = 4.2026
cp =  4.5362


#----------------------------------------------------------
def symbolise_all_participant_trials(eeg_data, time_plot_n):
    # eeg_data format
    # participant[data[trial][channel][signal], labels[trial][Valence,Arousal,Dominance,Liking], participant_info]


    # participant eeg data matrix
    # participant_eeg_matrix[
    #   participant_info,
    #   label,
    #   participant_1_data[
    #    trial_1[
    #       trial_word_lst,
    #       trial_word_prob_dictionary,
    #       trial_raw_word_probability,
    #       
    #        ]
    #    ]
    #    trial_2 []
    #
    #   ... ...
    #   participant_32    
    # ]

    participant_eeg_matrix = []

    participant_no = 0
    for participant in eeg_data:
        participant_info = participant[2]
        label = participant[1]
        data = participant[0]
        participant_no = participant_no + 1
        
        participant_data = []
        #for trial in range(0, 13):
        TRIAL_NUMBER = 16 # limit the runs
        for trial in range(0, TRIAL_NUMBER):
            trial_words = []
            channel_words = []
            for channel in range(22,23): # limit the runs
            #for channel in range(0, CHANNEL_NUMBER):
                trial_signal = data[trial][channel][::DOWN_SAMP_FACTOR]
                trial_norm = min_max_norm(trial_signal)
                symbols = symbolise_amplitude_states(trial_norm, ngram=1)
                #symbols = word # symbol_mapping(word, symb_approach=SYMB_METHOD)
                trial_words.extend(symbols)
                channel_words.append(symbols)

                # get the score for arousal
                Feature0 = label[trial, 0]
                Feature0 = roundint(Feature0, base=3)

                #
                # Plot EEG waveforms
                if DoPlotEEGTime == 1:
                    if participant_no == 1:
                        if trial == 0:
                            if channel == 22:
                                Ymin = -30
                                Ymax = 30
                                plot_EEG_time(participant_no, channel, trial, trial_signal[1:time_plot_n], Ymin, Ymax)
                                plot_EEGWaves_time(participant_no, channel, trial, trial_signal[1:time_plot_n])

                # Plot EEG information topology
                if DoPlotEEGInformationTopology == 1:
                    if participant_no == 1:
                    #if participant_no == 1 or participant_no == 2 or participant_no == 3:
                        if trial < 100:
                        #if trial == 0 or trial == 1 or trial == 2:
                            if channel == 22:
                                
                                plot_EEGInformationTopologyMain(trial_signal[1:time_plot_n], True, participant_no, channel, trial,  Feature0)

                        # if trial == 9 or trial == 10 or trial == 11 or trial == 12:
                        #     if channel == 22:
                        #         plot_EEGInformationTopology(participant_no, channel, trial, trial_signal[1:time_plot_n], Feature0)


            dict_word_prob, raw_word_probabilities = gen_symbol_probability(trial_words)
            participant_data.append([trial_words, dict_word_prob, raw_word_probabilities, channel_words])

        participant_eeg_matrix.append([participant_info, label, participant_data])
    return participant_eeg_matrix

#time_plot_n =  int(time_plot_secs*SAMP_RATE_Hz/DOWN_SAMP_FACTOR)
def run_infotop_customised_eeg(eeg_data):
 
    plot_EEGInformationTopologyMain(eeg_data, False, 0, 0, 0,  0)






def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_eeg(type, fs, order=5):
    freq_bands = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 40)
    }
    lowcut,highcut = freq_bands[type]

    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def butter_bandpass_filter_eeg(data, type, fs, order=5):
    b, a = butter_bandpass_eeg(type, fs, order=order)
    y = lfilter(b, a, data)
    return y

def GetMarkerSizeFromFrequency(y_alpha_symbols_blockwise, min_size, max_size):
    #
    # Used in plotting information topology
    # The markersize reflects the strength of the connection to
    # each particular point, scaled to fit within min_size, max_size
    #
    y_alpha_symbols_blockwise_int = [eval(i) for i in y_alpha_symbols_blockwise]
    Nsb = len(y_alpha_symbols_blockwise)
    counter = collections.Counter(y_alpha_symbols_blockwise)
    symbols_local = counter.most_common(Nsb)
    symbols_ranked = [num[0] for num in symbols_local]
    symbols_ranked_int = [eval(i) for i in symbols_ranked]
    symbols_ranked_count = [num[1] for num in symbols_local]

    min_count = float(min(symbols_ranked_count))
    max_count = float(max(symbols_ranked_count))

    div = abs(max_count - min_count)
    if div < 0.00001:
        new_scale = 1
    else:
        new_scale = float((max_size - min_size)/(max_count - min_count))
    res = []
    for i in y_alpha_symbols_blockwise_int:
        for j in range(0, len(symbols_ranked_int)):
            value_a = symbols_ranked_int[j]
            value_b = i
            if (symbols_ranked_int[j] == i):
                # scale to the min-max marker size
                c_var = min_size - (min_count*new_scale)
                val_a = new_scale*float(symbols_ranked_count[j]) + c_var
                val_b = round(val_a,1)
                res.append(val_b)

    return res

def plot_EEG_time(participant_no, eeg_ch_no,eeg_trial_no,eeg_ch_data, Ymin, Ymax):
    # ===========================================================================
    #
    # Plot EEG signal for one channel
    #
    # ===========================================================================

    OutputFilePNGA = "eeg{:d}_{:d}_{:d}.png".format(participant_no, eeg_ch_no,eeg_trial_no)
    # OutputFileJPGA = "Wdtheta{:s}_{:d}_{:d}_{:d}_{:d}_{:d}_{:d}_{:d}.jpg".format(FileID, Nx1, n, Ns, Noverlap, Ntoppercent, CorpusStart, CorpusEnd)
    # OutputFileEPSA = "Wdtheta{:s}_{:d}_{:d}_{:d}_{:d}_{:d}_{:d}_{:d}.eps".format(FileID, Nx1, n, Ns, Noverlap, Ntoppercent, CorpusStart, CorpusEnd)

    fig = plt.figure(figsize=(9, 3))
    ax = fig.add_axes([0.12, 0.17, 0.8, 0.73])
    ax.grid(True,which='major',axis='both',alpha=0.3)

    p4 = ax.plot(eeg_ch_data,
                 marker='o',  # each marker will be rendered as a circle
                 markersize=0,  # marker size
                 markerfacecolor='red',  # marker facecolor
                 markeredgecolor='black',  # marker edgecolor
                 markeredgewidth=0.2,  # marker edge width
                 linestyle='solid',  # line style will be dash line
                 linewidth=PlotLineWidth,  # line width
                 alpha=0.9,
                 color=royalblue3)

    #xlabelstr = "Symbols (n)" # {:s} H = {:1.1f}, W = {:1.1f}".format(WordID, He, Wd)
    ax.set_xlabel("time (n)", fontsize=PlotCaptionFontSize)
    #ylabelstr = "y(t) - EEG Ch({:d}), trial ({:d})".format(eeg_ch_no,eeg_trial_no)
    ylabelstr = "y(n)"
    ax.set_ylabel(ylabelstr, fontsize=PlotCaptionFontSize)

    titlestr = "EEG - pa:{:d}, ch:{:d}, tr:{:d}".format(participant_no, eeg_ch_no,eeg_trial_no)
    ax.set_title(titlestr, fontsize=PlotCaptionFontSize)
    ax.set_ylim(Ymin,Ymax)

    # Add all spines
    for spine in plt.gca().spines.values():
        spine.set_visible(True)

    if DoSave == 1:
        fig.savefig(OutputFilePNGA, format='png', dpi=300, bbox_inches='tight', transparent=True)
        #fig.savefig(OutputFileJPGA, format='jpg', dpi=300, bbox_inches='tight', transparent=True)
        # fig.savefig(OutputFileEPSA, format='eps', dpi=300, bbox_inches='tight', transparent=True, pad_inches=.25)

    plt.show()

def plot_EEGWaves_time(participant_no, eeg_ch_no,eeg_trial_no,eeg_ch_data):

    #-----------------------------------------------
    # Calculate alpha,beta,delta,gamma waves
    #-----------------------------------------------
    y_delta = butter_bandpass_filter_eeg(eeg_ch_data, 'delta', SAMP_RATE_Hz, order=6)
    y_theta = butter_bandpass_filter_eeg(eeg_ch_data, 'theta', SAMP_RATE_Hz, order=6)
    y_alpha = butter_bandpass_filter_eeg(eeg_ch_data, 'alpha', SAMP_RATE_Hz, order=6)
    y_beta  = butter_bandpass_filter_eeg(eeg_ch_data, 'beta', SAMP_RATE_Hz, order=6)
    y_gamma = butter_bandpass_filter_eeg(eeg_ch_data, 'gamma', SAMP_RATE_Hz, order=6)

    #-----------------------------------------------
    # Plot delta wave
    #-----------------------------------------------
    OutputFilePNGA = "eeg_delta{:d}_{:d}_{:d}.png".format(participant_no, eeg_ch_no,eeg_trial_no)
    # OutputFileJPGA = "Wdtheta{:s}_{:d}_{:d}_{:d}_{:d}_{:d}_{:d}_{:d}.jpg".format(FileID, Nx1, n, Ns, Noverlap, Ntoppercent, CorpusStart, CorpusEnd)
    # OutputFileEPSA = "Wdtheta{:s}_{:d}_{:d}_{:d}_{:d}_{:d}_{:d}_{:d}.eps".format(FileID, Nx1, n, Ns, Noverlap, Ntoppercent, CorpusStart, CorpusEnd)

    fig = plt.figure(figsize=(9, 3))
    ax = fig.add_axes([0.12, 0.17, 0.8, 0.73])
    ax.grid(True,which='major',axis='both',alpha=0.3)
    p4 = ax.plot(y_delta, label='Delta wave')
    ax.axis('tight')
    ax.set_xlabel("time (n)", fontsize=PlotCaptionFontSize)
    ylabelstr = "y(n)"
    ax.set_ylabel(ylabelstr, fontsize=PlotCaptionFontSize)
    titlestr = "EEG Delta Wave - pa:{:d}, ch:{:d}, tr:{:d}".format(participant_no, eeg_ch_no,eeg_trial_no)
    ax.set_title(titlestr, fontsize=PlotCaptionFontSize)
    #ax.set_ylim(Ymin,Ymax)
    for spine in plt.gca().spines.values():
        spine.set_visible(True)
    if DoSave == 1:
        fig.savefig(OutputFilePNGA, format='png', dpi=300, bbox_inches='tight', transparent=True)
        #fig.savefig(OutputFileJPGA, format='jpg', dpi=300, bbox_inches='tight', transparent=True)
        # fig.savefig(OutputFileEPSA, format='eps', dpi=300, bbox_inches='tight', transparent=True, pad_inches=.25)

    plt.show()

    #-----------------------------------------------
    # Plot theta wave
    #-----------------------------------------------
    OutputFilePNGA = "eeg_theta{:d}_{:d}_{:d}.png".format(participant_no, eeg_ch_no,eeg_trial_no)
    # OutputFileJPGA = "Wdtheta{:s}_{:d}_{:d}_{:d}_{:d}_{:d}_{:d}_{:d}.jpg".format(FileID, Nx1, n, Ns, Noverlap, Ntoppercent, CorpusStart, CorpusEnd)
    # OutputFileEPSA = "Wdtheta{:s}_{:d}_{:d}_{:d}_{:d}_{:d}_{:d}_{:d}.eps".format(FileID, Nx1, n, Ns, Noverlap, Ntoppercent, CorpusStart, CorpusEnd)

    fig = plt.figure(figsize=(9, 3))
    ax = fig.add_axes([0.12, 0.17, 0.8, 0.73])
    ax.grid(True,which='major',axis='both',alpha=0.3)
    p4 = ax.plot(y_theta, label='Theta wave')
    ax.axis('tight')
    ax.set_xlabel("time (n)", fontsize=PlotCaptionFontSize)
    ylabelstr = "y(n)"
    ax.set_ylabel(ylabelstr, fontsize=PlotCaptionFontSize)
    titlestr = "EEG Theta Wave - pa:{:d}, ch:{:d}, tr:{:d}".format(participant_no, eeg_ch_no,eeg_trial_no)
    ax.set_title(titlestr, fontsize=PlotCaptionFontSize)
    #ax.set_ylim(Ymin,Ymax)
    for spine in plt.gca().spines.values():
        spine.set_visible(True)
    if DoSave == 1:
        fig.savefig(OutputFilePNGA, format='png', dpi=300, bbox_inches='tight', transparent=True)
        #fig.savefig(OutputFileJPGA, format='jpg', dpi=300, bbox_inches='tight', transparent=True)
        # fig.savefig(OutputFileEPSA, format='eps', dpi=300, bbox_inches='tight', transparent=True, pad_inches=.25)

    plt.show()

    #-----------------------------------------------
    # Plot Alpha wave
    #-----------------------------------------------
    OutputFilePNGA = "eeg_alpha{:d}_{:d}_{:d}.png".format(participant_no, eeg_ch_no,eeg_trial_no)
    # OutputFileJPGA = "Wdtheta{:s}_{:d}_{:d}_{:d}_{:d}_{:d}_{:d}_{:d}.jpg".format(FileID, Nx1, n, Ns, Noverlap, Ntoppercent, CorpusStart, CorpusEnd)
    # OutputFileEPSA = "Wdtheta{:s}_{:d}_{:d}_{:d}_{:d}_{:d}_{:d}_{:d}.eps".format(FileID, Nx1, n, Ns, Noverlap, Ntoppercent, CorpusStart, CorpusEnd)

    fig = plt.figure(figsize=(9, 3))
    ax = fig.add_axes([0.12, 0.17, 0.8, 0.73])
    ax.grid(True,which='major',axis='both',alpha=0.3)
    p4 = ax.plot(y_alpha, label='Alpha wave')
    ax.axis('tight')
    ax.set_xlabel("time (n)", fontsize=PlotCaptionFontSize)
    ylabelstr = "y(n)"
    ax.set_ylabel(ylabelstr, fontsize=PlotCaptionFontSize)
    titlestr = "EEG Alpha Wave - pa:{:d}, ch:{:d}, tr:{:d}".format(participant_no, eeg_ch_no,eeg_trial_no)
    ax.set_title(titlestr, fontsize=PlotCaptionFontSize)
    #ax.set_ylim(Ymin,Ymax)
    for spine in plt.gca().spines.values():
        spine.set_visible(True)
    if DoSave == 1:
        fig.savefig(OutputFilePNGA, format='png', dpi=300, bbox_inches='tight', transparent=True)
        #fig.savefig(OutputFileJPGA, format='jpg', dpi=300, bbox_inches='tight', transparent=True)
        # fig.savefig(OutputFileEPSA, format='eps', dpi=300, bbox_inches='tight', transparent=True, pad_inches=.25)

    plt.show()

    #-----------------------------------------------
    # Plot Beta wave
    #-----------------------------------------------
    OutputFilePNGA = "eeg_beta{:d}_{:d}_{:d}.png".format(participant_no, eeg_ch_no,eeg_trial_no)
    # OutputFileJPGA = "Wdtheta{:s}_{:d}_{:d}_{:d}_{:d}_{:d}_{:d}_{:d}.jpg".format(FileID, Nx1, n, Ns, Noverlap, Ntoppercent, CorpusStart, CorpusEnd)
    # OutputFileEPSA = "Wdtheta{:s}_{:d}_{:d}_{:d}_{:d}_{:d}_{:d}_{:d}.eps".format(FileID, Nx1, n, Ns, Noverlap, Ntoppercent, CorpusStart, CorpusEnd)

    fig = plt.figure(figsize=(9, 3))
    ax = fig.add_axes([0.12, 0.17, 0.8, 0.73])
    ax.grid(True,which='major',axis='both',alpha=0.3)
    p4 = ax.plot(y_beta, label='Beta wave')
    ax.axis('tight')
    ax.set_xlabel("time (n)", fontsize=PlotCaptionFontSize)
    ylabelstr = "y(n)"
    ax.set_ylabel(ylabelstr, fontsize=PlotCaptionFontSize)
    titlestr = "EEG Beta Wave - pa:{:d}, ch:{:d}, tr:{:d}".format(participant_no, eeg_ch_no,eeg_trial_no)
    ax.set_title(titlestr, fontsize=PlotCaptionFontSize)
    #ax.set_ylim(Ymin,Ymax)
    for spine in plt.gca().spines.values():
        spine.set_visible(True)
    if DoSave == 1:
        fig.savefig(OutputFilePNGA, format='png', dpi=300, bbox_inches='tight', transparent=True)
        #fig.savefig(OutputFileJPGA, format='jpg', dpi=300, bbox_inches='tight', transparent=True)
        # fig.savefig(OutputFileEPSA, format='eps', dpi=300, bbox_inches='tight', transparent=True, pad_inches=.25)

    plt.show()

    #-----------------------------------------------
    # Plot theta wave
    #-----------------------------------------------
    OutputFilePNGA = "eeg_gamma{:d}_{:d}_{:d}.png".format(participant_no, eeg_ch_no,eeg_trial_no)
    # OutputFileJPGA = "Wdtheta{:s}_{:d}_{:d}_{:d}_{:d}_{:d}_{:d}_{:d}.jpg".format(FileID, Nx1, n, Ns, Noverlap, Ntoppercent, CorpusStart, CorpusEnd)
    # OutputFileEPSA = "Wdtheta{:s}_{:d}_{:d}_{:d}_{:d}_{:d}_{:d}_{:d}.eps".format(FileID, Nx1, n, Ns, Noverlap, Ntoppercent, CorpusStart, CorpusEnd)

    fig = plt.figure(figsize=(9, 3))
    ax = fig.add_axes([0.12, 0.17, 0.8, 0.73])
    ax.grid(True,which='major',axis='both',alpha=0.3)
    p4 = ax.plot(y_gamma, label='Gamma wave')
    ax.axis('tight')
    ax.set_xlabel("time (n)", fontsize=PlotCaptionFontSize)
    ylabelstr = "y(n)"
    ax.set_ylabel(ylabelstr, fontsize=PlotCaptionFontSize)
    titlestr = "EEG Gamma Wave - pa:{:d}, ch:{:d}, tr:{:d}".format(participant_no, eeg_ch_no,eeg_trial_no)
    ax.set_title(titlestr, fontsize=PlotCaptionFontSize)
    #ax.set_ylim(Ymin,Ymax)
    for spine in plt.gca().spines.values():
        spine.set_visible(True)
    if DoSave == 1:
        fig.savefig(OutputFilePNGA, format='png', dpi=300, bbox_inches='tight', transparent=True)
        #fig.savefig(OutputFileJPGA, format='jpg', dpi=300, bbox_inches='tight', transparent=True)
        # fig.savefig(OutputFileEPSA, format='eps', dpi=300, bbox_inches='tight', transparent=True, pad_inches=.25)

    plt.show()

def plot_EEGWaveInformationTopologyCore(y_alpha, y_theta, labelX, labelY, plot_title=True, participant_no=0, eeg_ch_no=0, eeg_trial_no=0, Feature0=0):
    # ===========================================================================
    #
    # Calculate EEG Information Topology symbolic information
    #
    # For the purposes of demonstration, we choose 2 dimensions
    # y_alpha, y_theta are the data inputs
    # Note - we name the variables alpha, theta here but they can be fed with any EEG wave vars.
    # ===========================================================================

    M = 4
    ap = 0.0075
    bp = 4.2026
    cp = 4.5362

    y_alpha_norm = min_max_norm(y_alpha)
    y_alpha_symb = symbolise_amplitude_states(y_alpha_norm, ngram=1)
    y_alpha_symbols = y_alpha_symb # symbol_mapping(y_alpha_symb, symb_approach=SYMB_METHOD)

    Nsym_alpha = len(y_alpha_symbols)
    selected_symbol = most_frequent_symbol(y_alpha_symbols)
    Hetheta, palpha, symbols_ranked_alpha = FastEntropy5_czml(y_alpha_symbols, Nsym_alpha, selected_symbol, M, ap, bp, cp, zml_model='zml', eta=0.2)
    Information_alpha = -np.log2(palpha)

    y_theta_norm = min_max_norm(y_theta)
    y_theta_symbols = symbolise_amplitude_states(y_theta_norm, ngram=1)

    Nsym_theta = len(y_theta_symbols)
    selected_symbol = most_frequent_symbol(y_theta_symbols)
    Hetheta, ptheta, symbols_ranked_theta = FastEntropy5_czml(y_theta_symbols, Nsym_theta, selected_symbol, M, ap, bp, cp, zml_model='zml', eta=0.2)
    Information_theta = -np.log2(ptheta)

    # This is where learning of the locked symbol rankings should take place.
    # For this experiment, we simply use the first run through as the learned rankings
    #
    if first_run == 1:
        symbols_ranked_alpha_learned = symbols_ranked_alpha
        symbols_ranked_theta_learned = symbols_ranked_theta

    # Now run through the data and for this experiment, we use 'eeg_block_n' second blocks
    # eg eeg_block_n = 1
    y_alpha_symbols_blockwise = GetBlockInformation(y_alpha, eeg_block_n, M, ap, bp, cp)
    y_theta_symbols_blockwise = GetBlockInformation(y_theta, eeg_block_n, M, ap, bp, cp)
    y_alpha_symbols_blockwise_int = [eval(i) for i in y_alpha_symbols_blockwise]
    y_theta_symbols_blockwise_int = [eval(i) for i in y_theta_symbols_blockwise]
    # Get the count for each:
    Nsb = len(y_alpha_symbols_blockwise)
    counter = collections.Counter(y_alpha_symbols_blockwise)
    symbols_local = counter.most_common(Nsb)
    symbols_ranked = [num[0] for num in symbols_local]
    symbols_ranked_int = [eval(i) for i in symbols_ranked]
    symbols_ranked_count = [num[1] for num in symbols_local]
    min_size = 70
    max_size = 500
    symbols_marker_size = GetMarkerSizeFromFrequency(y_alpha_symbols_blockwise, min_size, max_size)
    Nsyms = len(symbols_marker_size)
    rng = np.random.RandomState(0)
    colors = rng.rand(Nsyms)

    Ny = len(y_alpha_symbols_blockwise_int)

    xmargin = 0.4
    ymargin = 0.2
    Xmin = min(y_alpha_symbols_blockwise_int) - xmargin
    Xmax = max(y_alpha_symbols_blockwise_int) + xmargin

    Ymin = min(y_theta_symbols_blockwise_int) - ymargin
    Ymax = max(y_theta_symbols_blockwise_int) + ymargin

    # ===========================================================================
    #
    # Plot EEG information topology trajectories for one channel
    #
    # ===========================================================================

    arrow_args = dict(arrowstyle="->",
                      color='black',
                      lw=1.5,
                      ls='-')

    OutputFilePNGA = "eegit_{:s}_{:s}_{:d}_{:d}_{:d}_{:d}.png".format(labelX, labelY, participant_no, eeg_ch_no,eeg_trial_no, Feature0)
    # OutputFileJPGA = "Wdtheta{:s}_{:d}_{:d}_{:d}_{:d}_{:d}_{:d}_{:d}.jpg".format(FileID, Nx1, n, Ns, Noverlap, Ntoppercent, CorpusStart, CorpusEnd)
    # OutputFileEPSA = "Wdtheta{:s}_{:d}_{:d}_{:d}_{:d}_{:d}_{:d}_{:d}.eps".format(FileID, Nx1, n, Ns, Noverlap, Ntoppercent, CorpusStart, CorpusEnd)

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_axes([0.12, 0.12, 0.82, 0.78])
    ax.grid(True,which='major',axis='both',alpha=0.3)

    p1 = ax.scatter(y_alpha_symbols_blockwise_int, y_theta_symbols_blockwise_int,
                 s=symbols_marker_size,  # marker size
                 alpha=0.9,
                 c=colors,
                 edgecolors='grey',
                 cmap='summer',
                 zorder=2)

    for i in range(Ny-1):
        plt.annotate('',
                     xy=(y_alpha_symbols_blockwise_int[i+1], y_theta_symbols_blockwise_int[i+1]),
                     xytext=(y_alpha_symbols_blockwise_int[i], y_theta_symbols_blockwise_int[i]),
                     ha="center", va="center",
                     arrowprops=dict(connectionstyle="arc3,rad=-0.2",
                                     **arrow_args))

    #xlabelstr = "Symbols (n)" # {:s} H = {:1.1f}, W = {:1.1f}".format(WordID, He, Wd)
    ax.set_xlabel(labelX)
    #ylabelstr = "y(t) - EEG Ch({:d}), trial ({:d})".format(eeg_ch_no,eeg_trial_no)
    ylabelstr = labelY
    ax.set_ylabel(ylabelstr)
    if plot_title:
        titlestr = "EEG Information Topology ({:s}-{:s}) - pa:{:d}, ch:{:d}, tr:{:d}, fv:{:d}".format(labelX, labelY, participant_no, eeg_ch_no,eeg_trial_no, Feature0)
        ax.set_title(titlestr)

    #
    # ax.set_ylabel(r'$\tilde{\theta}_{\mathrm{i}}(\phi;n,M)$', fontsize=PlotCaptionFontSize)
    #
    # ax.grid(False)
    #
    Xmin = -0.5
    Xmax = 3.5
    Ymin = -0.5
    Ymax = 3.5
    ax.set_xlim(Xmin,Xmax)
    ax.set_ylim(Ymin,Ymax)

    # Add all spines
    for spine in plt.gca().spines.values():
        spine.set_visible(True)

    if DoSave == 1:
        fig.savefig(OutputFilePNGA, format='png', dpi=300, bbox_inches='tight', transparent=True)
        #fig.savefig(OutputFileJPGA, format='jpg', dpi=300, bbox_inches='tight', transparent=True)
        # fig.savefig(OutputFileEPSA, format='eps', dpi=300, bbox_inches='tight', transparent=True, pad_inches=.25)

    plt.show()

    print(Information_theta)
    print(Hetheta)
    print(ptheta)
    print(symbols_ranked_theta)

# def CalculateEEGWaves(eeg_ch_data):
#     # ===========================================================================


def plot_EEGInformationTopologyMain(eeg_ch_data, plot_title, participant_no, eeg_ch_no, eeg_trial_no, Feature0):
    # ===========================================================================
    #
    # Calculate EEG Information Topology
    #
    # For the purposes of demonstration, we choose 2 dimensions
    #
    # ===========================================================================

    #-----------------------------------------------
    # Calculate alpha,beta,delta,gamma waves
    #-----------------------------------------------
    y_delta = butter_bandpass_filter_eeg(eeg_ch_data, 'delta', SAMP_RATE_Hz, order=6)
    y_theta = butter_bandpass_filter_eeg(eeg_ch_data, 'theta', SAMP_RATE_Hz, order=6)
    y_alpha = butter_bandpass_filter_eeg(eeg_ch_data, 'alpha', SAMP_RATE_Hz, order=6)
    y_beta  = butter_bandpass_filter_eeg(eeg_ch_data, 'beta', SAMP_RATE_Hz, order=6)
    y_gamma = butter_bandpass_filter_eeg(eeg_ch_data, 'gamma', SAMP_RATE_Hz, order=6)

    labelX = "alpha"
    labelY = "gamma"
    plot_EEGWaveInformationTopologyCore(y_alpha, y_gamma, labelX, labelY,  plot_title, participant_no, eeg_ch_no, eeg_trial_no, Feature0)


    labelX = "alpha"
    labelY = "theta"
    plot_EEGWaveInformationTopologyCore(y_alpha, y_theta, labelX, labelY, plot_title, participant_no, eeg_ch_no, eeg_trial_no, Feature0)

def DEPRECATEDplot_EEGInformationTopology(participant_no, eeg_ch_no,eeg_trial_no,eeg_ch_data, Feature0):
    # ===========================================================================
    #
    # Calculate EEG Information Topology
    #
    # For the purposes of demonstration, we choose 2 dimensions
    #
    # ===========================================================================

    #-----------------------------------------------
    # Calculate alpha,beta,delta,gamma waves
    #-----------------------------------------------
    y_delta = butter_bandpass_filter_eeg(eeg_ch_data, 'delta', SAMP_RATE_Hz, order=6)
    y_theta = butter_bandpass_filter_eeg(eeg_ch_data, 'theta', SAMP_RATE_Hz, order=6)
    y_alpha = butter_bandpass_filter_eeg(eeg_ch_data, 'alpha', SAMP_RATE_Hz, order=6)
    y_beta  = butter_bandpass_filter_eeg(eeg_ch_data, 'beta', SAMP_RATE_Hz, order=6)
    y_gamma = butter_bandpass_filter_eeg(eeg_ch_data, 'gamma', SAMP_RATE_Hz, order=6)

    #
    M = 4
    ap = 0.0075
    bp = 4.2026
    cp = 4.5362

    y_alpha_norm = min_max_norm(y_alpha)
    y_alpha_symb = symbolise_amplitude_states(y_alpha_norm, ngram=1)
    y_alpha_symbols = y_alpha_symb # symbol_mapping(y_alpha_symb, symb_approach=SYMB_METHOD)

    Nsym_alpha = len(y_alpha_symbols)
    selected_symbol = most_frequent_symbol(y_alpha_symbols)
    Hetheta, palpha, symbols_ranked_alpha = FastEntropy5_czml(y_alpha_symbols, Nsym_alpha, selected_symbol, M, ap, bp, cp, zml_model='zml', eta=0.2)
    Information_alpha = -np.log2(palpha)

    y_theta_norm = min_max_norm(y_theta)
    y_theta_symbols = symbolise_amplitude_states(y_theta_norm, ngram=1)

    Nsym_theta = len(y_theta_symbols)
    selected_symbol = most_frequent_symbol(y_theta_symbols)
    Hetheta, ptheta, symbols_ranked_theta = FastEntropy5_czml(y_theta_symbols, Nsym_theta, selected_symbol, M, ap, bp, cp, zml_model='zml', eta=0.2)
    Information_theta = -np.log2(ptheta)

    # This is where learning of the locked symbol rankings should take place.
    # For this experiment, we simply use the first run through as the learned rankings
    #
    if first_run == 1:
        symbols_ranked_alpha_learned = symbols_ranked_alpha
        symbols_ranked_theta_learned = symbols_ranked_theta

    # Now run through the data and for this experiment, we use 'eeg_block_n' second blocks
    # eg eeg_block_n = 1
    y_alpha_symbols_blockwise = GetBlockInformation(y_alpha, eeg_block_n, M, ap, bp, cp)
    y_theta_symbols_blockwise = GetBlockInformation(y_theta, eeg_block_n, M, ap, bp, cp)
    y_alpha_symbols_blockwise_int = [eval(i) for i in y_alpha_symbols_blockwise]
    y_theta_symbols_blockwise_int = [eval(i) for i in y_theta_symbols_blockwise]
    # Get the count for each:
    Nsb = len(y_alpha_symbols_blockwise)
    counter = collections.Counter(y_alpha_symbols_blockwise)
    symbols_local = counter.most_common(Nsb)
    symbols_ranked = [num[0] for num in symbols_local]
    symbols_ranked_int = [eval(i) for i in symbols_ranked]
    symbols_ranked_count = [num[1] for num in symbols_local]
    min_size = 70
    max_size = 500
    symbols_marker_size = GetMarkerSizeFromFrequency(y_alpha_symbols_blockwise, min_size, max_size)
    Nsyms = len(symbols_marker_size)
    rng = np.random.RandomState(0)
    colors = rng.rand(Nsyms)

    Ny = len(y_alpha_symbols_blockwise_int)

    xmargin = 0.4
    ymargin = 0.2
    Xmin = min(y_alpha_symbols_blockwise_int) - xmargin
    Xmax = max(y_alpha_symbols_blockwise_int) + xmargin

    Ymin = min(y_theta_symbols_blockwise_int) - ymargin
    Ymax = max(y_theta_symbols_blockwise_int) + ymargin

    # ===========================================================================
    #
    # Plot EEG information topology trajectories for one channel
    #
    # ===========================================================================

    arrow_args = dict(arrowstyle="->",
                      color='black',
                      lw=1.5,
                      ls='-')

    OutputFilePNGA = "eegitalpha_theta{:d}_{:d}_{:d}_{:d}.png".format(participant_no, eeg_ch_no,eeg_trial_no, Feature0)
    # OutputFileJPGA = "Wdtheta{:s}_{:d}_{:d}_{:d}_{:d}_{:d}_{:d}_{:d}.jpg".format(FileID, Nx1, n, Ns, Noverlap, Ntoppercent, CorpusStart, CorpusEnd)
    # OutputFileEPSA = "Wdtheta{:s}_{:d}_{:d}_{:d}_{:d}_{:d}_{:d}_{:d}.eps".format(FileID, Nx1, n, Ns, Noverlap, Ntoppercent, CorpusStart, CorpusEnd)

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_axes([0.12, 0.12, 0.82, 0.78])
    ax.grid(True,which='major',axis='both',alpha=0.3)

    p1 = ax.scatter(y_alpha_symbols_blockwise_int, y_theta_symbols_blockwise_int,
                 s=symbols_marker_size,  # marker size
                 alpha=0.9,
                 c=colors,
                 edgecolors='grey',
                 cmap='summer',
                 zorder=2)

    for i in range(Ny-1):
        plt.annotate('',
                     xy=(y_alpha_symbols_blockwise_int[i+1], y_theta_symbols_blockwise_int[i+1]),
                     xytext=(y_alpha_symbols_blockwise_int[i], y_theta_symbols_blockwise_int[i]),
                     ha="center", va="center",
                     arrowprops=dict(connectionstyle="arc3,rad=-0.2",
                                     **arrow_args))

    #xlabelstr = "Symbols (n)" # {:s} H = {:1.1f}, W = {:1.1f}".format(WordID, He, Wd)
    ax.set_xlabel("alpha")
    #ylabelstr = "y(t) - EEG Ch({:d}), trial ({:d})".format(eeg_ch_no,eeg_trial_no)
    ylabelstr = "theta"
    ax.set_ylabel(ylabelstr)

    titlestr = "EEG Information Topology - pa:{:d}, ch:{:d}, tr:{:d}, fv:{:d}".format(participant_no, eeg_ch_no,eeg_trial_no, Feature0)
    ax.set_title(titlestr)

    #
    # ax.set_ylabel(r'$\tilde{\theta}_{\mathrm{i}}(\phi;n,M)$', fontsize=PlotCaptionFontSize)
    #
    # ax.grid(False)
    #
    Xmin = -0.5
    Xmax = 3.5
    Ymin = -0.5
    Ymax = 3.5
    ax.set_xlim(Xmin,Xmax)
    ax.set_ylim(Ymin,Ymax)

    # Add all spines
    for spine in plt.gca().spines.values():
        spine.set_visible(True)

    if DoSave == 1:
        fig.savefig(OutputFilePNGA, format='png', dpi=300, bbox_inches='tight', transparent=True)
        #fig.savefig(OutputFileJPGA, format='jpg', dpi=300, bbox_inches='tight', transparent=True)
        # fig.savefig(OutputFileEPSA, format='eps', dpi=300, bbox_inches='tight', transparent=True, pad_inches=.25)

    plt.show()

    print(Information_theta)
    print(Hetheta)
    print(ptheta)
    print(symbols_ranked_theta)


#==================================================================================
#
# Main function
#
#==================================================================================
#
# import matplotlib.pyplot as plt
# from matplotlib.legend_handler import HandlerPatch
# import matplotlib.patches as patches
# from matplotlib.lines import Line2D
#
# import numpy as np
#

def run_experiment_infotop(mat_files):
    eeg_data = load_eeg_data(mat_files, max_participants)
    symbolise_all_participant_trials(eeg_data, time_plot_n)



if __name__ == "__main__":



    # participant_file_path = r"C:\data\eeg\DEAP\participant_questionnaire.csv"
    # participant_info = load_participant_info(participant_file_path)

    #eeg_data_folder = r"C:\data\eeg\DEAP\data_preprocessed_matlab"
    mat_files = ['s01.mat', 's02.mat', 's03.mat', 's04.mat', 's05.mat', ]
    run_experiment_infotop(mat_files)
    #eeg_data = load_eeg_data(mat_files, max_participants)

    ############################################################
    #  Symbolize the EEG data
    ############################################################
    #participant_eeg_matrix = symbolise_all_participant_trials(eeg_data, time_plot_n)

  
