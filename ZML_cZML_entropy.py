import numpy as np
from ZML_cZML_distribution import *
import collections
from collections import Counter

def gen_symbol_probability(word_list, plot = 0):
    word_frequency = Counter(word_list)
    dict_word_freq = {}
    for word, count in word_frequency.items():
        dict_word_freq[word] = count
        
    # Calculate the probability of each word
    total_frequency = sum(dict_word_freq.values())
    word_probabilities = {word: frequency / total_frequency for word, frequency in dict_word_freq.items()}
    # build word probability dictionary
    dict_word_prob = {}
    for word, probability in word_probabilities.items():
        dict_word_prob[word] = probability
    
    if plot:
        words = list(word_probabilities.keys())
        probabilities = list(word_probabilities.values())
        plt.figure(figsize=(8, 3))
        plt.bar(words, probabilities)
        plt.xlabel('Words')
        plt.ylabel('Probability')
        plt.title('Word Probabilities')
        plt.xticks(rotation=90)  # Rotate x-axis labels for readability
        plt.tight_layout()
        plt.show()
        
    return dict_word_prob
#==================================================================================
# Rui's comment:
# This stays the samem, no changes
#   
def ShannonEntropy(p):
    # p : array of probability
    M = len(p)
    He = 0
    # Iterate over probabilities
    for r in range(M):
        He = He + (p[r] * np.log2(p[r]))
    He = -He
    return He
#==================================================================================
# Rui's comment:
# This stays the samem, no changes
#   
def CalcZMLEntropy(M):
    M1=M+1
    a=np.log2(M1)/np.log2(M)
    beta=M/M1
    gamma=(M**(a-1))/(M-1)**a
    p=np.zeros((1,M))
    psum=0
    for i in range(1,M+1):
        Pr=gamma/((i+beta)**a)
        p[0][i-1]=Pr
        psum=psum+Pr
        if i<=5:
            pass
    knorm=psum
    psum=0
    for i in range(1,M+1):
        Pr=p[0][i-1]/knorm
        p[0][i-1]=Pr
        psum=psum+Pr
        if i<=5:
            pass
    pc=np.cumsum(p)
    Ha=0
    for i in range(1,M+1):
        Ha=Ha+p[0][i-1]*np.log2(p[0][i-1])
    He=-Ha
    output=[He,pc]
    return output


#==================================================================================
# Rui's comment:
# Calculate cZML model 1 and 2 entropy
# default model is model 1, pass model=2 and correct eta if using model 2

def Calc_constrained_ZMLEntropy_model(M, model=1, eta=0.2):
    if model == 1: 
        p = Generate_cZML_Distribution_1(M)
    else:
        p = Generate_cZML_Distribution_2(M, eta)

    p = [p]
    pc=np.cumsum(p)
    Ha=0
    for i in range(1,M+1):
        Ha=Ha+p[0][i-1]*np.log2(p[0][i-1])
    He=-Ha
    output=[He,pc]
    return output


#==================================================================================
# Rui's comment:
# This stays the samem, no changes
#  

def compare_rank_distance3(symbols, Na, selected_symbol):
    # """Compares the distances between a given symbol thor
    #
    # :param symbols:
    # :param selected_symbol:
    # :return:
    # """
    # """
    #     Should we use a more general term for this?
    #     My understanding is this is essentially our P(rank.i)
    #     So could there be other measures by which we find P(rank.i) other than comparing distance?
    # """
    #
    # Note:
    #        5/05/2021 the comprankdistance2 function gives Da = [4 4 5 7], the
    #        first output seems to be in error and hence is corrected in
    #         comprankdistance3.m snd is used in fadstentropyblock4.m
    #
    #        21/12/2021 add error checking for selected symbol not present. In this case
    #                   move to next ranked symbol, in this case we search for the


    s = set(symbols)
    if selected_symbol not in s:
        #  try next ranked symbol
        counter = collections.Counter(symbols)
        symbols_local = counter.most_common(3)
        selected_symbol = symbols_local[0][0]
        xx = 1
        # SymHistsort = np.sort(SymHist)[::-1]
        # SymHistsortI = np.argsort(SymHist)[::-1]
        # # Sort the list:
        # TheListsort = uTheList[SymHistsortI]

    Na1 = len(symbols)
    CoinIndex = Na * np.ones((Na)) # indices of coincidences
    symbol_count = 0

    for i in range(Na - 1):
        if selected_symbol == symbols[i]:
            CoinIndex[symbol_count] = i+1
            symbol_count = symbol_count + 1

    DistancesRaw = np.diff(CoinIndex)
    Ncoin = symbol_count # make space for the first one and we drop one for differences
    DistancesRaw = np.array(DistancesRaw[0:Ncoin])

    # correct for definition of Coincidence Distance to count from 1 at first symbol
    is_empty = DistancesRaw.size == 0
    if not is_empty:
        Distances = np.array(DistancesRaw)
        for i in range(Ncoin-1):
            val = DistancesRaw[i]
            Distances[i+1] = val + 1
        #
        # Now, add the first coincidence back in
        Distances[0] = CoinIndex[0]
    else:
        Distances = np.array(DistancesRaw)

    # Notes: (a) Returns empty array if no coincidences
    #        (b) Could make this -1 so we can test later.

    MeanDistance = np.mean(Distances)
    return MeanDistance


#==================================================================================
# Rui's comment:
# This compute fast entropy based on FastEntropy4 using czml model 1 and 2
#  

def FastEntropy4_czml(symbols, Naw, selected_symbol, M, ap, bp, cp, zml_model='zml', eta=0.2):
    # set zml_model
    # zml_model='zml': using normal zml, 
    # zml_model='czml1': using cZML model 1,
    # zml_model='czml2': using cZML model 2

    distance = compare_rank_distance3(symbols, Naw, selected_symbol)
    #Ncoin = len(distance)
    is_empty = distance.all() == 0
    if is_empty:
        print("Selected Symbol Not Found")
        return 0
    R = 1
    k_est = ap * (np.power(distance, bp)) + cp
    #print('*A) For calculated Dmean={}, we have Kest={},\n'.format(mean_rank_distance, k_est))
    a2, beta2, gamma2 = GetZMLParams(k_est, model=zml_model, eta=eta)

    p = np.zeros(M)
    p_sum = 0
    rangeM = range(M)
    for r in range(M):
        Pr = gamma2 / np.power(((r+1) + beta2), a2)
        p[r] = Pr
        p_sum = p_sum + Pr
    k_norm = p_sum
    p_sum = 0
    for r in range(M):
        Pr = p[r] / k_norm
        p[r] = Pr
        p_sum = p_sum + Pr
    He = 0
    for r in range(M):
        He = He + (p[r] * np.log2(p[r]))
        #print('r: {}  p[r] = {}   log2(p[r]) = {}    He = {} \n'.format(r,p[r], numpy.log2(p[r]), He))
    Hea = -He
    return [Hea, p]

#==================================================================================
# Rui's comment:
# This compute fast ngram entropy based on FastEntropyNgram using czml model 1 and 2
#  
    # set zml_model
    # zml_model='zml': using normal zml, 
    # zml_model='czml1': using cZML model 1,
    # zml_model='czml2': using cZML model 2

def FastEntropyNgram_zml(symbols, Naw, selected_symbol, M, ap, bp, cp, zml_model='zml', eta=0.2):
    #
    # Calculates the entropy value of a given set of samples
    #
    # Args:
    #     symbols (numpy.array or list (int)):  A list of symbols.
    #     selected_symbol (int or char): A selected symbol from the symbol set.
    #     M (int): Number of distinct symbols in the set.
    #     ap (int): the a value for the model D used to estimate M.
    #     bp (int): the b value for the model D used to estimate M.
    #     cp (int): the c value for the model D used to estimate M.
    #
    # Results:
    #     float: The entropy value
    #
    #

    mean_rank_distance = compare_rank_distance3(symbols, Naw, selected_symbol)
    #mean_rank_distance = compare_rank_distance(symbols, selected_symbol)
    if mean_rank_distance == 0:
        print("Selected Symbol Not Found")
        return 0
    R = 1
    k_est = ap * (np.power(mean_rank_distance, bp)) + cp
    #print('*A) For calculated Dmean={}, we have Kest={},\n'.format(mean_rank_distance, k_est))
    a2, beta2, gamma2 = GetZMLParams(k_est, model=zml_model, eta=eta)

    p = np.zeros(M)
    p_sum = 0
    rangeM = range(M)
    for r in range(M):
        Pr = gamma2 / np.power(((r+1) + beta2), a2)
        p[r] = Pr
        p_sum = p_sum + Pr
    k_norm = p_sum
    p_sum = 0
    for r in range(M):
        Pr = p[r] / k_norm
        p[r] = Pr
        p_sum = p_sum + Pr
    He = 0
    for r in range(M):
        He = He + (p[r] * np.log2(p[r]))
        #print('r: {}  p[r] = {}   log2(p[r]) = {}    He = {} \n'.format(r,p[r], numpy.log2(p[r]), He))
    Hea = -He
    return [Hea, p, a2, beta2, gamma2]


def frequent_words(words):

    word_counts = Counter(words)
    common_words = word_counts.most_common()
    return common_words

# Function to create a list of tuples
def create_list_of_tuples(lst1, lst2):
    result = []  # Empty list to store the tuples
    for i in range(len(lst1)):
        # Create a tuple from corresponding elements
        tuple_element = (lst1[i], lst2[i])
        result.append(tuple_element)  # Append the tuple to the list
    return result

    # Function to split a list of tuples
def split_list_of_tuples(listoftuples):
    M = len(listoftuples)
    list_key = []
    list_values = []
    for x in range(M):
        (key, val) = listoftuples[x]
        list_key.append(key)
        list_values.append(val)

    return list_key, list_values

def compare_rank_distance5(symbols, Na, selected_symbol):
    # Compares the distances between a given symbol

    s = set(symbols)

    Ns = len(symbols)
    counter = collections.Counter(symbols)
    symbols_local = counter.most_common(Ns)
    symbols_ranked = [num[0] for num in symbols_local]

    if selected_symbol not in s:
        #  try next ranked symbol
        counter = collections.Counter(symbols)
        symbols_local = counter.most_common(3)
        selected_symbol = symbols_local[0][0]

        # SymHistsort = np.sort(SymHist)[::-1]
        # SymHistsortI = np.argsort(SymHist)[::-1]
        # # Sort the list:
        # TheListsort = uTheList[SymHistsortI]

    Na1 = len(symbols)
    CoinIndex = Na * np.ones((Na)) # indices of coincidences
    symbol_count = 0

    for i in range(Na - 1):
        if selected_symbol == symbols[i]:
            CoinIndex[symbol_count] = i+1
            symbol_count = symbol_count + 1

    DistancesRaw = np.diff(CoinIndex)
    Ncoin = symbol_count # make space for the first one and we drop one for differences
    DistancesRaw = np.array(DistancesRaw[0:Ncoin])

    # correct for definition of Coincidence Distance to count from 1 at first symbol
    is_empty = DistancesRaw.size == 0
    if not is_empty:
        Distances = np.array(DistancesRaw)
        for i in range(Ncoin-1):
            val = DistancesRaw[i]
            Distances[i+1] = val + 1
        #
        # Now, add the first coincidence back in
        Distances[0] = CoinIndex[0]
    else:
        Distances = np.array(DistancesRaw)

    # Notes: (a) Returns empty array if no coincidences
    #        (b) Could make this -1 so we can test later.

    MeanDistance = np.mean(Distances)
    return symbols_ranked, MeanDistance

def FastEntropy5_czml(symbols, Naw, selected_symbol, M, ap, bp, cp, zml_model='zml', eta=0.2):
    # set zml_model
    # zml_model='zml': using normal zml, 
    # zml_model='czml1': using cZML model 1,
    # zml_model='czml2': using cZML model 2

    symbols_ranked, distance = compare_rank_distance5(symbols, Naw, selected_symbol)
    #Ncoin = len(distance)
    is_empty = distance.all() == 0
    if is_empty:
        print("Selected Symbol Not Found")
        return 0
    R = 1
    k_est = ap * (np.power(distance, bp)) + cp
    #print('*A) For calculated Dmean={}, we have Kest={},\n'.format(mean_rank_distance, k_est))
    a2, beta2, gamma2 = GetZMLParams(k_est, model=zml_model, eta=eta)

    p = np.zeros(M)
    p_sum = 0
    rangeM = range(M)
    for r in range(M):
        Pr = gamma2 / np.power(((r+1) + beta2), a2)
        p[r] = Pr
        p_sum = p_sum + Pr
    k_norm = p_sum
    p_sum = 0
    for r in range(M):
        Pr = p[r] / k_norm
        p[r] = Pr
        p_sum = p_sum + Pr
    He = 0
    for r in range(M):
        He = He + (p[r] * np.log2(p[r]))
        #print('r: {}  p[r] = {}   log2(p[r]) = {}    He = {} \n'.format(r,p[r], numpy.log2(p[r]), He))
    Hea = -He
    return [Hea, p, symbols_ranked]

def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

def TESTFastEntropyNgramSymbol_zml(symbols, Naw, selected_symbol, locked_ranked_symbols, M, ap, bp, cp, zml_model='zml', eta=0.2):
    #
    # Calculates the entropy value of a given set of samples
    # Same as FastEntropyNgram_zml, except that the symbol corresponding to the pdf is output.
    #
    # Args:
    #     symbols (numpy.array or list (int)):  A list of symbols.
    #     selected_symbol (int or char): A selected symbol from the symbol set.
    #     M (int): Number of distinct symbols in the set.
    #     ap (int): the a value for the model D used to estimate M.
    #     bp (int): the b value for the model D used to estimate M.
    #     cp (int): the c value for the model D used to estimate M.
    #
    # Results:
    #     float: The entropy value
    #
    #

    # get ranked order of symbols from the list
    ranked_symbols = frequent_words(symbols)  # for this block
    #
    # We need to remap the symbols to locked_ranked_symbols

    mean_rank_distance = compare_rank_distance3(symbols, Naw, selected_symbol)
    #mean_rank_distance = compare_rank_distance(symbols, selected_symbol)
    if mean_rank_distance == 0:
        print("Selected Symbol Not Found")
        return 0
    R = 1
    k_est = ap * (np.power(mean_rank_distance, bp)) + cp
    #print('*A) For calculated Dmean={}, we have Kest={},\n'.format(mean_rank_distance, k_est))
    a2, beta2, gamma2 = GetZMLParams(k_est, model=zml_model, eta=eta)

    p = np.zeros(M)
    p_sum = 0
    rangeM = range(M)
    for r in range(M):
        Pr = gamma2 / np.power(((r+1) + beta2), a2)
        p[r] = Pr
        p_sum = p_sum + Pr
    k_norm = p_sum
    p_sum = 0
    for r in range(M):
        Pr = p[r] / k_norm
        p[r] = Pr
        p_sum = p_sum + Pr
    He = 0
    for r in range(M):
        He = He + (p[r] * np.log2(p[r]))
        #print('r: {}  p[r] = {}   log2(p[r]) = {}    He = {} \n'.format(r,p[r], numpy.log2(p[r]), He))
    Hea = -He


    # # Pick select entropies to display
    # idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    # #xkey_sel = [xkey[i] for i in idx]
    # booknames_sel = [booknames[i] for i in idx]

    # Now remap the pdf from ranked_symbols to locked_ranked_symbols
    #
    SortP = []
    SortSymbols = []

    #-----------------------------------
    # test
    key = ['BB', 'BA', 'AA', 'AB']
    val = [24, 12, 7, 2] # should resort to ['AB', 'AA', 'BA', 'BB'], => [2, 7, 12, 24]
    ranked_symbols = create_list_of_tuples(key, val)
    #-----------------------------------
    # test
    key = ['AB', 'AA', 'BA', 'BB']
    val = [59, 35, 15, 6]
    locked_ranked_symbols = create_list_of_tuples(key, val)
    #-----------------------------------

    #-----------------------------------
    # test
    key = ['BB', 'BA', 'AA', 'AB']
    val = [24, 12, 7, 2] # should resort to ['AB', 'BA', 'AA', 'BB'], => [2, 12, 7, 24]
    ranked_symbols = create_list_of_tuples(key, val)
    #-----------------------------------
    # test
    key = ['AB', 'BA', 'AA', 'BB']
    val = [59, 35, 15, 6]
    locked_ranked_symbols = create_list_of_tuples(key, val)
    #-----------------------------------

    #-----------------------------------
    # test
    key = ['BB', 'BA', 'AA', 'AB']
    val = [24, 12, 7, 2] # should resort to ['AA', 'BA', 'AB', 'BB'], => [7, 12, 2, 24]
    ranked_symbols = create_list_of_tuples(key, val)
    #
    # resort to ['AA', 'AB', 'BA', 'BB'] => [7, 2, 12, 24]
    # then resort to ['AA', 'BA', 'AB', 'BB'] => [7, 12, 2, 24]

    #-----------------------------------
    # test
    key = ['AA', 'BA', 'AB', 'BB']
    val = [59, 35, 15, 6]
    locked_ranked_symbols = create_list_of_tuples(key, val)
    #-----------------------------------

    ranked_symbols_key, ranked_symbols_val = split_list_of_tuples(ranked_symbols)
    locked_ranked_symbols_key, locked_ranked_symbols_val = split_list_of_tuples(locked_ranked_symbols)

    # ranked_symbols_key = []
    # ranked_symbols_values = []
    # for x in range(M):
    #     (key, val) = ranked_symbols[x]
    #     ranked_symbols_key.append(key)
    #     ranked_symbols_values.append(val)

    #ranked_symbols_key, ranked_symbols_values = zip(*ranked_symbols)

    # ranked_symbols_key = (x[0] for x in ranked_symbols)
    # ranked_symbols_values = (x[1] for x in ranked_symbols)

    #ranked_symbols_key, ranked_symbols_values in ranked_symbols

    ranked_symbols_sort = np.sort(ranked_symbols_key)
    ranked_symbols_sortI = np.argsort(ranked_symbols_key)
    ranked_locked_symbols_sort = np.sort(locked_ranked_symbols_key)
    ranked_locked_symbols_sortI = np.argsort(locked_ranked_symbols_key)

    ranked_symbols_sortIdesc = ranked_symbols_sortI[::-1]
    ranked_locked_symbols_sortIdesc = ranked_locked_symbols_sortI[::-1]

    # Now sort

    ranked_symbols_val_resort = np.array(ranked_symbols_val)[ranked_symbols_sortI]
    ranked_symbols_val_resort2 = np.array(ranked_symbols_val_resort)[ranked_locked_symbols_sortI]


    # Now remap probability order to locked symbols

    p_resort = np.array(p)[ranked_symbols_sortI]
    p_resort2 = np.array(p_resort)[ranked_locked_symbols_sortI]

    aa = 1

# #---------
#     # ranked_symbols_val_resort = np.array(ranked_symbols_val)[ranked_locked_symbols_sortIdesc]
#     # ranked_symbols_val_resort2 = np.array(ranked_symbols_val_resort)[ranked_locked_symbols_sortIdesc]
#     #
#     # ranked_symbols_val_resort = np.array(ranked_symbols_val)[ranked_symbols_sortIdesc]
#     # ranked_symbols_val_resort2 = np.array(ranked_symbols_val_resort)[ranked_locked_symbols_sortIdesc]
#
#     ranked_symbols_val_resort = np.array(ranked_symbols_val)[ranked_locked_symbols_sortIdesc]
#     ranked_symbols_val_resort2 = np.array(ranked_symbols_val_resort)[ranked_symbols_sortIdesc]    #XXX
#
#     ranked_symbols_val_resort = np.array(ranked_symbols_val)[ranked_symbols_sortIdesc]
#     ranked_symbols_val_resort2 = np.array(ranked_symbols_val_resort)[ranked_symbols_sortIdesc]
# # ---------
# # ---------
#     ranked_symbols_val_resort = np.array(ranked_symbols_val)[ranked_locked_symbols_sortI]
#     ranked_symbols_val_resort2 = np.array(ranked_symbols_val_resort)[ranked_locked_symbols_sortIdesc]
#
#     ranked_symbols_val_resort = np.array(ranked_symbols_val)[ranked_symbols_sortI]
#     ranked_symbols_val_resort2 = np.array(ranked_symbols_val_resort)[ranked_locked_symbols_sortIdesc]
#
#     ranked_symbols_val_resort = np.array(ranked_symbols_val)[ranked_locked_symbols_sortI]
#     ranked_symbols_val_resort2 = np.array(ranked_symbols_val_resort)[ranked_symbols_sortIdesc]
#
#     ranked_symbols_val_resort = np.array(ranked_symbols_val)[ranked_symbols_sortI]
#     ranked_symbols_val_resort2 = np.array(ranked_symbols_val_resort)[ranked_symbols_sortIdesc]
# # ---------
# # ---------
#     ranked_symbols_val_resort = np.array(ranked_symbols_val)[ranked_locked_symbols_sortIdesc]
#     ranked_symbols_val_resort2 = np.array(ranked_symbols_val_resort)[ranked_locked_symbols_sortI]
#
#     ranked_symbols_val_resort = np.array(ranked_symbols_val)[ranked_symbols_sortIdesc]
#     ranked_symbols_val_resort2 = np.array(ranked_symbols_val_resort)[ranked_locked_symbols_sortI]
#
#     ranked_symbols_val_resort = np.array(ranked_symbols_val)[ranked_locked_symbols_sortIdesc]
#     ranked_symbols_val_resort2 = np.array(ranked_symbols_val_resort)[ranked_symbols_sortI]
#
#     ranked_symbols_val_resort = np.array(ranked_symbols_val)[ranked_symbols_sortIdesc]
#     ranked_symbols_val_resort2 = np.array(ranked_symbols_val_resort)[ranked_symbols_sortI]
# # ---------
# # ---------
#     ranked_symbols_val_resort = np.array(ranked_symbols_val)[ranked_locked_symbols_sortI]
#     ranked_symbols_val_resort2 = np.array(ranked_symbols_val_resort)[ranked_locked_symbols_sortI]
#
#     ranked_symbols_val_resort = np.array(ranked_symbols_val)[ranked_symbols_sortI]
#     ranked_symbols_val_resort2 = np.array(ranked_symbols_val_resort)[ranked_locked_symbols_sortI]
#
#     ranked_symbols_val_resort = np.array(ranked_symbols_val)[ranked_locked_symbols_sortI]
#     ranked_symbols_val_resort2 = np.array(ranked_symbols_val_resort)[ranked_symbols_sortI]
#
#     ranked_symbols_val_resort = np.array(ranked_symbols_val)[ranked_symbols_sortI]
#     ranked_symbols_val_resort2 = np.array(ranked_symbols_val_resort)[ranked_symbols_sortI]
# # --------- @@@@@@@@@@@@@@@@@@@@
# # ---------
#
#     ranked_symbols_val_resort = np.array(ranked_symbols_val)[ranked_locked_symbols_sortI]
#     ranked_symbols_val_resort2 = np.array(ranked_symbols_val_resort)[ranked_locked_symbols_sortIdesc]  # QQQ
#
#     ranked_symbols_val_resort = np.array(ranked_symbols_val)[ranked_symbols_sortI]
#     ranked_symbols_val_resort2 = np.array(ranked_symbols_val_resort)[ranked_locked_symbols_sortIdesc]
#
#     ranked_symbols_val_resort = np.array(ranked_symbols_val)[ranked_locked_symbols_sortI]
#     ranked_symbols_val_resort2 = np.array(ranked_symbols_val_resort)[ranked_symbols_sortIdesc]
#
# # # ---
# #
#     ranked_symbols_val_resort = np.array(ranked_symbols_val)[ranked_locked_symbols_sortIdesc]
#     ranked_symbols_val_resort2 = np.array(ranked_symbols_val_resort)[ranked_locked_symbols_sortI]
#
#     ranked_symbols_val_resort = np.array(ranked_symbols_val)[ranked_symbols_sortIdesc]
#     ranked_symbols_val_resort2 = np.array(ranked_symbols_val_resort)[ranked_locked_symbols_sortI]
#
#     ranked_symbols_val_resort = np.array(ranked_symbols_val)[ranked_locked_symbols_sortIdesc]
#     ranked_symbols_val_resort2 = np.array(ranked_symbols_val_resort)[ranked_symbols_sortI]
#
# # # ---
#
#     ranked_symbols_val_resort = np.array(ranked_symbols_val)[ranked_locked_symbols_sortI]
#     ranked_symbols_val_resort2 = np.array(ranked_symbols_val_resort)[ranked_locked_symbols_sortI]
#
#     ranked_symbols_val_resort = np.array(ranked_symbols_val)[ranked_symbols_sortI]
#     ranked_symbols_val_resort2 = np.array(ranked_symbols_val_resort)[ranked_locked_symbols_sortI]  # QQQ
#
#     ranked_symbols_val_resort = np.array(ranked_symbols_val)[ranked_locked_symbols_sortI]
#     ranked_symbols_val_resort2 = np.array(ranked_symbols_val_resort)[ranked_symbols_sortI]
# #
#     # ---
#
#
#
#     ranked_symbols_val_resort = np.array(ranked_symbols_val)[ranked_locked_symbols_sortIdesc]
#     ranked_symbols_val_resort2 = np.array(ranked_symbols_val)[ranked_locked_symbols_sortIdesc]
#
#     ranked_symbols_val_resort2 = np.array(ranked_symbols_val)[ranked_symbols_sortIdesc]

    # for i in range(0, M):
    #     current_symbol = ranked_symbols[i]  # eg C [C,A,D,B]
    #     current_prob = p[i]
    #     current_locked_ranked_symbol = locked_ranked_symbols[i]  # eg A [A,D,B,C]
    #
    #     sym_input = SymbolsMain2[i]  # Symbols (180) -> CIAHBCAGH
    #     symM_ = SymbolsMap[sym_input]
    #     symM = symM_
    #     SymbolsM.append(symM)


    return [Hea, p, p_resort2, a2, beta2, gamma2]

def FastEntropyNgramSymbol_zml(symbols, Naw, selected_symbol, locked_ranked_symbols, M, ap, bp, cp, zml_model='zml', eta=0.2):
    #
    # Calculates the entropy value of a given set of samples
    # Same as FastEntropyNgram_zml, except that the symbol corresponding to the pdf is output.
    #
    # Args:
    #     symbols (numpy.array or list (int)):  A list of symbols.
    #     selected_symbol (int or char): A selected symbol from the symbol set.
    #     M (int): Number of distinct symbols in the set.
    #     ap (int): the a value for the model D used to estimate M.
    #     bp (int): the b value for the model D used to estimate M.
    #     cp (int): the c value for the model D used to estimate M.
    #
    # Results:
    #     float: The entropy value
    #
    #

    # get ranked order of symbols from the list
    ranked_symbols = frequent_words(symbols)  # for this block
    #
    # We need to remap the symbols to locked_ranked_symbols

    mean_rank_distance = compare_rank_distance3(symbols, Naw, selected_symbol)
    #mean_rank_distance = compare_rank_distance(symbols, selected_symbol)
    if mean_rank_distance == 0:
        print("Selected Symbol Not Found")
        return 0
    R = 1
    k_est = ap * (np.power(mean_rank_distance, bp)) + cp
    #print('*A) For calculated Dmean={}, we have Kest={},\n'.format(mean_rank_distance, k_est))
    a2, beta2, gamma2 = GetZMLParams(k_est, model=zml_model, eta=eta)

    p = np.zeros(M)
    p_sum = 0
    rangeM = range(M)
    for r in range(M):
        Pr = gamma2 / np.power(((r+1) + beta2), a2)
        p[r] = Pr
        p_sum = p_sum + Pr
    k_norm = p_sum
    p_sum = 0
    for r in range(M):
        Pr = p[r] / k_norm
        p[r] = Pr
        p_sum = p_sum + Pr
    He = 0
    for r in range(M):
        He = He + (p[r] * np.log2(p[r]))
        #print('r: {}  p[r] = {}   log2(p[r]) = {}    He = {} \n'.format(r,p[r], numpy.log2(p[r]), He))
    Hea = -He

    #--------------------------------------------------------------------------
    # Now remap the pdf from ranked_symbols to locked_ranked_symbols
    #
    ranked_symbols_key, ranked_symbols_val = split_list_of_tuples(ranked_symbols)
    locked_ranked_symbols_key, locked_ranked_symbols_val = split_list_of_tuples(locked_ranked_symbols)

    #ranked_symbols_sort = np.sort(ranked_symbols_key)
    ranked_symbols_sortI = np.argsort(ranked_symbols_key)
    #ranked_locked_symbols_sort = np.sort(locked_ranked_symbols_key)
    ranked_locked_symbols_sortI = np.argsort(locked_ranked_symbols_key)
    # ranked_symbols_sortIdesc = ranked_symbols_sortI[::-1]
    # ranked_locked_symbols_sortIdesc = ranked_locked_symbols_sortI[::-1]

    # Now remap probability order to locked symbols
    #
    p_resort = np.array(p)[ranked_symbols_sortI]
    p_map = np.array(p_resort)[ranked_locked_symbols_sortI]
    #--------------------------------------------------------------------------

    return [Hea, p, p_map, a2, beta2, gamma2]


# Get Ranked Symbols


def GetRankedSymbols_zml(symbols, Naw, selected_symbol, M, ap, bp, cp, zml_model='zml', eta=0.2):
    #
    # Get the ranked set of symbols from a set of given samples
    #
    # Args:
    #     symbols (numpy.array or list (int)):  A list of symbols.
    #     selected_symbol (int or char): A selected symbol from the symbol set.
    #     M (int): Number of distinct symbols in the set.
    #     ap (int): the a value for the model D used to estimate M.
    #     bp (int): the b value for the model D used to estimate M.
    #     cp (int): the c value for the model D used to estimate M.
    #
    # Results:
    #     float: The entropy value
    #
    #

    mean_rank_distance = compare_rank_distance3(symbols, Naw, selected_symbol)
    #mean_rank_distance = compare_rank_distance(symbols, selected_symbol)
    if mean_rank_distance == 0:
        print("Selected Symbol Not Found")
        return 0
    R = 1
    k_est = ap * (np.power(mean_rank_distance, bp)) + cp
    #print('*A) For calculated Dmean={}, we have Kest={},\n'.format(mean_rank_distance, k_est))
    a2, beta2, gamma2 = GetZMLParams(k_est, model=zml_model, eta=eta)

    p = np.zeros(M)
    p_sum = 0
    rangeM = range(M)
    for r in range(M):
        Pr = gamma2 / np.power(((r+1) + beta2), a2)
        p[r] = Pr
        p_sum = p_sum + Pr
    k_norm = p_sum
    p_sum = 0
    for r in range(M):
        Pr = p[r] / k_norm
        p[r] = Pr
        p_sum = p_sum + Pr
    He = 0
    for r in range(M):
        He = He + (p[r] * np.log2(p[r]))
        #print('r: {}  p[r] = {}   log2(p[r]) = {}    He = {} \n'.format(r,p[r], numpy.log2(p[r]), He))
    Hea = -He
    return [Hea, p, a2, beta2, gamma2]

