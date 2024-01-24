import numpy as np
from nltk.tokenize import word_tokenize
import itertools
from collections import Counter

def ismember(A, B):
    """
    Determine which elements of array A are also present in array B.

    Parameters:
    - A: numpy array
    - B: numpy array

    Returns:
    - LIA: numpy array of bool, indicating membership
    - LOCB: numpy array of int, indices of first occurrences in B
    """
    unique_b, inv_idx = np.unique(B, return_inverse=True)
    A_ascii = np.array([ord(char) for char in A])
   # Sort unique_b and inv_idx based on unique_b
    sorted_indices = np.argsort(unique_b)
    unique_b = unique_b[sorted_indices]
    inv_idx = inv_idx[sorted_indices]
    
    lia = np.isin(A_ascii, unique_b)

    #locb = np.where(lia, inv_idx[np.searchsorted(unique_b, A_ascii, sorter=inv_idx)], 0)
    return lia

def remove_non_ASCII_chars(astr, sym_clean_mode, RCSymbolA, RCSymbolD):
    astr_array = np.array(list(astr))  # Convert string to numpy array

    membership = ismember(astr_array, np.arange(123, 513))
    astr_array[membership] = RCSymbolA
    


    if sym_clean_mode == 5:
        membership = ismember(astr_array, np.arange(0, 32))
        astr_array[membership] = RCSymbolA

    elif sym_clean_mode == 6 or sym_clean_mode == 7:
        membership = ismember(astr_array, np.arange(0, 32))
        astr_array[membership] = RCSymbolD
 
        membership = ismember(astr_array, [44])
        astr_array[membership] = RCSymbolA

        membership = ismember(astr_array, [46])
        astr_array[membership] = RCSymbolA
   
        membership = ismember(astr_array, [45])
        astr_array[membership] = RCSymbolA

        membership = ismember(astr_array, [47])
        astr_array[membership] = RCSymbolA

    else:
        # Remove
        membership = ismember(astr_array, np.arange(44, 48))
        astr_array[membership] = RCSymbolA


    membership = ismember(astr_array, np.arange(33, 43))
    astr_array[membership] = RCSymbolA

    membership = ismember(astr_array, np.arange(48, 58))
    astr_array[membership] = RCSymbolA

    membership = ismember(astr_array, np.arange(58, 65))
    astr_array[membership] = RCSymbolA

    membership = ismember(astr_array, np.arange(91, 97))
    astr_array[membership] = RCSymbolA

    result_str = ''.join(astr_array)
    return result_str

def clean_extraneous_characters(astr, RCSymbolB):
    # Replace extraneous characters with .
    astr = astr.replace('?', '.')

    # Replace extraneous characters with space
    astr = astr.replace('-', ' ')
    astr = astr.replace('--', ' ')
    astr = astr.replace('&', ' ')

    # Remove extraneous characters
    astr = astr.replace(';', RCSymbolB)
    astr = astr.replace(':', RCSymbolB)
    astr = astr.replace(',', RCSymbolB)
    astr = astr.replace('(', RCSymbolB)
    astr = astr.replace(')', RCSymbolB)

    astr = astr.replace('/', RCSymbolB)
    astr = astr.replace('[', RCSymbolB)
    astr = astr.replace(']', RCSymbolB)
    astr = astr.replace('^', RCSymbolB)

    astr = astr.replace("'", RCSymbolB)  # apostrophe
    astr = astr.replace('"', RCSymbolB)
    astr = astr.replace('!', RCSymbolB)

    return astr

def clean_symbols_fn(in_array, sym_clean_mode):
    rc_symbol_a = '+'
    rc_symbol_b = ''
    rc_symbol_c = '*'
    rc_symbol_d = ' '
    rc_symbol_e = '['

    # Remove unwanted chars: (  ) \r \n ! " - , , ? ] [ * _ : /
    astr = in_array.replace('+', rc_symbol_b).replace('^', rc_symbol_b)
    

    # Filter letters, convert to lowercase
    astr = astr.lower()

    # Remove non-ASCII chars
    astr = remove_non_ASCII_chars(astr, sym_clean_mode, rc_symbol_a, rc_symbol_d)

    # Replace extraneous characters with fullstop and space and remove extraneous characters
    astr = clean_extraneous_characters(astr, rc_symbol_b)


    if sym_clean_mode == 5:
        astr = astr.replace(' ', rc_symbol_a)

    if sym_clean_mode == 5 or sym_clean_mode == 6 or sym_clean_mode == 7:
        astr = astr.replace('++++++++++++', rc_symbol_a).\
        replace('+++++++++++', rc_symbol_a).replace('++++++++++', rc_symbol_a).\
            replace('+++++++++', rc_symbol_a).replace('++++++++', rc_symbol_a).\
                replace('+++++++', rc_symbol_a).replace('++++++', rc_symbol_a).\
                    replace('+++++', rc_symbol_a).replace('++++', rc_symbol_a).\
                        replace('+++', rc_symbol_a).replace('++', rc_symbol_a).\
                            replace('+', '')

    if sym_clean_mode == 0:
        astr = astr.replace(rc_symbol_a, '')

    if sym_clean_mode == 7:
        astr_array = np.array(list(astr))  # Convert string to numpy array

        membership, _ = ismember(astr_array, [32])
        astr_array[membership] = rc_symbol_e
        astr = ''.join(astr_array)

    return astr

def string_to_symbol(str_len_lst, condition_vec, symbol_vec):
    converted = []
    for l in str_len_lst:
        for i in range(len(condition_vec)):
            if l <= condition_vec[i]:
                converted.append(symbol_vec[i])
                break
        else:
            converted.append(symbol_vec[-1])
    return converted

def test_string_to_symbol():
    conditions = [3, 6, 9, 12]
    letters = ['a', 'b', 'c', 'd', 'e']
    string_list = ["apple", "cat", "dog", "elephant", "fish", "gorilla", "house", "incredible", "adfaerafasfaeadfreawerasef"]
    str_len_lst = [len(s) for s in string_list]
    answer_list = ['b', 'b', 'b', 'c', 'b', 'c', 'b', 'd', 'e']
    converted_list = string_to_symbol(str_len_lst, conditions, letters)
    print(answer_list)
    print(converted_list)


def symbol_to_int(letters, symbol_1st):
    # Generate all possible combinations of letters
    combinations = list(itertools.product(letters, repeat=2))
    # Create a symbol mapping dictionary with unique integers
    symbol_mapping = {combo: idx + 1 for idx, combo in enumerate(combinations)}

    symbols1x = np.array([])
    for i in range(len(symbol_1st) - 1):
        symbol_pair = (symbol_1st[i], symbol_1st[i + 1])
        symbols1x = np.append(symbols1x, symbol_mapping.get(symbol_pair, len(symbol_mapping) + 1))

    return symbols1x


def test_symbol_to_int():
    letters = ['a', 'b', 'c']
    symbol_1st = ['a', 'b', 'a', 'c', 'c', 'b', 'a', 'b', 'b', 'a', 'c']
    int_lst = symbol_to_int(letters, symbol_1st)
    print(int_lst)

def symbolize_by_wordlengths(wordList1, conditions, letters):
    #
    # symbolize words and output wordlength symbols along with corresponding words
    # actual SL symbols
    #
    thesequence = wordList1
    wordLengths1 = [len(w) for w in thesequence]

    symbols1 = string_to_symbol(wordLengths1, conditions, letters)
    symbols1x = symbol_to_int(letters, symbols1)
    symbols1x = symbols1x.astype(int)

    symbseq = []
    for symb in symbols1x:
        symbseq.append(chr(symb + 96))

    return symbols1, symbols1x, symbseq


def gen_symbol_probability(symbol_list):
    symbol_frequency = Counter(symbol_list)
    dict_symbol_freq = {}
    for sym, count in symbol_frequency.items():
        dict_symbol_freq[sym] = count
        
    # Calculate the probability of each word
    total_frequency = sum(dict_symbol_freq.values())
    symbol_probabilities = {s: frequency / total_frequency for s, frequency in dict_symbol_freq.items()}
        
    return symbol_probabilities


def most_frequent_symbol(symbol_list):

    symbol_counts = Counter(symbol_list)
    most_common_symbol = symbol_counts.most_common(1)
    return most_common_symbol[0][0] if most_common_symbol else None


if __name__ == "__main__":
#     ismember('abcde', np.arange(123, 513))
    text = 'Emotion is a psycho-physiological process triggered by conscious and/or unconscious perception of an object or situation and is often associated with mood'
    word_list = word_tokenize(text)

    conditions = [3,7]
    letters = ['A', 'B', 'C']
    symbols, _, _ = symbolize_by_wordlengths(word_list, conditions, letters)
    print(symbols)