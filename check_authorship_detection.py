import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from symbolize_fn import *
from ZML_cZML_entropy import *

def pluginprob2(Xblock):
   # Now compute histogram probabilities of Csymbols
    # - assume first and last chars are spaces
    char_list = list(Xblock)
    char_probability = gen_symbol_probability(char_list)
    shannon_entropy = ShannonEntropy(list(char_probability.values()))

    return shannon_entropy, None, None

def check_authorship_detection(author1_file, author2_file, zml_model='zml', eta=0.2):

# ===================================================
#
#       Set parameters 
#
# ===================================================
    PlotLineWidth = 0.5
    SmallTextFontSize = 12
    PointAlpha = 0.4
    PointLineWidth = 1
    BoxLineWidth = 1.5
    SolutionSpaceAlpha = 0.3

    TextFontSize = 10
    PlotAxesFontSize = 14
    PlotCaptionFontSize = 14
    PlotTitleFontSize = 12
    LegendFontSize = 10
    DoSaveFile = 0  # Flag to indicate if results are saved

    #BaseDir = r'C:/data/corpora/'
    SymRunNo = 6
    RunNo = 1

#===================================================================   
#    1
#    Read in raw text files and symbolize
#    
#    Asym = Sherlock Holmes
#    Bsym = Dr Seuss
#
#===================================================================  

    SymCleanMode = 0  # Don't remove spaces, cr, newline

    #BaseFile = r'sherlockholmes2.txt'
    AsymRaw = open(author1_file).read()
    #print(len(AsymRaw))
    AsymX = clean_symbols_fn(AsymRaw, SymCleanMode)


    # Get Shannon entropy, set of unique symbols, histogram probabilities of supplied symbols
    Hea, se, pe = pluginprob2(AsymX)

    Na = len(AsymX)

    '''
    This is the python code for 
        U1 = 'A';          %   Utterance symbol 1  
        U2 = 'B';          %   Utterance symbol 2  
        U3 = 'C';          %   Utterance symbol 3   
        U4 = 'D';          %   Utterance symbol 4       
        U5 = 'E';          %   Utterance symbol 5        

        
        % Symbolic conversion of utterance length
        %
        % note: this method assumes the maximum length of utterance to be known
        astr = strrep(astr,'************',U5);   % 12   
        astr = strrep(astr,'***********',U5);    % 11
        astr = strrep(astr,'**********',U5);     % 10    
        astr = strrep(astr,'*********',U5);      %  9
        
        astr = strrep(astr,'********',U4);       %  8 
        astr = strrep(astr,'*******',U4);        %  7     
        astr = strrep(astr,'******',U4);         %  6
        
        astr = strrep(astr,'*****',U3);          %  5
        astr = strrep(astr,'****',U3);           %  4  
        
        astr = strrep(astr,'***',U2);            %  3
        
        astr = strrep(astr,'**',U1);             %  2      
        astr = strrep(astr,'*',U1);  
    '''
    conditions = [2, 3, 5, 8]
    letters = ['A', 'B', 'C', 'D', 'E']
    Asym = ''
    paragraphs_A = AsymX.split('\n')
    for pa in paragraphs_A:
        if pa.strip():
            p_Asym, _, _ = symbolize_by_wordlengths(word_tokenize(pa), conditions, letters)
            p_Asym = 'Z'.join([c for c in p_Asym])
            Asym += p_Asym + 'V'

    #BaseFile = r'drseusscat.txt'
    BsymRaw = open(author2_file).read()
    BsymX = clean_symbols_fn(BsymRaw, SymCleanMode)
    Nb = len(BsymX)

    Bsym = ''
    paragraphs_B = BsymX.split('\n')
    for pb in paragraphs_B:
        if pb.strip():
            p_Bsym, int_sybl, symobl_clean = symbolize_by_wordlengths(word_tokenize(pb), conditions, letters)
            p_Bsym = 'Z'.join([c for c in p_Bsym])
            Bsym += p_Bsym + 'V'


    for i in range(1, 20):

        Bstart = (i-1) * 500 + 1 + 20
        Bend = i * 500 + 20
        Bblock = Bsym[Bstart:Bend] 

        He, se, pe = pluginprob2(Bblock)


#===================================================================   
#    2
#    Synthesize aggregate symbolic file with injected authorship
#
#    Naw, Nbw = symbolic step length for each file
#===================================================================   

    Csym = ''

    Np = 3
    Naw = 60000  # start point for symbolic data set 1
    Nbw = 2000   # start point for symbolic data set 2
    MarkerArr = []
    SOffset = 50

    for i in range(1, Np):
        Astart = (i-1) * Naw + 1 + SOffset
        Aend = i * Naw + SOffset
        Bstart = (i-1) * Nbw + 1 + SOffset
        Bend = i * Nbw + SOffset
        
        Ablock = Asym[Astart:Aend]  # MATLAB indexing is 1-based
        Bblock = Bsym[Bstart:Bend]  # MATLAB indexing is 1-based
        
        Csym += Ablock + Bblock
        
        nax = len(Ablock)
        nbx = len(Bblock)
        
        Xa = [0] * nax
        Xb = [1] * nbx
        
        MarkerArr.extend(Xa + Xb)

    # Save the combined file:
    BaseFile = r''  # Double backslashes for Windows paths

    if DoSaveFile == 1:
        TheFile = f"{BaseFile}_{RunNo}.txt"
        print(f"\n\n Save the combined file: {TheFile} \n")
        
        with open(TheFile, 'wt') as fid:
            fid.write(Csym)

    
    # Synthesize aggregate SYMBOLIC file
    #

    Csym = ''

    Ndivb = 4  # number of segments in Nbw
    Nsegb = Nbw / Ndivb  # e.g., 2000/4 = 500 = now the size we use for all Heb calcs
    Ndiva = 35  # number of segments in Nba
    Nsega = Naw / Ndiva  # e.g., 40,000/20 = 2000 = now the size we use for all Hea calcs

    # Determine the rank order of symbols in Asym
    #
    Ha, sa, pa = pluginprob2(Asym)
    # Determine the rank order of symbols in Bsym
    #
    Hb, sb, pb = pluginprob2(Bsym)

    for i in range(0, Np ):
        Astart = (i) * Naw + 1 + SOffset
        Aend = (i+1) * Naw + SOffset
        Bstart = (i) * Nbw + 1 + SOffset
        Bend = (i+1) * Nbw + SOffset

        Ablock = Asym[Astart:Aend]  
        Bblock = Bsym[Bstart:Bend]  

        Csym += Ablock + Bblock  # combine blocks of symbols in sequence

        nax = len(Ablock)
        nbx = len(Bblock)
        Xa = [0] * nax
        Xb = [1] * nbx
        MarkerArr.extend(Xa + Xb)

    # Save the combined file
    if DoSaveFile == 1:
        BaseFile = ''
        TheFile = f'{BaseFile}_gx{RunNo}.txt'
        print(f'\n\n Save the combined file: {TheFile}\n')
        with open(TheFile, 'wt') as fid:
            fid.write(Csym)

#----------------------------------------------------------------------
#
# <<< Compute Entropy on the SYMBOLIC text >>>
#
    print('\n\n Compute Entropy on the SYMBOLIC text \n')

# 1-gram analysis where we use an intermediate step in which we
# symbolize utterance length
#
# DO A RUN OVER Nf BLOCKS OF TEXT TO DETERMINE He
#
    Nc = len(Csym)
    Ncw = 500

    Nf = Nc // Ncw

#========================================================
#   3
# Perform fast entropy on the symbolic data 
#
#========================================================
#  

    # Determine the rank order of symbols to select Csym
    Hc, sc, pc = pluginprob2(Csym)

    # Set alphabet (M) value
    M = 27

    # Now compute rank r distances for a specific symbol
    CSymbol = 'C'
    RunType = 'C'  # A, B, or C (A+B)

    Xstart = 0
    Hxa = np.zeros(Nf)
    Hea = np.zeros(Nf)
    HTarget = np.zeros(Nf)

    ap = 0.0073
    bp = 4.2432
    cp = 4.2013

    #FastEntropy4_czml(symbols, Naw, selected_symbol, M, ap, bp, cp, zml_model='zml', eta=0.2):
    for k in range(0, Nf):
        # Take symbolic segment
        Xend = Xstart + Ncw - 1
        Xblock = Csym[Xstart:Xend]  # MATLAB indexing is 1-based
        print(k)
        Hx, pa = FastEntropy4_czml(Xblock, Ncw, CSymbol, M, ap, bp, cp, zml_model, eta)  # VERIFIED 21-04-2021
        He, se, pe = pluginprob2(Xblock)

        # Translate Csym MarkerArr to H Target array
        XblockScore = MarkerArr[Xstart:Xend]  # MATLAB indexing is 1-based
        xsum = sum(XblockScore)
        Nxb = len(XblockScore)
        xscore = xsum / Nxb
        if xscore < 0.5:
            HTarget[k - 1] = 0
        else:
            HTarget[k - 1] = 1

        print(f'Ha({k}) = {Hx:.4f} He({k}) = {He:.4f}')
        Hxa[k - 1] = Hx
        Hea[k - 1] = He

        Xstart = Xend + 1

# END HXA ------------------------------------------------------------

# 2nd Pass

# START HXA2 -----------------------------------------------------------

# Synthesize aggregate PLAIN TEXT file
    Csym = ''
    Np = 3
    Naw = 60000
    Nbw = 2000

    MarkerArr = []

    for i in range(0, Np):
        # Take symbolic segment
        Astart = (i) * Naw
        Aend = (i+1)* Naw
        Bstart = (i) * Nbw
        Bend = (i+1) * Nbw

        Ablock = Asym[Astart:Aend]
        Bblock = Bsym[Bstart:Bend]

        Csym += Ablock + Bblock

        nax = len(Ablock)
        nbx = len(Bblock)
        Xa = [0] * nax
        Xb = [1] * nbx
        MarkerArr.extend(Xa + Xb)

    # Save the combined file:
    if DoSaveFile == 1:
        BaseFile = ''
        TheFile = f'{BaseFile}_{RunNo}.txt'
        print(f'\n\n Save the combined file: {TheFile} \n')
        
        with open(TheFile, 'wt') as fid:
            fid.write(Csym)


#===================================================================   
#    4
#    Asym = Sherlock Holmes
#    Bsym = Dr Seuss
#
#===================================================================   
#    
      
#===================================================================   
#    Csym = Aggregate SYMBOLIC 
#
#    Asym = Sherlock Holmes
#    Bsym = Dr Seuss
#
#===================================================================   
#  


    # Synthesize aggregate SYMBOLIC file
    Csym = ''

    Ndivb = 4  # number of segments in Nbw
    Nsegb = Nbw / Ndivb  # eg 2000/4 = 500 = now the size we use for all Heb calcs
    Ndiva = 35  # number of segments in Nba
    Nsega = Naw / Ndiva  # eg 40,000/20 = 2000 = now the size we use for all Hea calcs

    # Determine the rank order of symbols in Asym
    Ha, sa, pa = pluginprob2(Asym)

    # Determine the rank order of symbols in Bsym
    Hb, sb, pb = pluginprob2(Bsym)

    for i in range(0, Np):
        Astart = ((i) * Naw) 
        Aend = (i+1) * Naw
        Bstart = ((i) * Nbw) 
        Bend = (i+1) * Nbw
        Ablock = Asym[Astart:Aend]  # get next A block of symbols (ie larger)
        Bblock = Bsym[Bstart:Bend]  # get next B block of symbols (ie smaller)
        Csym += Ablock + Bblock  # combine blocks of symbols in sequence

    # Save the combined file:
    if DoSaveFile == 1:
        BaseFile = 'C:/data/corpora/Combined2'
        TheFile = f'{BaseFile}_gx{RunNo}.txt'
        print(f'\n\n Save the combined file: {TheFile} \n')
        with open(TheFile, 'wt') as fid:
            fid.write(Csym)


#----------------------------------------------------------------------
#
# <<< Compute Entropy on the SYMBOLIC text >>>
#
    print('\n\n Compute Entropy on the SYMBOLIC text \n');  

    # 1-gram analysis where we use an intermediate step in which we symbolize utterance length
    MeanHea = 0
    MeanHeb = 0
    Hp = []
    Xp = []
    Hpa = [0] * Nf
    Hpb = [0] * Nf
    Xh = [0] * Nf
    ia = 0  # actual i count index (note that there may be invalid blocks where no symbols are detected)
    ib = 0
    nc = 0
    Astart = 1
    xcount = 2
    Hea = 0
    Heb = 0

    # Perform fast entropy on the symbolic data
    Hc, sc, pc = pluginprob2(Csym)

    # Set alphabet (M) value
    M = 27

    # Now compute rank r distances for specific symbol
    CSymbol = 'C'
    RunType = 'C'  # A, B or C (A+B)

    Xstart = 0
    Hxa2 = [0] * Nf
    Hea2 = [0] * Nf
    HTarget = [0] * Nf


    ap = 0.0073
    bp = 4.2432
    cp = 4.2013

    for k in range(0, Nf):  # do for all blocks of size Ncw

        # Take symbolic segment
        Xend = Xstart + Ncw - 1
        Xblock = Csym[Xstart:Xend]
        leng = len(Xblock)

        Hx, pa = FastEntropy4_czml(Xblock, Ncw, CSymbol, M, ap, bp, cp, zml_model, eta)
        He, se, pe = pluginprob2(Xblock)

        # Translate Csym MarkerArr to H Target array
        XblockScore = MarkerArr[Xstart:Xend]
        xsum = sum(XblockScore)
        Nxb = len(XblockScore)
        xscore = xsum / Nxb
        if xscore < 0.5:
            HTarget[k-1] = 0
        else:
            HTarget[k-1] = 1

        Hxa2[k-1] = Hx
        Hea2[k-1] = He


        Xstart = Xend + 1

    # END HXA2 -----------------------------------------------------------

    Hxa3 = [sum(x) for x in zip(Hxa, Hxa2)]
    Hxa = [x / 2 for x in Hxa3]

    Hea3 = Hea2 # since Hea = 0
    Hea = [x / 2 for x in Hea3]

# ===================================================================
# Calculate normalized statistical bounds of entropy for plotting
# ===================================================================

# -------------------------------------------------------------------
# Mean normalization
# Normalize entropy arrays to each other in order to compare:
# Use first 10% of block in this case, but usually may use a longer data set.
    Nx = len(Hxa)
    Nm = int(0.1 * Nx)

    MeanHx = np.mean(Hxa[:Nm])
    MeanHe = np.mean(Hea[:Nm])

    Hem = Hea - MeanHe + MeanHx
    MeanHem = np.mean(Hem[:Nm])
    # -------------------------------------------------------------------

    # -------------------------------------------------------------------
    # Variance normalization
    #
    # Ensure zero mean at first
    #
    Hxz = Hxa - MeanHx
    Hez = Hem - MeanHem

    StdHx = np.std(Hxz[:Nm])
    StdHe = np.std(Hez[:Nm])

    Hev = (Hez / StdHe)
    StdHev = np.std(Hev[:Nm])

    Hev = ((Hez / StdHe) * StdHx) + MeanHem
    # StdHev = np.mean(Hev[:Nm])

    XLimit = 430
    YLimit = 4.30

    # Find mean line
    MLine = np.full(XLimit, MeanHx)

    # Find 3-std dev line
    HxStdDev3 = MeanHx + 3 * StdHx
    SLine3p = np.full(XLimit, HxStdDev3)

    HxStdDev3 = MeanHx - 3 * StdHx
    SLine3m = np.full(XLimit, HxStdDev3)

    # Find 5-std dev line
    HxStdDev5 = MeanHx + 5 * StdHx
    SLine5p = np.full(XLimit, HxStdDev5)

    HxStdDev5 = MeanHx - 5 * StdHx
    SLine5m = np.full(XLimit, HxStdDev5)


#===================================================================   
#    
#  Plot Fast Entropy results
#
#===================================================================   
    


    # Plot for the first figure
    fig, ax = plt.subplots()
    colr = 'MediumBlue'
    ax.plot(Hxa, '-o', color=colr, linewidth=PlotLineWidth, markerfacecolor=[0.7, 0.7, 0.8], markersize=1)
    ax.plot(MLine, color='DarkKhaki', linewidth=PlotLineWidth, linestyle='--', markerfacecolor=[0.7, 0.7, 0.8], markersize=1)
    ax.plot(SLine3m, color='Green', linewidth=PlotLineWidth, linestyle=':', markerfacecolor=[0.7, 0.7, 0.8], markersize=1)
    ax.plot(SLine5m, color='Green', linewidth=PlotLineWidth, linestyle=':', markerfacecolor=[0.7, 0.7, 0.8], markersize=1)
    ax.plot(SLine3p, color='Green', linewidth=PlotLineWidth, linestyle=':', markerfacecolor=[0.7, 0.7, 0.8], markersize=1)
    ax.plot(SLine5p, color='Green', linewidth=PlotLineWidth, linestyle=':', markerfacecolor=[0.7, 0.7, 0.8], markersize=1)
    ax.plot(HxStdDev5 * np.array(HTarget), color='BurlyWood', linewidth=PlotLineWidth, linestyle='-.', markerfacecolor=[0.7, 0.7, 0.8], markersize=2)

    if zml_model == 'zml':
        # Define rectangle parameters
        rect_params = {'color': 'red', 'facecolor': 'red'}

        # Create rectangles
        rect1 = patches.Rectangle(( 119, 4.247), 5, 0.0017, **rect_params)
        rect2 = patches.Rectangle((243, 4.23), 5, 0.0017, **rect_params)
        rect3 = patches.Rectangle((364, 4.198), 5, 0.0017, **rect_params)

        # Add rectangles to the plot
        ax.add_patch(rect1)
        ax.add_patch(rect2)
        ax.add_patch(rect3)

        # Define arrow properties
        arrow_params = {'arrowstyle': '->', 'color': 'DarkSlateGray'}

        # Create textarrows
        arrow1 = patches.FancyArrowPatch((100, 4.22), (119, 4.247),  connectionstyle="arc3,rad=0", **arrow_params)
        arrow2 = patches.FancyArrowPatch((100, 4.22), (243, 4.23),  connectionstyle="arc3,rad=0", **arrow_params)
        arrow3 = patches.FancyArrowPatch((100, 4.22), (364, 4.198),  connectionstyle="arc3,rad=0", **arrow_params)
        arrow4 = patches.FancyArrowPatch((100, 4.29), (33, 4.273),  connectionstyle="arc3,rad=0", **arrow_params)

        # Add textarrows to the plot
        ax.add_patch(arrow1)
        ax.add_patch(arrow2)
        ax.add_patch(arrow3)
        ax.add_patch(arrow4)

        # Create textarrow for 'Dr. Seuss'
        ax.text(30, 4.22, 'Dr. Seuss', color='DarkSlateGray', fontsize=12)

        # Create textarrow for 'Sherlock Holmes'
        ax.text(100, 4.29, 'Sherlock Holmes', color='DarkSlateGray', fontsize=12)

        # # # Set axis limits

    
    ax.set_xlim([1, XLimit])
    ax.set_ylim([4.19, YLimit])

    title_str = r'$N_{{a}} =$' + str(Naw) + ' ' + r'$N_{{b}} =$' +  str(Nbw) + ' ' + r'$N_{{c}} =$' + str(Ncw)
    plt.title(title_str, fontsize=16)

    ax.set_xlabel(r'$n$', fontsize=PlotCaptionFontSize, horizontalalignment='center')
    ax.set_ylabel(r'$\hat{H}_{f}(n)$', fontsize=PlotCaptionFontSize, horizontalalignment='center')

    plt.show()

    # -------------------------------------------------------
    # if DoSaveFile == 1:
    #     FigFileNameEPS = f'output/sherlockseussentropy{RunNo}_{Naw}_{Nbw}_{Ncw}.eps'
    #     FigFileNameJPG = f'output/sherlockseussentropy{RunNo}_{Naw}_{Nbw}_{Ncw}.jpg'
    #     FigFileNamePNG = f'output/sherlockseussentropy{RunNo}_{Naw}_{Nbw}_{Ncw}.png'
    #     print('\n\n Save the image files: %s \n' % FigFileNameJPG)
    #     fig.savefig(FigFileNameEPS, format='eps')
    #     fig.savefig(FigFileNameJPG, format='jpg')
    #     fig.savefig(FigFileNamePNG, format='png')

    # # Plot for the second figure
    fig, ax = plt.subplots()
    colr = 'Brown'
    #print(Hev)
    ax.plot(Hev, color=colr, linewidth=PlotLineWidth, linestyle='-', markerfacecolor=[0.7, 0.7, 0.8], markersize=1)
    ax.plot(MLine, color='DarkKhaki', linewidth=PlotLineWidth, linestyle='--', markerfacecolor=[0.7, 0.7, 0.8], markersize=1)
    ax.plot(SLine3m, color='Green', linewidth=PlotLineWidth, linestyle=':', markerfacecolor=[0.7, 0.7, 0.8], markersize=1)
    ax.plot(SLine5m, color='Green', linewidth=PlotLineWidth, linestyle=':', markerfacecolor=[0.7, 0.7, 0.8], markersize=1)
    ax.plot(SLine3p, color='Green', linewidth=PlotLineWidth, linestyle=':', markerfacecolor=[0.7, 0.7, 0.8], markersize=1)
    ax.plot(SLine5p, color='Green', linewidth=PlotLineWidth, linestyle=':', markerfacecolor=[0.7, 0.7, 0.8], markersize=1)
    ax.plot(HxStdDev5 * np.array(HTarget), color='BurlyWood', linewidth=PlotLineWidth, linestyle='-.', markerfacecolor=[0.7, 0.7, 0.8], markersize=2)

    ax.set_xlim([1, XLimit])
    ax.set_ylim([4.2, 4.30])

    ax.set_xlabel(r'$n$', fontsize=PlotCaptionFontSize, horizontalalignment='center')
    ax.set_ylabel(r'$\hat{H}_{0}(n)$', fontsize=PlotCaptionFontSize, horizontalalignment='center')
    if zml_model == 'zml':
        arrow_params = {'arrowstyle': '->', 'color': 'DarkSlateGray'}
        arrow4 = patches.FancyArrowPatch((100, 4.29), (33, 4.273),  connectionstyle="arc3,rad=0", **arrow_params)
        ax.add_patch(arrow4)
        ax.text(100, 4.29, 'Sherlock Holmes', color='DarkSlateGray', fontsize=12)

    plt.show()


        # if DoSaveFile == 1:
        #     FigFileNameEPS = f'output/sherlockseussentropy2{RunNo}_{Naw}_{Nbw}_{Ncw}.eps'
        #     FigFileNameJPG = f'output/sherlockseussentropy2{RunNo}_{Naw}_{Nbw}_{Ncw}.jpg'
        #     FigFileNamePNG = f'output/sherlockseussentropy2{RunNo}_{Naw}_{Nbw}_{Ncw}.png'
        #     print('\n\n Save the image files: %s \n' % FigFileNameJPG)
        #     fig.savefig(FigFileNameEPS, format='eps')
        #     fig.savefig(FigFileNameJPG, format='jpg')
        #     fig.savefig(FigFileNamePNG, format='png')

    print('\nDone\n')



if __name__ == "__main__":
    check_authorship_detection(zml_model='zml')
    # check_authorship_detection(zml_model='czml1')
    # check_authorship_detection(zml_model='czml2', eta=0.2)
    # Input text with multiple '\n' characters

