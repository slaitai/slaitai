import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

#import slaitai_entropy as entropy
import ZML_cZML_entropy as entropy



def BuildEntropyModel(Nq,Nw,Ns,Kstart,Kstep,Kend,Rmax,Mmain,
                        Rmodel,DoPlot,DoFileSave,DoVerbose, zml_model='zml',  eta=0.2):
    # set zml_model
    # zml_model='zml': using normal zml, 
    # zml_model='czml1': using cZML model 1,
    # zml_model='czml2': using cZML model 2



    # [ Test ]
    # Comment out for normal use:
    # Note: we save the data here so that it can be read into Python code for
    # development and debugging, hence comparing precisely with Matlab results
    #
    # BaseDir = 'C:\data\uniform\'
    # TheFileName = 'udata1.txt'
    # [xrand] = GenerateRandomUniform(Nq, BaseDir, TheFileName)
    # FlagRand = 1
    np.random.seed(123)
    FlagRand = 0

    if DoVerbose == 1:
        print('\n Build Rank-Based Entropy Model (Biased Distribution) ... \n')

    Krange = int((Kend - Kstart) / Kstep + 1)
    dmeanmat = np.zeros((Krange, Kend))

    Mrange = np.arange(Kstart, Kend + 1, Kstep)
    print(Mrange)

    # DO FORWARD MODEL
    ix = 0
    KnCount = 1

    # M = K values in Mrange = Kn below

    for Kn in Mrange:
        if DoVerbose == 1:
            print(f'Kn={Kn} ==============================================\n')
  
        Mrange[ix] = Kn
        ix = ix + 1

        # Allocate vectors
        d2 = np.zeros(Kn)
        d_all = np.zeros(Kn)
        dsum_all = np.zeros(Kn)

        # Compute set of ranked probabilities according to Zipf-Mandelbrot-Li model for this K value
        if zml_model == 'zml':
            Ha, pc = entropy.CalcZMLEntropy(Kn)
        elif zml_model == 'czml1':
            Ha, pc = entropy.Calc_constrained_ZMLEntropy_model(Kn, model=1)
        else:
            Ha, pc = entropy.Calc_constrained_ZMLEntropy_model(Kn, model=2, eta=eta)

        for h in range(Ns):
            # Generate data according to probability model

            if FlagRand == 0:
                x = np.random.rand(Nq)
            else:
                # enable the commented code to calculate xrand
                #x = xrand
                pass

            dist = 0
            distv = np.ones(Kn)
            lastdistv = np.ones(Kn)
            reldistv = np.zeros(Kn)

            distm = np.zeros((Nw, Kn))
            symbolcount = np.zeros(Kn)
            firstsymbolflag = np.ones(Kn)

            for i in range(Nw):
                dist = dist + 1

                vala = x[i]

                j = 0
                valb = pc[j]

                if vala <= valb:
                    val = d2[j]
                    d2[j] = val + 1

                    if firstsymbolflag[j] == 1:
                        firstsymbolflag[j] = 0
                        distv[j] = dist
                        lastdistv[j] = dist
                    else:
                        lastdistv[j] = distv[j]
                        distv[j] = dist
                        reldistv[j] = distv[j] - lastdistv[j] + 1

                for j in range(1, Kn):
                    xval = x[i]
                    pval1 = pc[j - 1]
                    pval2 = pc[j]

                    if xval > pval1 and xval <= pval2:
                        d2[j] = d2[j] + 1

                        if firstsymbolflag[j] == 1:
                            firstsymbolflag[j] = 0
                            distv[j] = dist
                            lastdistv[j] = dist
                        else:
                            lastdistv[j] = distv[j]
                            distv[j] = dist
                            dv1 = distv[j]
                            dv2 = lastdistv[j]
                            reldistv[j] = dv1 - dv2 + 1

                        break
                for k in range(Kn):
                    if (d2[k] > 1):
                        symbolcount[k] = symbolcount[k] + 1
                        smc = int(symbolcount[k])
                        distm[smc-1, k] = reldistv[k]
                        val = d2[k]
                        d2[k] = val - 1

            SumDM = distm.sum(axis=0) 

            for jj in range(Kn):
                if symbolcount[jj] != 0:
                    d_all[jj] = SumDM[jj] / symbolcount[jj]
                else:
                    d_all[jj] = 0

            dsum_all = dsum_all + d_all

        dmean_all = dsum_all / Ns

        for j in range(Kn):
            val = dmean_all[j]
            dmeanmat[KnCount - 1, j] = val

        KnCount = KnCount + 1

    if DoVerbose == 1:
        print('Curve fitting to create forward model (***)...\n')



    if DoPlot == 1:
        #plt.figure()
        plt.figure(figsize=(15, 8))

    for R in range(1, Rmax + 1):
        if DoVerbose == 1:
            print(f'R={R}\n')

        dmeanvec = dmeanmat[:, R - 1]

        # Fit this model using new data
        fitopt = curve_fit(lambda x, a, b, c: a * x**b + c, Mrange, dmeanvec, maxfev=1000)

        # Hence, determine dmeanvec, from the supplied M value
        # dmeanvec(M) = = a*x^b+c

        # Forward model parameters for this particular R value.
        ap, bp, cp = fitopt[0]
        print(f'Build Stage (R={R}): (ap,bp,cp): {ap:.4f}, {bp:.4f}, {cp:.4f}')

        # create actual curve output - Forward Model
        FitDmean = ap * Mrange**bp + cp

        # FORWARD MODEL CURVES
        if DoPlot == 1:
            plt.plot(Mrange, dmeanvec, color='g', marker = '', linewidth=2, label=f'R={R}')  # plot raw data as green lines
            plt.plot(Mrange, FitDmean, color='r',  marker = '', linewidth=3, label=f'Fitted R={R}')  # plot fitted curve as red line


    if DoPlot == 1:
        plt.xlabel('M', fontsize=20)
        plt.ylabel('D(M)', fontsize=20)
        plt.xticks([0, 20, 40, 60, 80, 100])
        plt.legend(fontsize=16)
        plt.xlim(0, 100)
        plt.title(f'R=1..{Rmax}', fontsize=18)
        plt.grid(True)
        plt.show()

    # DO INVERSE MODEL
    print('Do inverse model..\n')

    if DoPlot == 1:
        plt.figure(figsize=(15, 8))

    for R in range(1, Rmax + 1):
        if DoVerbose == 1:
            print(f'R={R}\n')
        
        dmeanvec = dmeanmat[:, R - 1]

        fitopt = curve_fit(lambda x, a, b, c: a * x**b + c, dmeanvec, Mrange, maxfev=1000)
        ap, bp, cp = fitopt[0]
        print(f'Build Stage (R={R}): (ap,bp,cp): {ap:.4f}, {bp:.4f}, {cp:.4f}')

        # create actual curve output - Inverse Model
        FitK = ap * dmeanmat[:, R - 1]**bp + cp

        # INVERSE MODEL
        if DoPlot == 1:
            plt.plot(dmeanmat[:, R - 1], Mrange, color='g',  marker = '', linewidth=2, label=f'R={R}')  # plot raw data as green lines
            plt.plot(dmeanmat[:, R - 1], FitK, color='r', marker = '', linewidth=3, label=f'Fitted R={R}')  # plot fitted curve as red line

    if DoPlot == 1:
        plt.xlabel('D', fontsize=20)
        plt.ylabel('M', fontsize=20)
        plt.ylim(0, 120)
        plt.title(f'R=1..{Rmax}', fontsize=18)
        plt.legend(fontsize=16)
        plt.grid(True)
        plt.show()


    if DoVerbose == 1:
        print(f'Do Rank R={Rmodel} Model..\n')

    # R = 1  # USED FOR TEST PHASE
    # this was not inverted previously... it should be K = y-axis and D=x-axis

    # Fit this model using new data
    fitopt = curve_fit(lambda x, a, b, c: a * x**b + c, dmeanmat[:, Rmodel - 1], Mrange, maxfev=1000)

    # Hence, determine dmeanvec, from the supplied M value
    # dmeanvec(M) = = a*x^b+c

    # Final ZML Model Parameters to return
    ap, bp, cp = fitopt[0]
    print(f'Build Stage (R={Rmodel}): (ap,bp,cp): {ap:.4f}, {bp:.4f}, {cp:.4f}')

    # Save the current workspace
    if DoFileSave == 1:
        TheEntropyModel = f'EntropyModel_R{Rmodel}_M{Mmain}_'
        # FileName = f'{TheEntropyModel}{datestr(now, 'yyyy-mmm-dd-HH-MM')}.mat'
        # np.savez(FileName, ap=ap, bp=bp, cp=cp)

    return ap, bp, cp

if __name__ == "__main__":

    DoPlot = 1
    DoFileSave = 1
    DoVerbose = 1

    M = 4 # Alphabet size

    Nq = 50000  # No of samples
    Nw = 5000   # 20 # Window length to obtain entropy
    Ns = 50    #2 #  No of random trials to get statistical average
    Kstart = 8
    Kend = 60
    Kstep = 1
    Rmax = 5

    R = 1 # Estimate model parameters for rank R = 1
    [ap, bp, cp] = BuildEntropyModel(Nq, Nw, Ns, Kstart, Kstep, Kend, Rmax, M, R, DoPlot, DoFileSave, DoVerbose);
    # Quite slow!!

    print("------------------------------- Build Entropy Model Output Test------------------------------")
    print('Entropy Model (a,b,c):'+str(ap)+str(bp)+str(cp))