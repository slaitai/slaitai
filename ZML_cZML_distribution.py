
import numpy as np
import matplotlib.pyplot as plt

############################################################
#   GenerateZMLDistribution 
#           and
#   CheckcZMLDistribution

# the code is copied from slaitai_entropy.py
# the only changes are
#   1. the M var becomes a paramater in CheckcZMLDistribution
#   2. return p directly
############################################################

def GenerateZMLDistribution(M):
    p = np.zeros(M)

    M1=M+1
    a=np.log2(M1)/np.log2(M)
    beta=M/M1
    gamma=(M**(a-1))/(M-1)**a

    # First pass
    psum=0
    for r in range(1,M1):
        Pr=gamma/((r+beta)**a)
        p[r-1]=Pr
        psum=psum+Pr
        if r<=5:
            pass

    knorm=psum
    gammaprime = gamma / knorm

    # Now normalize against the sum
    # Second pass
    for r in range(1,M1):
        Pr=gammaprime/((r+beta)**a)
        p[r-1]=Pr

    return p



############################################################
#   Implementation of Constrained ZML algorithm  
#   in Generate_cZML_Distribution
#           with
#   Check_cZML_Distribution
############################################################

def Generate_cZML_Distribution_1(M):
    p = np.zeros(M)

    # euqation A34 in the paper

    a_tilde=np.log2(M)/np.log2(M-1)

    # euqation A35 in the paper
    beta_tilde=M/(M-2)
    # euqation A36 in the paper
    gamma_tilde=(M**(a_tilde-1))/(M-2)**a_tilde

    # First pass
    psum=0
    # r is the rank, always starts by 1
    # however, vector p starts by 0, that's why p[r-1]
    # this is the same as ZML implementation
    for r in range(1,M+1):
        # my understanding of Pr in cZML is that
        # it computes in the same way as ZML as p=gamma/(rank + beta)**alpha
        # but alpha, beta, and gamma are computed differently
        # hence in cZML they are named alpha_tilde, beta_tilde and gamma_tilde
        Pr=gamma_tilde/((r+beta_tilde)**a_tilde)
        p[r-1]=Pr
        psum=psum+Pr

    knorm=psum
    gammaprime = gamma_tilde / knorm

    # Now normalize against the sum
    # Second pass
    for r in range(1, M+1):
        Pr=gammaprime/((r+beta_tilde)**a_tilde)
        p[r-1]=Pr

    return p

def Generate_cZML_Distribution_2(M, eta):
    p = np.zeros(M)

    # euqation A79 in the paper
    a_tilde=np.log2(M)/np.log2(eta*(M-1))
    # euqation A80 in the paper
    beta_tilde=M/(eta*(M-1)-1)
    # euqation A81 in the paper
    gamma_tilde=(M**(a_tilde-1))/(eta*(M-1)-1)**a_tilde

    # First pass
    psum=0
    # r is the rank, always starts by 1
    # however, vector p starts by 0, that's why p[r-1]
    # this is the same as ZML implementation
    for r in range(1,M+1):
        # my understanding of Pr in cZML is that
        # it computes in the same way as ZML as p=gamma/(rank + beta)**alpha
        # but alpha, beta, and gamma are computed differently
        # hence in cZML they are named alpha_tilde, beta_tilde and gamma_tilde
        Pr=gamma_tilde/((r+beta_tilde)**a_tilde)
        p[r-1]=Pr
        psum=psum+Pr

    knorm=psum
    gammaprime = gamma_tilde / knorm

    # Now normalize against the sum
    # Second pass
    for r in range(1, M+1):
        Pr=gammaprime/((r+beta_tilde)**a_tilde)
        p[r-1]=Pr

    return p

# def Check_cZML_Distribution_1(M=10):
#     # Check ZML Distribution Generator:

#     [pz] = Generate_cZML_Distribution_1(M)
#     print('\nCZML distribution:')
#     for r in range(M):
#         print('{:5.4f}, '.format(pz[r]), sep=' ', end='', flush=True)
#     print('\n')


# def Check_cZML_Distribution_2(M=10, eta=0.2):
#     # Check ZML Distribution Generator:

#     [pz] = Generate_cZML_Distribution_2(M, eta)
#     print('\nCZML model 2 distribution:')
#     for r in range(M):
#         print('{:5.4f}, '.format(pz[r]), sep=' ', end='', flush=True)
#     print('\n')

############################################################
#   Generate parameters alpha_tilde in cZML
#   and alpha in ZML
#
#   This is for reproducing Figure 3
############################################################

def GetZMLParams(M, model='zml', eta=.2):
    if model=='zml':
        alpha=np.log2(M+1)/np.log2(M)
        beta=M/(M+1)
        gamma=(M**(alpha-1))/(M-1)**alpha
    elif model == 'czml1':
        alpha=np.log2(M)/np.log2(M-1)
        beta=M/(M-2)
        gamma=(M**(alpha-1))/(M-2)**alpha
    else:
        alpha=np.log2(M)/np.log2(eta*(M-1))
        beta=M/(eta*(M-1)-1)
        gamma=(M**(alpha-1))/(eta*(M-1)-1)**alpha
    return alpha, beta, gamma

def GenerateZML_Alpha(M):
    p = np.zeros(M)

    for m in range(3,M+1):
        a=np.log2(m+1)/np.log2(m)
        p[m-1]=a

    return p


def Generate_cZML_Alpha(M):
    p = np.zeros(M)

    for m in range(3,M+1):
        a_tilde=np.log2(m)/np.log2(m-1)
        p[m-1]=a_tilde

    return p

############################################################
#   plot Figure 3 & 4
############################################################

def plot_fig3(data):
    x_values = np.arange(3, 13)
    y_values = data[2:]  # Using elements from the third element onward, removing nan

    plt.plot(x_values, y_values, marker='o', label='a_tilde/a')

    plt.ylim(0.8, 1.4)
    plt.yticks(np.arange(0.8, 1.5, 0.1))

    plt.title('cZML/ZML')
    plt.xlabel('Number')
    plt.ylabel('a_tilde/a')
    plt.legend()

    plt.show()

def plot_fig4(zml_distr_4m, c_zml_distr_4m):
    x_values = np.arange(1, 5)
    print(zml_distr_4m)
    print(c_zml_distr_4m)
    plt.plot(x_values, zml_distr_4m, marker='o', label='ZML')
    plt.plot(x_values, c_zml_distr_4m, marker='s', label='cZML')

    plt.ylim(0.1, 0.45)
    plt.yticks(np.arange(0.1, 0.5, 0.05))

    plt.xticks(x_values)
    plt.xlabel('Number')
    plt.ylabel('Values')
    plt.legend()

    plt.show()


def plot_fig5(zml_distr_10m, c_zml_distr_2_10m):
    x_values = np.arange(1, 11)
    print(zml_distr_10m)
    print(c_zml_distr_2_10m)
    plt.plot(x_values, zml_distr_10m, marker='o', label='ZML')
    plt.plot(x_values, c_zml_distr_2_10m, marker='s', label='cZML model 2')

    plt.ylim(0, 0.3)
    plt.yticks(np.arange(0, 0.3, 0.05))

    plt.xticks(x_values)
    plt.xlabel('Number')
    plt.ylabel('Values')
    plt.legend()

    plt.show()

if __name__ == "__main__":

    a_tilde_czml = Generate_cZML_Alpha(12)
    a_zml = GenerateZML_Alpha(12)
    print(np.array(a_tilde_czml)/np.array(a_zml))
    plot_fig3(np.array(a_tilde_czml)/np.array(a_zml))


    zml_distr_4m = GenerateZMLDistribution(4)
    c_zml_distr_4m = Generate_cZML_Distribution_1(4)
    plot_fig4(np.array(zml_distr_4m), np.array(c_zml_distr_4m))


    zml_distr_10m = GenerateZMLDistribution(10)
    c_zml_distr_2_10m = Generate_cZML_Distribution_2(10, 0.2)
    plot_fig5(np.array(zml_distr_10m), np.array(c_zml_distr_2_10m))



