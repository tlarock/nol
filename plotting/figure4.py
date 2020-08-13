import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


start_probe = 0
num_probes = 50000

names = {
    'twitter/': ('twitter', 90000)
}

font_size = 30
tick_size = 32
label_size = 34
legend_font = 13
sample_dir = 'node-0.01/'
out_reward_name = 'new-nodes'
plots_base = '../results/plots/cumulative_reward/'
results_base = '../results/'
legend=True

for name in names:
    out_name = names[name][0]
    print(out_name)

    N = names[name][1]

    input_dir = results_base + name + sample_dir
    print(input_dir)

    nonpara_results = '/Users/larock/git/nol/baseline/net_complete/mab_explorer/results/' + names[name][0]+ '_rn_results'
    input_files = [
            (input_dir + 'default-new_nodes-NOL-epsilon-0.3-decay-1/network1/NOL_a0.01.csv', r'NOL($\epsilon=0.3$)', '-'),
            (input_dir + 'default-new_nodes-NOL-HTR-epsilon-0.3-decay-1/network1/NOL-HTR_a0.01.csv', r'NOL-HTR($\epsilon_0=0.3$,$k=\ln(n)$)', '-'),
            (nonpara_results + '_KNN-UCB.csv', r'KNN($k=20$)-UCB($\alpha=2.0$)', '-'),
            (input_dir + 'baseline-new_nodes-rand/network1/rand_a0.csv', 'Random', '-'),
            (input_dir + 'baseline-new_nodes-high-jump//network1/high_a0.csv', 'High + Jump', '-'),
            (input_dir + 'baseline-new_nodes-high/network1/high_a0.csv', 'High', '-'),
            (input_dir + 'baseline-new_nodes-low/network1/low_a0.csv', 'Low', '-'),
            ]

    ## initialize plot
    plt.figure(figsize=(8, 6))
    plt.rcParams['xtick.labelsize'] = tick_size
    plt.rcParams['ytick.labelsize'] = tick_size
    plt.rcParams['axes.labelsize'] = label_size
    plt.rcParams['font.size'] = font_size
    plt.rcParams['legend.fontsize'] = legend_font
    for i in range(len(input_files)):
        try:
            df = pd.read_table(input_files[i][0])
        except Exception as e:
            print(e)
            df = None
            continue
        every_n = 10
        indices = list(range(start_probe, num_probes, every_n))
        avg = np.array(df['AvgRewards'][indices])
        std = np.array(df['StdRewards'][indices])
        yminus = avg-std
        yplus = avg+std

        if 'default' not in input_files[i][0]:
            plt.plot(df['Probe'][indices]/ float(N), df['AvgRewards'][indices], color = 'C'+str(i+1), linewidth = 2.0, label=input_files[i][1], linestyle=input_files[i][2], alpha = 1.0)
            plt.fill_between(df['Probe'][indices]/ float(N), yminus, yplus, color = 'C'+str(i+1), alpha=0.3)
        else:
            plt.plot(df['Probe'][indices]/ float(N), df['AvgRewards'][indices], linewidth = 2.0, label=input_files[i][1], linestyle=input_files[i][2], alpha = 1.0)
            plt.fill_between(df['Probe'][indices]/ float(N), yminus, yplus, alpha=0.3)
        print('label ' + str(input_files[i][1]) + ' max: ' + str(max(df['AvgRewards'][indices])))

    if df is not None:

        if N == 1:
            plt.xlabel('t')
        else:
            plt.xlabel('% Nodes Probed')
        plt.ylabel(r'Avg $c_r(t)$')
        if legend:
            leg = plt.legend()
            # set the linewidth of each legend object
            for legobj in leg.legendHandles:
                legobj.set_linewidth(3.0)
    if names[name][0] == 'twitter':
                inset_start_probe = 500
                inset_end_probe = 2000
                inset_indices = list(range(inset_start_probe, inset_end_probe, every_n))
                for i in range(len(input_files)):
                    try:
                        print(input_files[i][1])
                        df = pd.read_table(input_files[i][0])
                        avg = np.array(df['AvgRewards'])
                        std = np.array(df['StdRewards'])
                        yminus = avg-std
                        yplus = avg+std
                    except Exception as e:
                        print(e)
                        df = None
                        continue

                    a = plt.axes([.33, .691, .22, .22])
                    a.tick_params(axis='x', labelsize=14)
                    if 'default' not in input_files[i][0]:
                        a.plot(df['Probe'][inset_indices]/ float(N), df['AvgRewards'][inset_indices], color='C'+str(i+1), linewidth = 2.0, label=input_files[i][1], linestyle=input_files[i][2], alpha = 1.0)
                        a.fill_between(df['Probe'][inset_indices]/ float(N), yminus[inset_indices], yplus[inset_indices], color='C'+str(i+1), alpha=0.3)
                    else:
                        a.plot(df['Probe'][inset_indices]/ float(N), df['AvgRewards'][inset_indices], linewidth = 2.0, label=input_files[i][1], linestyle=input_files[i][2], alpha = 1.0)
                        a.fill_between(df['Probe'][inset_indices]/ float(N), yminus[inset_indices], yplus[inset_indices], alpha=0.3)
                    plt.setp(a, yticks=[])

                plt.savefig(plots_base + '/' + str(out_name) + '-' + out_reward_name + '.pdf')
