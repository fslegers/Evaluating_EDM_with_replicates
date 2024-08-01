import time

from src.classes import *
from src.simulate_lorenz import *
import os


def get_parameters_from_name(file):
    # length
    length = file[9:12]
    length = length.replace(",", "")
    length = int(length)

    # noise
    noise_i = file.find('noise = ') + 8

    # n_repl
    n_repl_i = file.find('n_repl = ') + 9
    n_repl = file[n_repl_i:n_repl_i + 2]
    n_repl = n_repl.replace(".", "")
    n_repl = int(n_repl)

    noise = file[noise_i:n_repl_i-11]
    noise = float(noise)

    return length, noise, n_repl

def make_df(rho, test):
    df = pd.DataFrame()
    path = (f"C:/Users/5605407/Documents/PhD/Chapter_1/Resultaten/"
             f"RMSE vs variance among replicates/rho = {rho}/{test}")

    for filename in os.listdir(path):
        if filename.endswith('.csv'):
            file_path = os.path.join(path, filename)
            file_df = pd.read_csv(file_path)
            file_df = file_df.dropna(how='any')

            len, noise, n_repl = get_parameters_from_name(filename)

            for variance in file_df['variance'].unique():
                sub_df = file_df[file_df['variance'] == variance]

                # get 10th and 90th percentiles and mean RMSE
                percentiles = sub_df['RMSE'].describe(percentiles=[0.25, 0.75]).loc[['mean', '25%', '75%']]

                row = pd.DataFrame({'length': [len], 'noise': [noise], 'n_repl': [n_repl], 'variance': [variance],
                                    '25th': [percentiles['25%']], '75th': [percentiles['75%']], 'mean_RMSE': [percentiles['mean']]})
                df = pd.concat([df, row])

    return df

def plot_RMSE_vs_variance(df, test, rho):
    plt.rc('font', size=14)
    colors = plt.get_cmap('tab10')
    markers = ['s', 'o', 'v', '*']

    for length in [25]:
        for n_repl in df['n_repl'].unique():
            i = 0
            for noise in df['noise'].unique():
                color = colors(i)
                marker = markers[i]

                sub_df = df[(df['length'] == length) & (df['noise'] == noise) & (df['n_repl'] == n_repl)]

                plt.plot(sub_df['variance'], sub_df['mean_RMSE'], color=color)
                plt.scatter(sub_df['variance'], sub_df['mean_RMSE'], color=color, marker=marker, s=50, label=f'{noise}')

                plt.fill_between(sub_df['variance'], sub_df['25th'], sub_df['75th'], alpha=.1, color=color, zorder=1)
                plt.plot(sub_df['variance'], sub_df['25th'], linestyle="--", dashes=(5, 5), color=color, alpha=.5)
                plt.plot(sub_df['variance'], sub_df['75th'], linestyle="--", dashes=(5, 5), color=color, alpha=.5)
                i += 1

            plt.ylim((0, 0.5))

            plt.savefig(f"C:/Users/5605407/Documents/PhD/Chapter_1/Resultaten/RMSE vs variance among replicates/Figures/rho = {rho}/{test}/"
                        f"{length}, {n_repl}")
            plt.show()

            time.sleep(2)


if __name__ == '__main__':

    df_28_IC  = make_df(28, 'begin_conditions')
    df_28_rho = make_df(28, 'rho')
    df_28_b = make_df(28, 'b')

    plot_RMSE_vs_variance(df_28_IC, 'begin_conditions', 28)
    plot_RMSE_vs_variance(df_28_rho, 'rho', 28)
    plot_RMSE_vs_variance(df_28_b, 'b', 28)

    ##################################################

    df_20_IC = make_df(20, 'begin_conditions')
    df_20_rho = make_df(20, 'rho')
    df_20_b = make_df(20, 'b')

    plot_RMSE_vs_variance(df_20_IC, 'begin_conditions', 20)
    plot_RMSE_vs_variance(df_20_rho, 'rho', 20)
    plot_RMSE_vs_variance(df_20_b, 'b', 20)









