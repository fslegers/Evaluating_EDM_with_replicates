import time

import matplotlib.pyplot as plt
import pandas as pd

from src.classes import *
from src.simulate_lorenz import *
from matplotlib.backends.backend_pdf import PdfPages
import os


def get_parameters_from_name(file):
    #f"length = {length}, noise = {noise}, n_repl = {n_repl}.csv")
    # Length
    length = file[9:12]
    length = length.replace(",", "")
    length = int(length)

    # noise
    noise_i = file.find('noise = ') + 8

    # var
    var_i = file.find('var = ') + 6
    var = file[var_i:var_i + 2]
    var = var.replace(".", "")
    var = float(var)

    noise = file[noise_i:var_i - 8]
    noise = float(noise)

    return length, noise, var

def make_df(rho, test):
    df = pd.DataFrame()
    path = (f"C:/Users/5605407/Documents/PhD/Chapter_1/Resultaten/"
            f"RMSE vs n_repl/rho = {rho}/{test}")

    for filename in os.listdir(path):
        if filename.endswith('.csv'):
            file_path = os.path.join(path, filename)
            file_df = pd.read_csv(file_path)
            file_df = file_df.dropna(how='any')

            len, noise, var = get_parameters_from_name(filename)

            for n_repl in file_df['n_repl'].unique():
                sub_df = file_df[file_df['n_repl'] == n_repl]

                # get 10th and 90th percentiles and mean RMSE
                percentiles = sub_df['RMSE'].describe(percentiles=[0.25, 0.75]).loc[['mean', '25%', '75%']]

                row = pd.DataFrame({'length': [len], 'noise': [noise], 'n_repl': [n_repl], 'variance': [var],
                                    '25th': [percentiles['25%']], '75th': [percentiles['75%']], 'mean_RMSE': [percentiles['mean']]})
                df = pd.concat([df, row])

    return df

def plot_RMSE_vs_n_repl(df, rho, test):
    plt.rc('font', size=14)

    colors = plt.get_cmap('tab10')
    markers = ['s', 'o', 'v']

    for var in df['variance'].unique():

        i = 0
        for noise in df['noise'].unique():
            color = colors(i)
            marker = markers[i]

            sub_df = df[(df['noise'] == noise) & (df['variance'] == var)]

            plt.plot(sub_df['n_repl'], sub_df['mean_RMSE'], color=color)
            plt.scatter(sub_df['n_repl'], sub_df['mean_RMSE'], color=color, marker=marker, s = 50, label=f'{float(var)}')

            plt.fill_between(sub_df['n_repl'], sub_df['25th'], sub_df['75th'], alpha=.1, color=color, zorder=1)
            plt.plot(sub_df['n_repl'], sub_df['25th'], linestyle="--", dashes=(5, 5), color=color, alpha=.5)
            plt.plot(sub_df['n_repl'], sub_df['75th'], linestyle="--", dashes=(5, 5), color=color, alpha=.5)
            i += 1

        if rho == 28:
            plt.ylim((0, 0.55))
        else:
            plt.ylim((0, 0.25))
        plt.xticks([0, 4, 8, 12, 16])
        plt.savefig(f"C:/Users/5605407/Documents/PhD/Chapter_1/Resultaten/RMSE vs n_repl/Figures/"
                    f"rho = {rho}/{test}/25, var={var}.png")
        plt.show()

        time.sleep(2)




if __name__ == '__main__':

    df_28  = make_df(28, 'begin_conditions')
    plot_RMSE_vs_n_repl(df_28, 28, 'begin_conditions')

    df_28 = make_df(28, 'rho')
    plot_RMSE_vs_n_repl(df_28, 28, 'rho')

    df_28 = make_df(28, 'b')
    plot_RMSE_vs_n_repl(df_28, 28, 'b')

    ################################################

    df_20  = make_df(20, 'begin_conditions')
    plot_RMSE_vs_n_repl(df_20, 20, 'begin_conditions')

    df_20 = make_df(20, 'rho')
    plot_RMSE_vs_n_repl(df_20, 20, 'rho')

    df_20 = make_df(20, 'b')
    plot_RMSE_vs_n_repl(df_20, 20, 'b')









