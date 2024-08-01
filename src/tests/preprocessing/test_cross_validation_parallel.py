from src.classes import *
from src.simulate_lorenz import *
import numpy as np
from multiprocessing import Pool
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from functools import partial


def calculate_performance(result):
    diff = np.subtract(result['obs'], result['pred'])

    performance = {}

    # performance['MAE'] = np.mean(abs(diff))
    performance['RMSE'] = math.sqrt(np.mean(np.square(diff)))

    # try:
    #     performance['corr'] = pearsonr(result['obs'], result['pred'])[0]
    # except:
    #     performance['corr'] = None

    return performance


def sample_lorenz(vec_0, params, n_points, obs_noise):
    x, y, z, t = simulate_lorenz(vec_0, params, n_points * 100, n_points, obs_noise)
    x, t = sample_from_ts(x, t, sampling_interval=5, n_points=n_points)
    t = [i for i in range(len(t))]
    return x, t


def sample_multiple_initial_values(vec_0, n_points, n_repl, obs_noise, var, rho):
    list_x, list_t, list_preprocessing = [], [], []

    i = 0
    while i < n_repl:
        x_0 = np.random.normal(vec_0[0], var)
        y_0 = np.random.normal(vec_0[1], var)
        z_0 = np.random.normal(vec_0[2], var)

        x, t = sample_lorenz([x_0, y_0, z_0], [10, rho, 8 / 3], n_points, 0.0)

        # Preprocessing
        x, _ = preprocessing(x, t, loc=i)
        x += np.random.normal(0.0, obs_noise, len(x))
        x, _ = preprocessing(x, t, loc=i)

        list_x.append(x)
        list_t.append(t)

        i += 1

    return list_x, list_t


def sample_multiple_rhos(vec_0, n_points, n_repl, obs_noise, var, rho):
    list_x, list_t = [], []

    i = 0
    while i < n_repl:
        rho = np.random.normal(rho, var)
        x, t = sample_lorenz(vec_0, [10, rho, 8 / 3], n_points, 0.0, 0)

        # Preprocessing
        x, _ = preprocessing(x, t, loc=i + 1)
        x += np.random.normal(0.0, obs_noise, len(x))
        x, _ = preprocessing(x, t, loc=i + 1)

        list_x.append(x)
        list_t.append(t)

        i += 1

    return list_x, list_t


def modelling(vec_0, ts_length, obs_noise, rho, cv, frac):

    # if test == "begin_conditions":
    xs, ts = sample_multiple_initial_values(vec_0=vec_0, n_points=ts_length + 9, obs_noise=obs_noise, var=1, rho=rho, n_repl=4)
    # else:
    #     xs, ts, trends, means, std_devs = sample_multiple_rhos(vec_0, ts_length + 9, obs_noise, var)

    # Put them together into one library
    collection = []
    for i, (x, t) in enumerate(zip(xs, ts)):
        for j in range(len(x)):
            collection.append(Point(x[j], t[j], "A", i))

    # Split into training and test set
    ts_train = [point for point in collection if point.time_stamp <= ts_length]
    ts_test = [point for point in collection if point.time_stamp > ts_length]

    # Train model
    model = EDM(cv_method=cv, cv_fraction=frac)
    model.train(ts_train, max_dim=10)
    simplex, smap = model.predict(ts_test, hor=1)
    del(simplex)

    # Calculate RMSEs
    smap = smap.dropna()
    results = calculate_performance(smap)

    return results


def modelling_parallel(vec, length, obs_noise, rho):

    results = {}
    results['LB_25'] = modelling(vec, length, obs_noise, rho, cv="LB", frac=0.25)['RMSE']
    results['LB_50'] = modelling(vec, length, obs_noise, rho, cv="LB", frac=0.50)['RMSE']
    results['RB_4']  = modelling(vec, length, obs_noise, rho, cv="RB", frac=0.25)['RMSE']
    results['RB_8']  = modelling(vec, length, obs_noise, rho, cv="RB", frac=0.125)['RMSE']

    return results

def loop(n_iter, n_processes, rho, length, noise):
    # simulate one big giant Lorenz attractor without transient
    x, y, z, t = simulate_lorenz([1, 1, 1], [10, rho, 8/3], 2000, 35, 0)
    x, y, z = x[1000:], y[1000:], z[1000:]

    # select different initial points from this trajectory
    initial_vecs = []
    indices = np.random.randint(1000, size=n_iter)
    for i in indices:
        initial_vecs.append([x[i], y[i], z[i]])

    # Prepare to save results
    LB25 = []
    LB50 = []
    RB4 = []
    RB8 = []

    # Parallelize the loop
    partial_modelling_parallel = partial(modelling_parallel, rho=rho, length=length, obs_noise=noise)

    with Pool(processes=n_processes) as pool:
        results = pool.map(partial_modelling_parallel, initial_vecs)

    for result in results:
        LB25.append(result['LB_25'])
        LB50.append(result['LB_50'])
        RB4.append(result['RB_4'])
        RB8.append(result['RB_8'])

    len_25 = len(LB25)
    len_50 = len(LB50)
    len_4 = len(RB4)
    len_8 = len(RB8)

    # perform repeated measures ANOVA
    data = {'RMSE': LB25 + LB50 + RB4 + RB8,
            'group': ["LB_25"] * len(LB25) + ["LB_50"] * len(LB50) +
                     ["RB_4"] * len(RB4) + ["RB_8"] * len(RB8),
            'subject': list(range(1, len(LB25) + 1)) * 4
            }
    df = pd.DataFrame(data)

    path_name = (f"C:/Users/5605407/Documents/PhD/Chapter_1/Resultaten/"
                 f"cross_validation/IC, rho={rho}, length={length}, noise={noise}")

    # Fit repeated measures ANOVA model
    model = ols('RMSE ~ C(group) + C(subject)', data=df).fit()

    # Perform ANOVA
    anova_table = sm.stats.anova_lm(model, typ=2)
    anova_table.to_csv(f"{path_name} +, anova_results.csv")

    # Perform pairwise post-hoc tests (e.g., Tukey HSD)
    tukey_results = pairwise_tukeyhsd(endog=df['RMSE'], groups=df['group'])
    tukey_df = pd.DataFrame(data=tukey_results._results_table.data[1:],
                            columns=tukey_results._results_table.data[0])
    tukey_df.to_csv(f"{path_name} +, tukey_results.csv", index=False)


if __name__ == '__main__':

    n_iter = 100
    n_processes = 6

    for rho in [28, 20]:
        for length in [25, 75]:
                for noise in [0, 0.05, 0.1]:
                    for test in ['begin_conditions']:
                        loop(n_iter=n_iter,
                             n_processes=n_processes,
                             rho=rho,
                             noise=noise,
                             length=length)




