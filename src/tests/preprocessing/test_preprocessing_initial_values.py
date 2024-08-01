from src.classes import *
from src.simulate_lorenz import *
import numpy as np

import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from multiprocessing import Pool
from functools import partial


def calculate_performance(result):
    diff = np.subtract(result['obs'], result['pred'])

    performance = {}

    performance['MAE'] = np.mean(abs(diff))
    performance['RMSE'] = math.sqrt(np.mean(np.square(diff)))

    try:
        performance['corr'] = pearsonr(result['obs'], result['pred'])[0]
    except:
        performance['corr'] = None

    return performance


def sample_lorenz(vec_0, params, n_points, obs_noise):
    x, y, z, t = simulate_lorenz(vec_0, params, n_points * 100, n_points, obs_noise)
    x, t = sample_from_ts(x, t, sampling_interval=5, n_points=n_points)
    t = [i for i in range(len(t))]
    return x, t


def sample_multiple_initial_values(vec_0, rho, n_points, remove_trend, normalization, n_repl, obs_noise, var):

    list_x, list_t, trends, means, std_devs = [], [], [], [], []

    for i in range(n_repl):
        x_0 = np.random.normal(vec_0[0], var)
        y_0 = np.random.normal(vec_0[1], var)
        z_0 = np.random.normal(vec_0[2], var)
        x, t = sample_lorenz([x_0, y_0, z_0], [10, rho, 8/3], n_points, 0.0)

        # add noise
        obs_noise_repl = obs_noise * np.std(x)
        x += np.random.normal(loc=0, scale=obs_noise_repl, size=len(x))

        if remove_trend:
            x, trend_model = remove_linear_trend(x, t)
            trends.append(trend_model)

        if normalization:
            x, mean, std_dev = normalize(x)
            means.append(mean)
            std_devs.append(std_dev)

        list_x.append(x)
        list_t.append(t)

    return list_x, list_t, trends, means, std_devs


def sample_multiple_rhos(vec_0, rho, n_points, remove_trend, normalization, n_repl, obs_noise, var):

    list_x, list_t, trends, means, std_devs = [], [], [], [], []

    i = 0
    while i < n_repl:
        rho = np.random.normal(rho, var)
        x, t = sample_lorenz(vec_0, [10, rho, 8/3], n_points, 0.0)

        # Preprocessing
        x, _ = preprocessing(x, t, loc=i + 1)
        x += np.random.normal(0.0, obs_noise, len(x))
        x, _ = preprocessing(x, t, loc=i + 1)

        if remove_trend:
            x, trend_model = remove_linear_trend(x, t)
            trends.append(trend_model)

        if normalization:
            x, mean, std_dev = normalize(x)
            means.append(mean)
            std_devs.append(std_dev)

        list_x.append(x)
        list_t.append(t)

        i += 1

    return list_x, list_t, trends, means, std_devs


def modelling(vec_0, rho, train_length, remove_trend, normalization, n_repl, obs_noise, var, test):

    if test == "begin_conditions":
        xs, ts, trends, means, std_devs = sample_multiple_initial_values(vec_0, rho, train_length+9, remove_trend,
                                                                         normalization, n_repl, obs_noise, var)
    else:
        xs, ts, trends, means, std_devs = sample_multiple_rhos(vec_0, rho, train_length + 9, remove_trend,
                                                                         normalization, n_repl, obs_noise, var)

    # Put them together into one library
    collection = []
    for i, (x, t) in enumerate(zip(xs, ts)):
        for j in range(len(x)):
            collection.append(Point(x[j], t[j], "A", i))

    ts_train = [point for point in collection if point.time_stamp < train_length]
    ts_test = [point for point in collection if point.time_stamp >= train_length]

    model = EDM()
    model.train(ts_train, max_dim=10)
    simplex, smap = model.predict(ts_test, hor=1)

    smap = smap.dropna()

    # Reverse preprocessing
    if normalization:
        loc = 0
        while loc < n_repl:
            mean, std_dev = means[loc], std_devs[loc]
            loc_filter = smap['location'] == loc
            smap.loc[loc_filter, 'obs'] = reverse_normalization(smap.loc[loc_filter, 'obs'], mean, std_dev)
            smap.loc[loc_filter, 'pred'] = reverse_normalization(smap.loc[loc_filter, 'pred'], mean, std_dev)
            loc += 1

    if remove_trend:
        loc = 0
        while loc < n_repl:
            trend = trends[loc]
            loc_filter = smap['location'] == loc
            try:
                smap.loc[loc_filter, 'obs'] = add_linear_trend(trend, smap.loc[loc_filter, 'obs'],
                                                               smap.loc[loc_filter, 'time_stamp'])
                smap.loc[loc_filter, 'pred'] = add_linear_trend(trend, smap.loc[loc_filter, 'pred'],
                                                                smap.loc[loc_filter, 'time_stamp'])
            except:
                print("...")
            loc += 1

    results = calculate_performance(smap)
    return results


def modelling_parallel(vec, len, n_repl, obs_noise, rho, test):
    results = {}
    var = 1.0
    results['FF'] = modelling(vec, rho, len, False, False, n_repl=n_repl, obs_noise=obs_noise, var=var, test=test)['RMSE']
    results['TT'] = modelling(vec, rho, len, True, True, n_repl=n_repl, obs_noise=obs_noise, var=var, test=test)['RMSE']
    return results


def loop(n_iter, n_processes, rho, length, n_repl, noise, test):
    # simulate one big giant Lorenz attractor without transient
    x, y, z, t = simulate_lorenz([1, 1, 1], [10, rho, 8 / 3], 2000, 35, 0)
    x, y, z = x[1000:], y[1000:], z[1000:]

    # select different initial points from this trajectory
    initial_vecs = []
    indices = np.random.randint(1000, size=n_iter)
    for i in indices:
        initial_vecs.append([x[i], y[i], z[i]])

    # Prepare to save results
    no_preprocessing = []
    trend_removed_normalized = []

    # Parallelize the loop
    partial_modelling_parallel = partial(modelling_parallel, rho=rho, len=length, n_repl=n_repl, obs_noise=noise, test=test)

    with Pool(processes=n_processes) as pool:
        results = list(pool.map(partial_modelling_parallel, initial_vecs))

    for result in results:
        no_preprocessing.append(result['FF'])
        trend_removed_normalized.append(result['TT'])

    # perform repeated measures ANOVA
    data = {'RMSE': no_preprocessing + trend_removed_normalized,
            'group': ["FF"] * n_iter + ["TT"] * n_iter,
            'subject': list(range(1, n_iter + 1)) * 2
            }
    df = pd.DataFrame(data)

    path_name = (f"C:/Users/5605407/Documents/PhD/Chapter_1/Resultaten/"
                 f"preprocessing/IC, rho={rho}, length={length}, n_repl={n_repl}, noise={noise}, test={test}")

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
    n_processes = 4

    for rho in [28, 20]:
        for len in [25, 75]:
            for n_repl in [1, 5, 9, 17]:
                for noise in [0, 0.05, 0.1]:
                    for test in ['begin_conditions', 'rho']:
                        loop(n_iter=n_iter,
                             n_processes=n_processes,
                             rho=rho,
                             n_repl=n_repl,
                             noise=noise,
                             length=len,
                             test=test)





