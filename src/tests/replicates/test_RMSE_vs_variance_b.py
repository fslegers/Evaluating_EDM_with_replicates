from multiprocessing import Pool
from functools import partial

from src.classes import *
from src.simulate_lorenz import *


def sample_lorenz(vec_0, params, n_points, obs_noise, offset=0):
    x, y, z, t = simulate_lorenz(vec_0, params, n_points * 100, n_points, obs_noise)
    x, t = sample_from_ts(x, t, sampling_interval=5, n_points=n_points)
    t = [i + offset for i in range(len(t))]
    return x, t

def calculate_performance(result):

    # Calculate performance measures
    diff = np.subtract(result['obs'], result['pred'])
    performance = {}
    performance['MAE'] = np.mean(abs(diff))
    performance['RMSE'] = math.sqrt(np.mean(np.square(diff)))

    try:
        performance['corr'] = pearsonr(result['obs'], result['pred'])[0]
    except:
        performance['corr'] = None

    return performance


def perform_EDM(index, length, n_replicates, noise, variance, rho, x, y, z):

    # Determine max dimensions
    max_dim = int(np.sqrt(length))

    xs, ts = [], []

    # Get original trajectory
    x_, t_ = sample_lorenz([x[0], y[0], z[0]], [10, rho, 8/3], length+9, 0.0, 0)

    # preprocessing
    x_, _ = preprocessing(x_, t_, loc=0)
    x_ = x_ + np.random.normal(0, noise, size=len(x_))
    x_, _ = preprocessing(x_, t_, loc=0)

    xs.append(x_)
    ts.append(t_)

    # Add replicates
    if variance == 0.0:
        repl_indices = [0 for _ in range(n_replicates)]
    else:
        repl_indices = np.random.randint(0 - variance, 0 + variance, size=n_replicates)

    for j in range(n_replicates):
        # generate time series on attractor
        repl_index = repl_indices[j]
        x_, t_ = sample_lorenz([x[repl_index], y[repl_index], z[repl_index]], [10, rho, 8/3], length-1, 0.0, 0)

        # preprocessing
        x_, _ = preprocessing(x_, t_, loc=j)
        x_ += np.random.normal(0, noise, size=len(x_))
        x_, _ = preprocessing(x_, t_, loc=j)

        xs.append(x_)
        ts.append(t_)

    # Put them together into one library
    collection = []
    for index, (a, b) in enumerate(zip(xs, ts)):
        for j in range(len(a)):
            collection.append(Point(a[j], b[j], "", index))
    del (xs, ts, x, x_, t_)

    # Split train and test set
    ts_train = [point for point in collection if point.time_stamp < length]
    ts_test = [point for point in collection if point.time_stamp >= length]
    del (collection)

    # Train model and predict test set
    model = EDM()
    model.train(ts_train, max_dim=max_dim)
    _, smap = model.predict(ts_test, hor=1)

    # Measure performance
    smap = smap.dropna(how='any')
    results = calculate_performance(smap)

    return results


def partial_function(variance, indices, n_replicates, length, noise, rho, x, y, z):
    results = []

    for i in indices:
        partial_result = perform_EDM(i, length=length, n_replicates=n_replicates, rho=rho, noise=noise,
                                     variance=variance, x=x[i-100:i+100], y=y[i-100:i+100], z=z[i-100:i+100])
        RMSE = partial_result['RMSE']

        row = {'noise': noise, 'variance': variance, 'rho': rho, 'length': length, 'n_repl': n_replicates,
               'RMSE': RMSE}

        results.append(row)

    return pd.DataFrame(results)


def run_imap_multiprocessing(func, argument_list, num_processes):
    pool = Pool(processes=num_processes)
    result_list = []
    for result in pool.imap(func=func, iterable=argument_list):
        result_list.append(result)
    combined_df = pd.concat(result_list, ignore_index=True)
    return combined_df


def loop(rho, n_repl, length, noise):
    np.random.seed(123)

    n_processes = 6
    n_iterations = 250
    variances = np.arange(0, 100, 5)

    # Sample initial point for each iteration
    indices = np.random.randint(100, 1100, size=n_iterations)

    x, y, z, t = simulate_lorenz([1, 1, 1], [10, rho, 8 / 3], 2100, 36.75, 0)
    x, y, z = x[900:], y[900:], z[900:]

    # Define partial function
    partial_func = partial(partial_function,
                           indices=indices,
                           n_replicates=n_repl,
                           length=length,
                           noise=noise,
                           rho=rho,
                           x=x,
                           y=y,
                           z=z)

    # First, test with variance in begin conditions
    result_list = run_imap_multiprocessing(func=partial_func,
                                           num_processes=n_processes,
                                           argument_list=variances)

    data = pd.DataFrame(result_list)
    file_name = (f"C:/Users/5605407/Documents/PhD/Chapter_1/Resultaten/"
                 f"RMSE vs variance among replicates/rho = {rho}/b/"
                 f"length = {length}, noise = {noise}, n_repl = {n_repl}.csv")
    data.to_csv(file_name, index=False)
    del (result_list, data)


if __name__ == "__main__":

    i = 1
    rho = 28
    for n_repl in [1, 2, 4, 8, 12]:
        for length in [25]:
            for noise in [0.0, 0.05, 0.1]:
                print(f"starting round {i} of 6")
                loop(rho, n_repl, length, noise)
                i += 1
    rho = 20
    for n_repl in [1, 2, 4, 8, 12]:
        for length in [25]:
            for noise in [0.0, 0.05, 0.1]:
                print(f"starting round {i} of 6")
                loop(rho, n_repl, length, noise)
                i += 1


