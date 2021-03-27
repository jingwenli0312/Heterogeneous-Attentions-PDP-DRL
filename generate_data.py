import os
import numpy as np
from utils.data_utils import check_extension, save_dataset
import scipy.stats as stats
import torch
import matplotlib.pyplot as plt
import pickle
import argparse


def generate_pdp_data(dataset_size, pdp_size, is_gaussian, sigma):

    if is_gaussian:
        def truncated_normal(graph_size, sigma):
            mu = 0.5
            lower, upper = 0, 1
            X = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)

            return torch.stack([torch.from_numpy(X.rvs(graph_size)), torch.from_numpy(X.rvs(graph_size))], 1).tolist()

        def generate(dataset_size, graph_size):
            data = []
            for i in range(dataset_size):
                data.append(truncated_normal(graph_size, sigma))

            return data
        return list(zip(truncated_normal(dataset_size, sigma),  # Depot location
                        generate(dataset_size, graph_size)
                ))
    else:
        return list(zip(np.random.uniform(size=(dataset_size, 2)).tolist(),  # Depot location
                        np.random.uniform(size=(dataset_size, pdp_size, 2)).tolist()
                        ))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", help="Filename of the dataset to create (ignores datadir)")
    parser.add_argument("--data_dir", default='data', help="Create datasets in data_dir/problem (default 'data')")
    parser.add_argument("--name", type=str, required=True, help="Name to identify dataset")
    parser.add_argument("--problem", type=str, default='pdp')
    parser.add_argument("--is_gaussian", type=str, default=False)
    parser.add_argument('--data_distribution', type=str, default=None,
                        help="Distributions to generate for problem, default 'all'.")

    parser.add_argument("--dataset_size", type=int, default=10000, help="Size of the dataset")
    parser.add_argument('--graph_sizes', type=int, nargs='+', default=[20, 40, 80, 120],
                        help="Sizes of problem instances (default 20, 40, 80, 120)")
    parser.add_argument('--sigma', type=float, nargs='+', default={0.6, 0.8, 1.0})
    parser.add_argument("-f", action='store_true', help="Set true to overwrite")
    parser.add_argument('--seed', type=int, default=1234, help="Random seed")

    opts = parser.parse_args()

    assert opts.filename is None or (len(opts.problems) == 1 and len(opts.graph_sizes) == 1), \
        "Can only specify filename when generating a single dataset"

    distributions_per_problem = {
        'pdp': [None]
    }
    problems = {
        opts.problem: [opts.data_distribution]
    }

    for problem, distributions in problems.items():
        for distribution in distributions or [None]:
            for graph_size in opts.graph_sizes:

                datadir = os.path.join(opts.data_dir, problem)
                os.makedirs(datadir, exist_ok=True)

                if opts.filename is None:
                    if opts.is_gaussian == False:
                        filename = os.path.join(datadir, "{}{}{}_{}_seed{}.pkl".format(
                            problem,
                            "_{}".format(distribution) if distribution is not None else "",
                            graph_size, opts.name, opts.seed))
                    else:
                        filename = os.path.join(datadir, "{}{}{}_{}_seed{}_{}_{}.pkl".format(
                            problem,
                            "_{}".format(distribution) if distribution is not None else "",
                            graph_size, opts.name, opts.seed, 'gaussian', opts.sigma[0]))
                else:
                    filename = check_extension(opts.filename)

                assert opts.f or not os.path.isfile(check_extension(filename)), \
                    "File already exists! Try running with -f option to overwrite."

                np.random.seed(opts.seed)
                if problem == 'pdp':
                    dataset = generate_pdp_data(opts.dataset_size, graph_size, opts.is_gaussian, opts.sigma[0])
                else:
                    assert False, "Unknown problem: {}".format(problem)
                print(dataset[0])
                save_dataset(dataset, filename)
   


