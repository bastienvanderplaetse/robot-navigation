import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import subprocess
import sys

from os import listdir
from os.path import exists, isfile, join


""" CONFIGURATION """

def check_args(argv):
    """ Checks and parses the arguments of the command typed by the user

    Parameters
    ----------
    argv :
        The arguments of the command typed by the user

    Returns
    -------
    ArgumentParser
        the values of the arguments of the command typed by the user
    """
    parser = argparse.ArgumentParser(description="Evaluate a nmt Pytorch model \
        with different configuration files")
    parser.add_argument('FOLDER', type=str, help="the name of the folder \
        containing the configuration files to use")
    parser.add_argument('PARAMETER', type=str, help="the name of the parameter \
        that evolved")

    args = parser.parse_args()

    return args


""" FUNCTIONS """

def get_config_files(folder):
    """ Get all configuration files located in a folder

    Parameters
    ----------
    folder : str
        The name of the folder containing the configuration files

    Returns
    -------
    array
        the array containing the configuration file names
    """
    files = None
    if not exists(folder):
        print("Folder does not exist ({0})".format(folder))
    else:
        files = [join(folder, f) for f in listdir(folder) if isfile(join(folder, f))]
    return files

def run_training(configurations):
    for config in configurations:
        command = "python bin/nmtpy train -C {0}".format(config)
        print(command)
        subprocess.call(["python", "bin/nmtpy", "train", "-C", "{0}".format(config), "-d", "gpu"], shell=True)

def load_scores(filename):
    scores = dict()
    for file in filename:
        with open(file, 'r') as f:
            data = json.load(f)
            for key, value in data.items():
                scores[key] = value
    return scores

def prepare_scores(scores, parameter):
    labels = []
    scores_values = []
    max_len = 0
    for key, values in scores.items():
        labels.append("{0} = {1}".format(parameter, key))
        scores_values.append(values)
        if len(values) > max_len :
            max_len = len(values)

    for values in scores_values:
        if len(values) < max_len:
            values.extend([None] * (max_len-len(values)))
    x_values = [i+1 for i in range(max_len)]

    return scores_values, labels, x_values

def plot_lines(x_values, lines, labels, filename, label_x, label_y, has_legend=True, step=10):
    fig, ax = plt.subplots()

    for index, line in enumerate(lines):
        ax.plot(x_values, line, label=labels[index])


    ax.set_ylabel(label_y)
    ax.set_xlabel(label_x)

    plt.xticks(np.arange(min(x_values)-1, max(x_values)+1, step))

    if has_legend:
        lgd = ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        plt.savefig(filename, bbox_extra_artists=(lgd,), bbox_inches="tight")
    else:
        plt.savefig(filename)
    plt.close(fig)
    plt.clf()

def get_score_files():
    files = None
    folder = "scoresevaluation"
    if not exists(folder):
        print("Folder does not exist ({0})".format(folder))
    else:
        files = [join(folder, f) for f in listdir(folder) if isfile(join(folder, f))]
        files = [f for f in files if len(f.split(".")) > 1 and f.split(".")[-1] == "json"]
    return files



""" EXECUTION """

def run(args):
    """ Executes the main process of the script

    Parameters
    ----------
    args : ArgumentParser
        The arguments of the commande typed by the user
    """
    # configurations = get_config_files(args.FOLDER)
    # bound = len(configurations)//2
    # config = configurations[:]
    # print(config)
    # run_training(config)

    scores_file = get_score_files()
    scores = load_scores(scores_file)
    scores_values, labels, x_values = prepare_scores(scores, args.PARAMETER)
    plot_lines(x_values, scores_values, labels, args.PARAMETER+'.png', "Epoch", "Blue metric")


if __name__ == "__main__":
    args = check_args(sys.argv)
    run(args)
