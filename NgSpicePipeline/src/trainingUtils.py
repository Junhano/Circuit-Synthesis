import pandas as pd
import numpy as np
import os
import subprocess
import re


def updateFile(trainingfilePath, outputFilePath, argumentMap):
    with open(trainingfilePath, 'r') as read_file:
        file_content = read_file.read()
        for key, val in argumentMap.items():
            temp_pattern = "{" + str(key) + "}"
            file_content = file_content.replace(temp_pattern, str(val))

        with open(outputFilePath, 'w') as write_file:
            write_file.write(file_content)


def convert(filenames):
    files = []
    for file in filenames:
        file_data = pd.read_csv(file, header=None)
        files.append(file_data.apply(lambda x: re.split(r"\s+", str(x))[2], axis=1))

    combine = pd.concat(files, axis=1)
    return np.array(combine, dtype=float)


def getData(param_outfile_names, perform_outfile_names, out):
    param_fullname = [os.path.join(out, file) for file in param_outfile_names]
    perform_fullname = [os.path.join(out, file) for file in perform_outfile_names]
    x = convert(param_fullname)
    y = convert(perform_fullname)
    return x, y


def runSimulation(x1_list, x2_list):


    netlist = "NgSpicePipeline/assets/nmos-testing-pro.sp"
    updated_netlist = "NgSpicePipeline/assets/formatted-nmos-testing.sp"
    pm = "NgSpicePipeline/assets/45nm_CS.pm"
    if type(x1_list) != np.ndarray:
        argumentMap = {
            "model_path": pm,
            "r_array": x1_list,
            "w_array": x2_list,
            "num_samples": 1,
            "out": "NgSpicePipeline/out/"
        }
    else:
        argumentMap = {
            "model_path": pm,
            "r_array": " ".join(list(x1_list.astype(str))),
            "w_array": " ".join(list(x2_list.astype(str))),
            "num_samples": len(x2_list),
            "out": "NgSpicePipeline/out/"
        }
    updateFile(netlist, updated_netlist, argumentMap)

    ngspice_exec = "ngspice/Spice64/bin/ngspice.exe"
    subprocess.run([ngspice_exec, '-r', 'rawfile.raw', '-b', '-i', updated_netlist])

    param_outfile_names = ["r-test.csv", "w-test.csv"]  # must be in order
    perform_outfile_names = ["bw-test.csv", "pw-test.csv", "a0-test.csv"]  # must be in order

    x, y = getData(param_outfile_names, perform_outfile_names, argumentMap["out"])
    return [x, y]


def run_training():
    arguments = {
        "model_path": "NgSpicePipeline/assets/45nm_CS.pm",
        "start1": "2.88u",
        "stop1": "6.63u",
        "change1": "0.3750u",
        "start2": 620,
        "stop2": 1450,
        "change2": 5.5,
        "out": "NgSpicePipeline/out/"
    }
    netlist = "NgSpicePipeline/assets/nmos-training.sp"
    formatted_netlist = "NgSpicePipeline/assets/formatted-nmos-training.sp"
    updateFile(netlist, formatted_netlist, arguments)

    ngspice_exec = "ngspice/Spice64/bin/ngspice.exe"
    subprocess.run([ngspice_exec, '-r', 'rawfile.raw', '-b', '-i', formatted_netlist])

    param_outfile_names = ["r.csv", "w.csv"]  # must be in order
    perform_outfile_names = ["bw.csv", "pw.csv", "a0.csv"]  # must be in order

    x, y = getData(param_outfile_names, perform_outfile_names, arguments["out"])

    return x, y

def generate_duplicate_data(train, test, thresholds):

    return_train, return_test = train, test

    for threshold in thresholds:
        new_train = train * threshold
        return_train = np.concatenate((return_train, new_train), axis=0)
        return_test = np.concatenate((return_test, return_test), axis=0)

    return return_train, return_test


def baseline_testing(X_train, X_test, thresholds = None):

    if thresholds is None:
        thresholds = [0.01, 0.05, 0.1]

    total = X_test.shape[0]
    correct = [0 for _ in range(len(thresholds))]

    for datapoint in X_test:
        for index, threshold in enumerate(thresholds):
            for train_datapoint in X_train:
                diff = np.abs((datapoint - train_datapoint)) / datapoint
                if np.all(diff <= threshold):
                    correct[index] += 1
                    break

    return [i/total for i in correct]


def generate_new_dataset_maximum_performance(performance, parameter, order, sign):

    # parameter x -> performance y using simulator
    # Go through original Dataset D where D consists of pairs of (x,y)
    # For each pair of (x,y)
    # Go through Dataset D again and find out the maximum y' from pair (x',y')
    # that is greater than y in all performance requirement
    # Generate New pair of (x',y) and put them into new dataset


    num_performance = performance.shape[1]
    def sort_helper(greater_val):

        return_val = 0
        counter = 100
        for x in order:
            return_val += counter * greater_val[x]
            counter -= 5
        return return_val

    new_performance = []
    new_parameter = []

    for i in range(len(performance)):
        temp_performance = performance[i, :]

        greater_list = []
        for x in range(len(performance)):
            order_temp_performance = temp_performance[np.array(order)] * sign
            order_compare_performance = performance[x, :][np.array(order)] * sign

            if np.all(order_compare_performance >= order_temp_performance):
                greater_list.append(list(order_compare_performance) + list(parameter[x,:]))


        greater_list.sort(key=sort_helper)
        new_performance.append(temp_performance)

        new_parameter.append(greater_list[0][num_performance:])

    return np.array(new_performance), np.array(new_parameter)






if __name__ == '__main__':
    rerun_training = False
    if rerun_training:
        x, y = run_training()
    else:
        param_outfile_names = ["r.csv", "w.csv"]  # must be in order
        perform_outfile_names = ["bw.csv", "pw.csv", "a0.csv"]  # must be in order
        out = "../out/"
        x, y = getData(param_outfile_names, perform_outfile_names, out)

    data = np.hstack((x, y)).astype(float)

    x1, x2 = np.array(x[0, 0]), np.array(x[0, 1])

    x_sim, y_sim = runSimulation(x1, x2)
    print(x.shape, x_sim.shape)
    print(x[0], y[0])
    print(x_sim[0], y_sim[0])

    # for i in range(x.shape[0]):
    #     assert np.all(x[i] == x_sim[i]),  (x[i] == x_sim[i])
    #     assert np.all(y[i] == y_sim[i]), (y[i] == y_sim[i])


