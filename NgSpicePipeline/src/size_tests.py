import os
import pickle

import numpy as np
import torch.optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch import cuda
from torch.utils.data import DataLoader

from NgSpicePipeline.src.NgSpiceTraining import train
from NgSpicePipeline.src.Simulator import Simulator
from NgSpicePipeline.src.trainingUtils import generate_new_dataset_maximum_performance
from Training import dataset, models


def baseline_testing(X_train, X_test, thresholds=None):
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

    return [i / total for i in correct]


if __name__ == '__main__':
    ngspice_exec = "../../ngspice/Spice64/bin/ngspice.exe"
    # train_netlist_two_stage = "../assets/TwoStageAmplifier.sp"
    # test_netlist_two_stage = "../assets/TwoStageAmplifier-Test.sp"
    # param_list_two_stage = ["w0", "w1", "w2"]
    # perform_list_two_stage = ["bw", "pw", "a0"]
    #
    # arguments_two_stage = {
    #     "model_path": "../assets/45nm_CS.pm",
    #     "w0_start": "25u",
    #     "w0_stop": "30u",
    #     "w0_change": "0.25u",
    #     "w2_start": "52u",
    #     "w2_stop": "55.5u",
    #     "w2_change": "0.5u",
    #     "w1_start": "6u",
    #     "w1_stop": "9u",
    #     "w1_change": "0.5u",
    #     "out": "../out/"
    # }
    # simulator_two_stage = Simulator(ngspice_exec, train_netlist_two_stage, test_netlist_two_stage, param_list_two_stage,
    #                                 perform_list_two_stage,
    #                                 arguments_two_stage)
    train_netlist_nmos = "../assets/nmos-training.sp"
    test_netlist_nmos = "../assets/nmos-testing-pro.sp"
    param_list_nmos = ["r", "w"]
    perform_list_nmos = ["bw", "pw", "a0"]

    arguments_nmos = {
        "model_path": "../assets/45nm_CS.pm",
        "w_start": 620,
        "w_stop": 1450,
        "w_change": 5,
        "r_start": "2.88u",
        "r_stop": "6.63u",
        "r_change": "0.20u",
        "out": "../out/"
    }
    simulator_nmos = Simulator(ngspice_exec, train_netlist_nmos, test_netlist_nmos, param_list_nmos, perform_list_nmos,
                               arguments_nmos)

    train_netlist_cascade = "../assets/nmos-training-cascode.sp"
    test_netlist_cascade = "../assets/nmos-testing-cascode.sp"
    param_list_cascade = ["r", "w0", "w1"]
    perform_list_cascade = ["bw", "pw", "a0"]

    arguments_cascade = {
        "model_path": "../assets/45nm_CS.pm",
        "w0_start": 620,
        "w0_stop": 1450,
        "w0_change": 50,
        "w1_start": 620,
        "w1_stop": 1450,
        "w1_change": 50,
        "r_start": "2.88u",
        "r_stop": "6.63u",
        "r_change": "0.7500u",
        "out": "../out/"
    }
    simulator_cascade = Simulator(ngspice_exec, train_netlist_cascade, test_netlist_cascade, param_list_cascade,
                                  perform_list_cascade,
                                  arguments_cascade)

    train_netlist_two_stage = "../assets/TwoStageAmplifier.sp"
    test_netlist_two_stage = "../assets/TwoStageAmplifier-Test.sp"
    param_list_two_stage = ["w0", "w1", "w2"]
    perform_list_two_stage = ["bw", "pw", "a0"]

    arguments_two_stage = {
        "model_path": "../assets/45nm_CS.pm",
        "w0_start": "25u",
        "w0_stop": "30u",
        "w0_change": "0.5u",
        "w2_start": "52u",
        "w2_stop": "55.5u",
        "w2_change": "0.5u",
        "w1_start": "6u",
        "w1_stop": "9u",
        "w1_change": "0.5u",
        "out": "../out/"
    }
    simulator_two_stage = Simulator(ngspice_exec, train_netlist_two_stage, test_netlist_two_stage, param_list_two_stage,
                                    perform_list_two_stage,
                                    arguments_two_stage)
    simulator_two_stage.delete_existing_data = True

    train_netlist_lna = "../assets/LNA.sp"
    test_netlist_lna = "../assets/LNA_test"
    param_list_lna = ["ls", "ld", "lg", "r", "w"]
    perform_list_lna = ["Gmax", "Gp", "s11", "nf"]
    arguments_lna = {
        "model_path": "../assets/45nm_CS.pm",
        "ls_start": "58.3p",
        "ls_stop": "60.8p",
        "ls_change": "0.5p",
        "ld_start": "4.4n",
        "ld_stop": "5.4n",
        "ld_change": "0.5n",
        "lg_start": "14.8n",
        "lg_stop": "15.8n",
        "lg_change": "0.24n",
        "r_start": "800",
        "r_stop": "1050",
        "r_change": "50",
        "w_start": "51u",
        "w_stop": "53u",
        "w_change": "0.4u",
        "out": "../out/"
    }
    simulator_lna = Simulator(ngspice_exec, train_netlist_lna, test_netlist_lna, param_list_lna, perform_list_lna,
                              arguments_lna)
    simulator = simulator_cascade
    simulator.delete_existing_data = True
    num_param, num_perform = len(simulator.parameter_list), len(simulator.performance_list)

    device = 'cuda:0' if cuda.is_available() else 'cpu'
    print(device)

    x, y = simulator.run_training()
    data = np.hstack((x, y))
    scaler_arg = MinMaxScaler()
    data = scaler_arg.fit_transform(data)
    param, perform = data[:, :num_param], data[:, num_param:]
    assert (x.shape[1] == num_param and y.shape[1] == num_perform)
    print(f"Param shape: {param.shape}. Perform shape: {perform.shape}")
    # create new D' dataset. Definition in generate_new_dataset_maximum_performance
    perform, param = generate_new_dataset_maximum_performance(performance=perform, parameter=param, order=[0, 2, 1],
                                                              sign=[1, -1, 1], greater=False)

    X_train, X_test, y_train, y_test = train_test_split(perform, param, test_size=0.1)
    print("Train", X_train.shape, y_train.shape)
    print("Test", X_test.shape, y_test.shape)
    val_data = dataset.CircuitSynthesisGainAndBandwidthManually(X_test, y_test)
    val_dataloader = DataLoader(val_data, batch_size=100)
    num_trials = 5
    size = [500, 1000, 1400]
    all_accs = {}
    for num in size:
        accs = []
        for i in range(num_trials):
            n = np.random.randint(0, X_train.shape[0], num)
            x_train_sub = X_train[n, :]
            y_train_sub = y_train[n, :]
            assert (x_train_sub.shape[1] == num_perform and y_train_sub.shape[1] == num_param), (
                x_train_sub.shape, num_perform, y_train_sub.shape, num_param)

            train_data = dataset.CircuitSynthesisGainAndBandwidthManually(x_train_sub, y_train_sub)
            train_dataloader = DataLoader(train_data, batch_size=100)
            model = models.Model500GELU(num_perform, num_param)
            optimizer = torch.optim.Adam(model.parameters())
            loss_fn = torch.nn.L1Loss()
            losses, val_losses, train_accs, val_accs = train(model, train_dataloader, val_dataloader, optimizer, loss_fn,
                                                             scaler_arg,
                                                             simulator, device=device, num_epochs=2000,
                                                             margin=[0.01, 0.05, 0.1], train_acc=False, sign=[1, -1, 1])

            final_acc = val_accs[-1]
            accs.append(final_acc)
        all_accs[str(num)] = accs

    file = open(f"{os.path.splitext(simulator.train_netlist)[0]}size_tests", 'wb')
    pickle.dump(all_accs, file)
    file.close()

    filehandler = open(f"{os.path.splitext(simulator.train_netlist)[0]}size_tests", 'rb')
    object = pickle.load(filehandler)
    print(object)

    # # Build the plot
    # fig, ax = plt.subplots()
    # ax.bar(x_pos, CTEs, yerr=error, align='center', alpha=0.5, ecolor='black', capsize=10)
    # ax.set_ylabel('Coefficient of Thermal Expansion ($\degree C^{-1}$)')
    # ax.set_xticks(x_pos)
    # ax.set_xticklabels(materials)
    # ax.set_title('Coefficent of Thermal Expansion (CTE) of Three Metals')
    # ax.yaxis.grid(True)
    #
    # # Save the figure and show
    # plt.tight_layout()
    # plt.savefig('bar_plot_with_error_bars.png')
    # plt.show()
    #
    # # pd.DataFrame(data).to_csv(os.path.join(arguments_two_stage["out"], "TwoStageAmpData.csv"))
