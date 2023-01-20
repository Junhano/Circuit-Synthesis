import os

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def plot_multiple_loss_with_confidence_comparison(loss_mean, loss_upper_bound, loss_lower_bound,
                                                  labels, subsets, save_folder, visual_config, epochs):
    plt.clf()
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    color = visual_config["color"]
    font_size = visual_config["font_size"]
    plt.rcParams.update({'font.size': font_size})
    fig = plt.figure()
    for percentage_index in range(len(subsets)):
        plt.clf()
        ax = fig.add_subplot()
        percentage_loss_mean_cross_comparison = [i[percentage_index] for i in loss_mean]
        percentage_loss_upper_bound_cross_comparison = [i[percentage_index] for i in loss_upper_bound]
        percentage_loss_lower_bound_cross_comparison = [i[percentage_index] for i in loss_lower_bound]

        for compared_item_index in range(len(percentage_loss_mean_cross_comparison)):
            ax.plot(np.arange(epochs), percentage_loss_mean_cross_comparison[compared_item_index], label=labels[compared_item_index],
                    color=color[compared_item_index])
            ax.fill_between(np.arange(epochs), percentage_loss_lower_bound_cross_comparison[compared_item_index],
                            percentage_loss_upper_bound_cross_comparison[compared_item_index], alpha=.3,
                            color=color[compared_item_index])

        ax.set_xlim([0, None])
        ax.set_ylim([0, None])
        ax.legend()
        plt.ylabel("{} {} Loss".format("Train", "L1"))
        plt.xlabel("Epochs")

        image_save_path = os.path.join(save_folder, "subset-{}-loss.png".format(subsets[percentage_index]))
        plt.savefig(image_save_path, dpi=250)


def plot_multiple_accuracy_per_epochs_with_confidence_comparison(accuracy_mean, accuracy_upper_bound, accuracy_lower_bound,
                                                                 labels, subsets, save_folder, visual_config,
                                                                 epochs, check_every, first_eval):
    plt.clf()
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    color = visual_config["color"]
    font_size = visual_config["font_size"]
    plt.rcParams.update({'font.size': font_size})

    step = epochs // check_every
    if first_eval is not None:
        eva_epochs = [i * check_every for i in range(step + 1)]
        eva_epochs[0] = first_eval
    else:
        eva_epochs = [(i + 1) * check_every for i in range(step)]

    fig = plt.figure()
    for percentage_index in range(len(subsets)):
        plt.clf()
        ax = fig.add_subplot()
        percentage_accuracy_mean_cross_comparison = [i[percentage_index] for i in accuracy_mean]
        percentage_accuracy_upper_bound_cross_comparison = [i[percentage_index] for i in accuracy_upper_bound]
        percentage_accuracy_lower_bound_cross_comparison = [i[percentage_index] for i in accuracy_lower_bound]

        for compared_item_index in range(len(percentage_accuracy_mean_cross_comparison)):
            ax.plot(eva_epochs, percentage_accuracy_mean_cross_comparison[compared_item_index], label=labels[compared_item_index],
                    color=color[compared_item_index])
            ax.fill_between(eva_epochs, percentage_accuracy_lower_bound_cross_comparison[compared_item_index],
                            percentage_accuracy_upper_bound_cross_comparison[compared_item_index], alpha=.3,
                            color=color[compared_item_index])

        ax.set_xlim([0, None])
        ax.set_ylim([0, None])
        ax.legend()
        plt.ylabel("Test Success Rate")
        plt.xlabel("Epochs")

        image_save_path = os.path.join(save_folder, "subset-{}-accuracy-per-epochs.png".format(subsets[percentage_index]))
        plt.savefig(image_save_path, dpi=250)

def plot_multiple_margin_with_confidence_comparison(margin_array_mean, margin_array_upper_bound, margin_array_lower_bound,
                                                    labels, subsets, save_folder, visual_config):
    plt.clf()
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    color = visual_config["color"]
    eval_margin = visual_config["margin_threshold"]

    font_size = visual_config["font_size"]
    plt.rcParams.update({'font.size': font_size})

    for percentage_index in range(len(subsets)):
        plt.clf()
        percentage_margin_mean_cross_comparison = [i[percentage_index] for i in margin_array_mean]
        percentage_margin_upper_bound_cross_comparison = [i[percentage_index] for i in margin_array_upper_bound]
        percentage_margin_lower_bound_cross_comparison = [i[percentage_index] for i in margin_array_lower_bound]

        for compared_item_index in range(len(percentage_margin_mean_cross_comparison)):

            plt.plot(eval_margin, percentage_margin_mean_cross_comparison[compared_item_index],
                     label=labels[compared_item_index], color=color[compared_item_index])
            plt.fill_between(eval_margin, percentage_margin_lower_bound_cross_comparison[compared_item_index],
                             percentage_margin_upper_bound_cross_comparison[compared_item_index], alpha=.3, color=color[compared_item_index])


        plt.axvline(x=0.05, linestyle='dashed', color="k")
        plt.legend()
        plt.xscale('log')
        plt.xlabel("Accuracy")
        plt.ylabel("Test Success Rate")

        image_save_path = os.path.join(save_folder, "subset-{}-margin.png".format(subsets[percentage_index]))
        plt.savefig(image_save_path, dpi=250)




def plot_multiple_subset_parameter_margin_accuracy_with_confidence_entrypoint(train_config, visual_config, result, save_name):
    plt.clf()
    eval_margin = visual_config["margin_threshold"]
    result_dict = dict()

    multi_margin_mean = []
    multi_margin_upper_bound = []
    multi_margin_lower_bound = []

    subset = train_config["subset"]
    master_save_path = os.path.join(os.getcwd(), "out_plot", save_name)

    for index, percentage_margin in enumerate(result["test_margins"]):
        plt.clf()
        save_path = os.path.join(master_save_path, "subset-parameter-" + str(subset[index]) + "-margin.png")
        temp_margin_mean, temp_margin_upper_bound, temp_margin_lower_bound = generate_plot_subset_margin_given_result(eval_margin,
                                                                                                                          percentage_margin, train_config,
                                                                                                                          visual_config, save_path, index)
        multi_margin_mean.append(temp_margin_mean)
        multi_margin_upper_bound.append(temp_margin_upper_bound)
        multi_margin_lower_bound.append(temp_margin_lower_bound)


    result_dict["subset_parameter_multi_margin_mean"] = multi_margin_mean
    result_dict["subset_parameter_multi_margin_upper_bound"] = multi_margin_upper_bound
    result_dict["subset_parameter_multi_margin_lower_bound"] = multi_margin_lower_bound

    return result_dict


def plot_multiple_margin_with_confidence_entrypoint(train_config, visual_config, result, save_name, dataset_type):
    plt.clf()

    result_dict = dict()
    if train_config["train_margin_accuracy"]:
        multi_train_mean, multi_train_upper_bound, \
        multi_train_lower_bound= generate_plot_margin_given_result(result["train_margins"],
                                                                                train_config, visual_config, save_name, "train", dataset_type)
        result_dict["multi_train_mean"] = multi_train_mean
        result_dict["multi_train_upper_bound"] = multi_train_upper_bound
        result_dict["multi_train_lower_bound"] = multi_train_lower_bound
    if train_config["test_margin_accuracy"]:

        multi_test_mean, multi_test_upper_bound, multi_test_lower_bound= generate_plot_margin_given_result(result["test_margins"],
                                                                    train_config, visual_config, save_name, "test", dataset_type)

        result_dict["multi_test_mean"] = multi_test_mean
        result_dict["multi_test_upper_bound"] = multi_test_upper_bound
        result_dict["multi_test_lower_bound"] = multi_test_lower_bound

    return result_dict

def plot_multiple_accuracy_with_confidence_entrypoint(train_config, visual_config, result, save_name):
    plt.clf()

    result_dict = dict()
    if train_config["train_accuracy_per_epoch"]:
        multi_train_accuracy, multi_train_accuracy_lower_bound, \
        multi_train_accuracy_upper_bound = generate_plot_accuracy_given_result(result["train_accuracy_per_epoch"],
                                                                            train_config, visual_config, save_name, "train")

        result_dict["multi_train_accuracy"] = multi_train_accuracy
        result_dict["multi_train_accuracy_lower_bound"] = multi_train_accuracy_lower_bound
        result_dict["multi_train_accuracy_upper_bound"] = multi_train_accuracy_upper_bound

    if train_config["test_accuracy_per_epoch"]:
        multi_test_accuracy, multi_test_accuracy_lower_bound, \
        multi_test_accuracy_upper_bound = generate_plot_accuracy_given_result(result["validation_accuracy_per_epoch"],
                                                                            train_config, visual_config, save_name, "test")
        result_dict["multi_test_accuracy"] = multi_test_accuracy
        result_dict["multi_test_accuracy_lower_bound"] = multi_test_accuracy_lower_bound
        result_dict["multi_test_accuracy_upper_bound"] = multi_test_accuracy_upper_bound

    return result_dict

def plot_multiple_loss_with_confidence_entrypoint(train_config, visual_config, result, save_name):
    plt.clf()

    multi_train_loss, multi_train_loss_lower_bounds, multi_train_loss_upper_bounds = generate_plot_loss_given_result(result["train_loss"],
                                                                                                                        train_config, visual_config, save_name, "train")
    multi_test_loss, multi_test_loss_lower_bounds, multi_test_loss_upper_bounds = generate_plot_loss_given_result(result["validation_loss"],
                                                                                                                     train_config, visual_config, save_name, "test")

    result_dict = dict()
    result_dict["multi_train_loss"] = multi_train_loss
    result_dict["multi_test_loss"] = multi_test_loss
    result_dict["multi_train_loss_lower_bound"] = multi_train_loss_lower_bounds
    result_dict["multi_test_loss_lower_bound"] = multi_test_loss_lower_bounds
    result_dict["multi_train_loss_upper_bound"] = multi_train_loss_upper_bounds
    result_dict["multi_test_loss_upper_bound"] = multi_test_loss_upper_bounds

    return result_dict



def generate_plot_subset_margin_given_result(eval_margin, margins, train_config, visual_config, save_path, index):
    plt.clf()
    font_size = visual_config["font_size"]
    plt.rcParams.update({'font.size': font_size})

    subset = train_config["subset"]
    color = visual_config["color"][:len(subset)]
    log = visual_config["log"]

    vertical_point = 0.05
    multi_mean = []
    multi_lower_bound = []
    multi_upper_bound = []

    for i in range(index+1):
        temp_mean = []
        temp_lower_bound = []
        temp_upper_bound = []
        for margin in eval_margin:
            temp_run_result = []
            for run in range(len(margins)):
                inner_run_performance = margins[run][i]
                greater_num = 0
                for j in inner_run_performance:
                    if j <= margin:
                        greater_num += 1
                temp_run_result.append(greater_num / len(inner_run_performance))

            success = np.array(temp_run_result)
            success_mean = np.average(success)
            success_std = stats.sem(success)

            temp_mean.append(success_mean)
            temp_lower_bound.append(success_mean - success_std)
            temp_upper_bound.append(success_mean + success_std)
        multi_mean.append(temp_mean)
        multi_lower_bound.append(temp_lower_bound)
        multi_upper_bound.append(temp_upper_bound)

    for i in range(index+1):
        temp_label = "{}% training parameter".format(subset[i] * 100)

        plt.plot(eval_margin, multi_mean[i], label=temp_label, color=color[i])
        plt.fill_between(eval_margin, multi_lower_bound[i], multi_upper_bound[i], alpha=.3, color=color[i])


    plt.axvline(x=vertical_point, linestyle='dashed', color="k")
    plt.legend()
    if log:
        plt.xscale('log')
    plt.xlabel("Accuracy")
    plt.ylabel("Test Success Rate")


    plt.savefig(save_path, dpi=250)


    return multi_mean, multi_upper_bound, multi_lower_bound


def generate_plot_margin_given_result(margin_errors, train_config, visual_config, save_folder, save_name, dataset_type):


    eval_margin = visual_config["margin_threshold"]

    multi_mean = []
    multi_lower_bound = []
    multi_upper_bound = []

    save_path = os.path.join(os.path.join(os.path.join(os.getcwd(), "out_plot"), save_folder), save_name + f"-margin_{dataset_type}.png")


    #Outer axis is different percentage, inner axis is different run, most inner axis is each prediction number
    for index, percentage_performance in enumerate(margin_errors):
        temp_mean = []
        temp_lower_bound = []
        temp_upper_bound = []
        for margin in eval_margin:
            temp_run_result = []
            for run in range(len(percentage_performance)):
                if train_config["subset_parameter_check"]:
                    inner_run_performance = percentage_performance[run][index]
                else:
                    inner_run_performance = percentage_performance[run][0]
                greater_num = 0
                for i in inner_run_performance:
                    if i <= margin:
                        greater_num += 1
                temp_run_result.append(greater_num / len(inner_run_performance))

            success = np.array(temp_run_result)
            success_mean = np.average(success)
            success_std = stats.sem(success)

            temp_mean.append(success_mean)
            temp_lower_bound.append(success_mean - success_std)
            temp_upper_bound.append(success_mean + success_std)
        multi_mean.append(temp_mean)
        multi_lower_bound.append(temp_lower_bound)
        multi_upper_bound.append(temp_upper_bound)


    plot_margin(multi_mean, multi_upper_bound, multi_lower_bound, visual_config, train_config, save_path)

    return multi_mean, multi_upper_bound, multi_lower_bound


def generate_plot_accuracy_given_result(accuracy, train_config, visual_config, save_folder, save_name):


    save_path = os.path.join(os.path.join(os.path.join(os.getcwd(), "out_plot"), save_folder),
                             save_name + "-accuracy.png")

    multi_accuracy = []
    multi_accuracy_lower_bound = []
    multi_accuracy_upper_bound = []

    for percentage_performance_index in range(len(accuracy)):

        if train_config["subset_parameter_check"]:
            nested_subset_data_index = percentage_performance_index
        else:
            nested_subset_data_index = 0
        temp_performance = [accuracy[percentage_performance_index][i][nested_subset_data_index] for i
                            in range(len(accuracy[percentage_performance_index]))]

        temp_performance_mean = np.average(temp_performance, axis=0)
        temp_performance_std = stats.sem(temp_performance, axis=0)
        multi_accuracy.append(temp_performance_mean)
        multi_accuracy_lower_bound.append(temp_performance_mean - temp_performance_std)
        multi_accuracy_upper_bound.append(temp_performance_mean + temp_performance_std)


    plot_accuracy(multi_accuracy, multi_accuracy_upper_bound, multi_accuracy_lower_bound, visual_config, train_config, save_path)

    return multi_accuracy, multi_accuracy_lower_bound, multi_accuracy_upper_bound


def generate_plot_loss_given_result(loss, train_config, visual_config, save_folder, save_name):


    multi_loss = []
    multi_loss_lower_bounds = []
    multi_loss_upper_bounds = []

    save_path = os.path.join(os.path.join(os.path.join(os.getcwd(), "out_plot"), save_folder),
                             save_name + "-loss.png")

    for percentage_loss_index in range(len(loss)):
        if train_config["subset_parameter_check"]:
            nested_subset_data_index = percentage_loss_index
        else:
            nested_subset_data_index = 0

        temp_loss = [loss[percentage_loss_index][i][nested_subset_data_index] for i
                     in range(len(loss[percentage_loss_index]))]

        temp_loss_mean = np.average(temp_loss, axis=0)
        temp_loss_std = stats.sem(temp_loss, axis=0)

        multi_loss.append(temp_loss_mean)
        multi_loss_lower_bounds.append(temp_loss_mean - temp_loss_std)
        multi_loss_upper_bounds.append(temp_loss_mean + temp_loss_std)

    plot_loss(multi_loss, multi_loss_upper_bounds, multi_loss_lower_bounds, visual_config, train_config, save_path)

    return multi_loss, multi_loss_lower_bounds, multi_loss_upper_bounds

def plot_loss(multi_loss_mean, multi_loss_upper_bounds, multi_loss_lower_bounds, visual_config, train_config, save_name):
    plt.clf()
    font_size = visual_config["font_size"]
    plt.rcParams.update({'font.size': font_size})

    fig = plt.figure()
    ax = fig.add_subplot()
    num_subset = len(multi_loss_mean)
    color = visual_config["color"][:num_subset]
    subset = train_config["subset"]
    epochs = train_config["epochs"]

    for i in range(len(multi_loss_mean)):
        if subset[i] <= 0.5:
            temp_label = "{}% data".format(subset[i] * 100)
        else:
            split_size = np.gcd(int(subset[i] * 100), 100)
            fold = int(100 / split_size)
            temp_label = "{}-fold".format(fold)

        ax.plot(np.arange(epochs), multi_loss_mean[i], label=temp_label, color=color[i])
        ax.fill_between(np.arange(epochs), multi_loss_lower_bounds[i], multi_loss_upper_bounds[i], alpha=.3, color=color[i])


        ax.set_xlim([0, None])
        ax.set_ylim([0, None])
        ax.legend()
        plt.ylabel("{} {} Loss".format(save_name, "L1"))
        plt.xlabel("Epochs")

        plt.savefig(save_name, dpi=250)

def plot_accuracy(multi_accuracy_mean, multi_accuracy_upper_bounds, multi_accuracy_lower_bounds, visual_config, train_config, save_name):

    plt.clf()
    font_size = visual_config["font_size"]
    plt.rcParams.update({'font.size': font_size})

    fig = plt.figure()
    ax = fig.add_subplot()

    subset = train_config["subset"]
    color = visual_config["color"]

    epochs = train_config["epochs"]
    check_every = train_config["check_every"]
    first_eval = train_config["first_eval"]

    step = epochs // check_every

    if first_eval is not None:
        eva_epochs = [i * check_every for i in range(step + 1)]
        eva_epochs[0] = first_eval
    else:
        eva_epochs = [(i + 1) * check_every for i in range(step)]

    for i in range(len(multi_accuracy_mean)):
        if subset[i] <= 0.5:
            temp_label = "{}% data".format(subset[i] * 100)
        else:
            split_size = np.gcd(int(subset[i] * 100), 100)
            fold = int(100 / split_size)
            temp_label = "{}-fold".format(fold)

        ax.plot(eva_epochs, multi_accuracy_mean[i], label=temp_label, color=color[i])
        ax.fill_between(eva_epochs, multi_accuracy_lower_bounds[i],
                        multi_accuracy_upper_bounds[i], alpha=.3, color=color[i])


    ax.set_xlim([0, None])
    ax.set_ylim([0, None])
    ax.legend()
    plt.ylabel("Test Success Rate")
    plt.xlabel("Epochs")

    plt.savefig(save_name, dpi=250)


def plot_margin(multi_margin_mean, multi_margin_upper_bounds, multi_margin_lower_bounds, visual_config, train_config, save_name):
    plt.clf()
    font_size = visual_config["font_size"]
    plt.rcParams.update({'font.size': font_size})


    subset = train_config["subset"]
    color = visual_config["color"][:len(subset)]
    log = visual_config["log"]

    vertical_point = 0.05

    eval_margin = visual_config["margin_threshold"]

    for i in range(len(multi_margin_mean)):
        if subset[i] <= 0.5:
            temp_label = "{}% data threshold".format(subset[i] * 100)
        else:
            split_size = np.gcd(int(subset[i] * 100), 100)
            fold = int(100 / split_size)
            temp_label = "{}-fold threshold".format(fold)

        plt.plot(eval_margin, multi_margin_mean[i], label=temp_label, color=color[i])
        plt.fill_between(eval_margin, multi_margin_lower_bounds[i], multi_margin_upper_bounds[i], alpha=.3, color=color[i])


    plt.axvline(x=vertical_point, linestyle='dashed', color="k")
    plt.legend()
    if log:
        plt.xscale('log')
    plt.xlabel("Accuracy")
    plt.ylabel("Test Success Rate")


    plt.savefig(save_name, dpi=250)