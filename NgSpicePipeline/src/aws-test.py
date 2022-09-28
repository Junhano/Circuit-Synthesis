from pipeline import *
from circuit_config import *
import torch
from visualutils import *

if __name__ == '__main__':

    if __name__ == '__main__':
        device = 'cpu'
        pipeline_simulator = two_stage_circuit()
        rerun_training = True
        model = models.Model500GELU
        loss = torch.nn.L1Loss()
        loss_name = 'L1'
        epochs = 100

        subset = [0.5, 0.9]
        check_every = 50
        first_eval = 1
        baseline, test_margins, train_margins, test_loss, train_loss, test_accuracy, \
        _, mean_err, mean_perform_err, mean_baseline_err, mean_baseline_performance_err, mean_err_std, \
        mean_performance_err_std, mean_baseline_err_std, \
        mean_baseline_performance_err_std = CrossFoldValidationPipeline(pipeline_simulator, rerun_training, model, loss,
                                                                        epochs, check_every, subset,
                                                                        generate_new_dataset=True, device=device,
                                                                        first_eval=first_eval)

        margins = [0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]

        print(mean_err)
        print(mean_perform_err)
        print(mean_baseline_err)
        print(mean_baseline_performance_err)

        print(mean_err_std)
        print(mean_performance_err_std)
        print(mean_baseline_err_std)
        print(mean_baseline_performance_err_std)

        multi_test_mean_margin, multi_test_upper_bound_margin, multi_test_lower_bound_margin, baseline_test_mean_margin, \
        baseline_test_upper_bound_margin, baseline_test_lower_bound_margin = graph_multiple_margin_with_confidence_cross_fold(
            test_margins, margins, subset, baseline)

        multi_accuracy, multi_accuracy_lower_bound, \
        multi_accuracy_upper_bound = plot_multiple_accuracy_with_confidence_cross_fold(test_accuracy, epochs,
                                                                                       check_every, subset,
                                                                                       first_eval=first_eval)

        multi_loss, multi_loss_lower_bounds, multi_loss_upper_bounds = plot_multiple_loss_with_confidence_cross_fold(
            test_loss, epochs, subset, loss_name)

        save_info_dict = {
            "multi_test_mean_margin": multi_test_mean_margin,
            "multi_test_upper_bound_margin": multi_test_upper_bound_margin,
            "multi_test_lower_bound_margin": multi_test_lower_bound_margin,
            "baseline_test_mean_margin": baseline_test_mean_margin,
            "baseline_test_upper_bound_margin": baseline_test_upper_bound_margin,
            "baseline_test_lower_bound_margin": baseline_test_lower_bound_margin,
            "multi_accuracy": multi_accuracy,
            "multi_accuracy_lower_bound": multi_accuracy_lower_bound,
            "multi_accuracy_upper_bound": multi_accuracy_upper_bound,
            "multi_loss": multi_loss,
            "multi_loss_lower_bounds": multi_loss_lower_bounds,
            "multi_loss_upper_bounds": multi_loss_upper_bounds,
            "mean_err": mean_err,
            "mean_performance_err": mean_perform_err,
            "mean_baseline_err": mean_baseline_err,
            "mean_baseline_performance_err": mean_baseline_performance_err,
            "mean_err_std": mean_err_std,
            "mean_performance_err_std": mean_performance_err_std,
            "mean_baseline_err_std": mean_baseline_err_std,
            "mean_baseline_performance_err_std": mean_baseline_performance_err_std
        }

        save_output_data(save_info_dict, "two-stage")