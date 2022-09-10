from NgSpicePipeline.src.Simulator import Simulator
import numpy as np

if __name__ == '__main__':
    ngspice_exec = "../../ngspice/Spice64/bin/ngspice.exe"
    train_netlist_lna = "../assets/LNA.sp"
    test_netlist_lna = "../assets/LNA_test.sp"
    param_list_lna = ["ls", "ld", "lg", "w"]
    perform_list_lna = ["Gp", "s11", "nf"]
    arguments_lna = {
        "model_path": "../assets/45nm_CS.pm",
        "ls_start": "58.3p",
        "ls_stop": "60.3p",
        "ls_change": "0.25p",
        "ld_start": "4.4n",
        "ld_stop": "6n",
        "ld_change": "0.2n",
        "lg_start": "14.8n",
        "lg_stop": "16.4n",
        "lg_change": "0.2n",
        "w_start": "51u",
        "w_stop": "52.8u",
        "w_change": "0.3u",
        "out": "../out/"
    }
    arguments_lna = {
        "model_path": "../assets/45nm_CS.pm",
        "ls_start": "58.3p",
        "ls_stop": "60.3p",
        "ls_change": "1p",
        "ld_start": "4.4n",
        "ld_stop": "6n",
        "ld_change": "0.75n",
        "lg_start": "14.8n",
        "lg_stop": "16.4n",
        "lg_change": "0.5n",
        "w_start": "51u",
        "w_stop": "52.8u",
        "w_change": "0.8u",
        "out": "../out/"
    }
    simulator_lna = Simulator(ngspice_exec, train_netlist_lna, test_netlist_lna, param_list_lna, perform_list_lna,
                              arguments_lna)
    simulator_lna.delete_existing_data = True

    # train_netlist_cascade = "../assets/nmos-training-cascode.sp"
    # test_netlist_cascade = "../assets/nmos-testing-cascode.sp"
    # param_list_cascade = ["r", "w0", "w1"]
    # perform_list_cascade = ["bw", "pw", "a0"]
    #
    # arguments_cascade = {
    #     "model_path": "../assets/45nm_CS.pm",
    #     "w0_start": 620,
    #     "w0_stop": 1450,
    #     "w0_change": 50,
    #     "w1_start": 620,
    #     "w1_stop": 1450,
    #     "w1_change": 50,
    #     "r_start": "2.88u",
    #     "r_stop": "6.63u",
    #     "r_change": "0.7500u",
    #     "out": "../out/"
    # }
    # simulator_cascade = Simulator(ngspice_exec, train_netlist_cascade, test_netlist_cascade, param_list_cascade,
    #                               perform_list_cascade,
    #                               arguments_cascade)
    simulator_lna.save_error_log = True
    param, perform = simulator_lna.run_training()
    print(param.shape, perform.shape)
    param_sim, perform_sim = simulator_lna.runSimulation(param) #simulator_lna.runSimulation(parameters=param)
    perform_sim = np.array(perform_sim)
    perform = np.array(perform)
    print("sim\t real")
    count = 0
    print("="*5,"Perform","="*5)
    for i in range(perform_sim.shape[0]):
        cond = np.allclose(perform_sim[i, :], perform[i, :])
        if cond:
            count += 1
        print(i,perform_sim[i, :], perform[i, :], cond)
    print(count, "/", perform.shape[0])
    print("=" * 5, "End Perform", "=" * 5)
    print("=" * 5, "param", "=" * 5)
    count = 0
    for i in range(param_sim.shape[0]):
        cond = np.allclose(param_sim[i, :], param[i, :])
        if cond:
            count += 1
        print(i,param_sim[i, :], param[i, :], cond)
    print("=" * 5, "end param", "=" * 5)
    print(count, "/", param.shape[0])
    assert (np.allclose(perform_sim, perform)), "failed"
    print("success")
