from pipeline import pipeline
import argparse




if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # parser.add_argument('--path', help='Train Config File Path', default="./config/train_config.yaml")
    parser.add_argument('--path', help='Train Config File Path', default="./config/config_template/problem2-compare-method-Softargmax-10fold-absolute-Error-margin.yaml")

    args = parser.parse_args()
    config_path = args.path

    pipeline(config_path)
