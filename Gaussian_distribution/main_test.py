from test_OOD import Test
import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--config_path", type=str, default=None)
    args = parser.parse_args()    
    args.config_path = 'config.yaml'
    test = Test(args.config_path)
    test.test_per_epoch()
