from trainer_OOD import Trainer
import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--config_path", type=str, default=None)
    args = parser.parse_args()    
    args.config_path = "./config/config.yaml"
    trainer = Trainer(args.config_path)
    trainer.train()
