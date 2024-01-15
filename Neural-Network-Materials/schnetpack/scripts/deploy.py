#!/usr/bin/env python3
import torch
import argparse

def save_jit_model(model, model_path):

    # add metadata
    metadata = dict()
    metadata["cutoff"] = str(model.representation.interactions[0].cutoff_network.cutoff.item()).encode("ascii")
    scripted_model = torch.jit.script(model)
    torch.jit.save(scripted_model, model_path, _extra_files=metadata)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("model_path")
    parser.add_argument("deployed_model_path")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    model = torch.load(args.model_path, map_location=args.device)
    save_jit_model(model, args.deployed_model_path)

    print(f"stored deployed model at {args.deployed_model_path}.")
