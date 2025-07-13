import pandas as pd

if __name__ == '__main__':
    train = pd.read_json("./dataset-e-Care/train_full.jsonl", lines=True)
    test = pd.read_json("./dataset-e-Care/dev_full.jsonl", lines=True)
    full = pd.concat([train, test], axis=0)
    full.to_json("./dataset-e-Care/full.jsonl", orient='records', lines=True)
    print(full)