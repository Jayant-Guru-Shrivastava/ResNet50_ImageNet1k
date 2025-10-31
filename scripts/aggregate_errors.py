import pandas as pd, argparse, os

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--out_dir', default='runs/analysis')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df = pd.read_csv(args.csv)
    df['is_miscls'] = df.true_idx != df.pred_idx

    # Top confusing pairs (true -> predicted)
    pairs = (df[df.is_miscls]
             .groupby(['true_label','pred_label'])
             .size().sort_values(ascending=False))
    pairs.to_csv(f"{args.out_dir}/top_confusions.csv")

    # Per-class accuracy
    per_cls = (df.assign(correct=~df.is_miscls)
               .groupby(['true_idx','true_label'])['correct']
               .mean().sort_values())
    per_cls.to_csv(f"{args.out_dir}/per_class_accuracy.csv")

    print("Wrote:", f"{args.out_dir}/top_confusions.csv", "and", f"{args.out_dir}/per_class_accuracy.csv")

if __name__ == "__main__":
    main()
