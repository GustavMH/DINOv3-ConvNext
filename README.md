# Loss-component ablation

Train ConvNeXt-Tiny from the frozen DINOv3 ViT-L/16 (SAT-493M) teacher with one term of the distillation loss disabled, at two budgets, then evaluate every student with a frozen-feature linear segmentation probe.

### Install and prepare

```bash
python -m pip install -r requirements.txt
bash scripts/download.sh chesapeake
bash scripts/run_pilot.sh prepare
```

### Train

```bash
for s in 0 1 2 3; do python -m satdistill.distill --teacher sat --epochs 10  --workers 32 --seed $s --tag full10s$s  --out-dir results; done
for s in 0 1 2;   do python -m satdistill.distill --teacher sat --epochs 100 --workers 32 --seed $s --tag full100s$s --out-dir results; done

python -m satdistill.distill --teacher sat --epochs 10  --workers 32 --tag nolocal    --method-arg loss_local_weight=0 --out-dir results
python -m satdistill.distill --teacher sat --epochs 10  --workers 32 --tag noqueue    --method-arg queue_size=0      --out-dir results
python -m satdistill.distill --teacher sat --epochs 100 --workers 32 --tag e100nolocal --method-arg loss_local_weight=0 --out-dir results
python -m satdistill.distill --teacher sat --epochs 100 --workers 32 --tag e100noqueue --method-arg queue_size=0      --out-dir results
```

### Evaluate

```bash
bash scripts/run_pilot.sh probe
python -m satdistill.summarize --out-dir results
```

`loss_local_weight=0` removes the local patch-relational KL term; `queue_size=0` empties the teacher queue, which makes the global KL term identically zero — confirmed in the resolved training configuration and by `train_loss/global_loss = 0.0000` in the logged components, so this is a component ablation rather than a hyper-parameter change. Probes fit on Maryland train and score on Maryland test and New York test. The aggregated tables are in `results/results.txt`.

Model names in `summary.csv` derive from `--tag`, so a re-run reproduces the numbers under different names. The full ablation takes about 20 GPU-hours. Tested with LightlyTrain 0.18.0, timm 1.0.30 and torch 2.15.0 on a B200 with bf16 AMP. Trained checkpoints and per-run logs are not included. Scores can vary with software versions, hardware, precision and random initialisation.
