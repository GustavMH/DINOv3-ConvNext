# Training-length ablation

Train ConvNeXt-Tiny with web and satellite DINOv3 teachers for 100 epochs. Evaluate checkpoints at epochs 5, 10, 20, 40, 60, 80 and 100 using three probe seeds.

## Install and prepare

```bash
python -m pip install -r requirements.txt
bash scripts/download.sh chesapeake
bash scripts/download.sh loveda
bash scripts/run_pilot.sh prepare
```

## Train

```bash
python -m satdistill.experiment train --teacher web --accelerator gpu --devices 1 --precision bf16-mixed
python -m satdistill.experiment train --teacher sat --accelerator gpu --devices 1 --precision bf16-mixed
```

## Evaluate

```bash
python -m satdistill.probe_all_checkpoints
```

Probes train for 30 epochs with seeds 0, 1 and 2. Maryland training crops are used to evaluate Maryland test and New York test. LoveDA training crops are used to evaluate LoveDA validation.

Scores are saved as JSON files in results/full100/training_length_ablation.

Results may vary with package versions and hardware.