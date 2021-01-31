# Heterogeneous Attentions for Solving Pickup and Delivery Problem via Deep Reinforcement Learning

Attention based model for learning to solve the Pickup and Delivery Problem (PDP) using heterogeneous attention mechanism. Training with REINFORCE with greedy rollout baseline.

## Paper
For more details, please see our paper [Heterogeneous Attentions for Solving Pickup andDelivery Problem via Deep Reinforcement Learning](./paper) which has been accepted at [IEEE Transactions on Intelligent Transportation Systems](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=6979). If this code is useful for your work, please cite our paper:

```
@article{li2021heterogeneous,
  title={Heterogeneous Attentions for Solving Pickup and Delivery Problem via Deep Reinforcement Learning},
  author={Li, Jingwen and Xin, Liang and Cao, Zhiguang and Andrew, Lim and Song, Wen and Zhang, Jie},
  journal={IEEE Transactions on Intelligent Transportation Systems},
  year={2021}
}
``` 

## Dependencies

* Python>=3.6
* NumPy
* SciPy
* [PyTorch](http://pytorch.org/)>=1.1
* tqdm
* [tensorboard_logger](https://github.com/TeamHG-Memex/tensorboard_logger)
* Matplotlib (optional, only for plotting)

## Quick start

For training PDP instances with 20 nodes and using rollout as REINFORCE baseline:
```bash
python run.py --graph_size 20 --baseline rollout --run_name 'PDP20_rollout'
```

## Usage

### Generating data

Training data is generated on the fly. To generate validation and test data with uniform distribution (same as used in the paper) for pdp, and sigma is meaningful when 'is_gaussian' is True.
```bash
python generate_data.py --name validation --seed 1234 --is_gaussian False --sigma 1.0
python generate_data.py --name test --seed 6666 --is_gaussian False --sigma 1.0
```
To generate test data with gaussian distribution with sigma=1.0 (same as used in the paper) for pdp:
```bash
python generate_data.py --name test --seed 6666 --is_gaussian True --sigma 1.0
```

### Training

For training PDP instances with 20 nodes and using rollout as REINFORCE baseline and using the generated validation set:
```bash
python run.py --graph_size 20 --baseline rollout --run_name 'pdp20_rollout' --val_dataset data/pdp/pdp20_validation_seed1234.pkl
```

#### Multiple GPUs
By default, training will happen *on all available GPUs*. To disable CUDA at all, add the flag `--no_cuda`. 
Set the environment variable `CUDA_VISIBLE_DEVICES` to only use specific GPUs:
```bash
CUDA_VISIBLE_DEVICES=2,3 python run.py 
```
Note that using multiple GPUs has limited efficiency for small problem sizes (up to 50 nodes).

#### Warm start

The `--load_path` option can be used to load an earlier run, in which case also the optimizer state will be loaded:
```bash
python run.py --graph_size 20 --load_path 'outputs/pdp_20/pdp20_rollout_{datetime}/epoch-0.pt'
```

### Evaluation
To evaluate a model, you can add the `--eval-only` flag to `run.py`, or use `eval.py`, which will additionally measure timing and save the results:
```bash
python eval.py data/pdp/pdp20_test_seed6666.pkl --model 'outputs/pdp_20/pdp20_rollout_{datetime}/epoch-{epoch_number}.pt' --decode_strategy greedy
```
If the epoch is not specified, by default the last one in the folder will be used.

#### Sampling
To report the best of 1280 sampled solutions, use
```bash
python eval.py data/pdp/pdp20_test_seed6666.pkl --model 'outputs/pdp_20/pdp20_rollout_{datetime}/epoch-{epoch_number}.pt' --decode_strategy sample --width 1280 --eval_batch_size 1
```

### Other options and help
```bash
python run.py -h
python eval.py -h
```

## Acknowledgements
Thanks to [wouterkool/attention-learn-to-route](https://github.com/wouterkool/attention-learn-to-route) for getting me started with the code for the Pointer Network.
