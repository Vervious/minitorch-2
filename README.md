# MiniTorch Module 4

<img src="https://minitorch.github.io/minitorch.svg" width="50%">

* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module4.html

This module requires `fast_ops.py`, `cuda_ops.py`, `scalar.py`, `tensor_functions.py`, `tensor_data.py`, `tensor_ops.py`, `operators.py`, `module.py`, and `autodiff.py` from Module 3.


Additionally you will need to install and download the MNist library.

(On Mac, this may require installing the `wget` command)

```
pip install python-mnist
mnist_get_data.sh
```


* Tests:

```
python run_tests.py
```

This assignment requires the following files from the previous assignments. You can get these by running

```bash
python sync_previous_module.py previous-module-dir current-module-dir
```

The files that will be synced are:

        minitorch/tensor_data.py minitorch/tensor_functions.py minitorch/tensor_ops.py minitorch/operators.py minitorch/scalar.py minitorch/scalar_functions.py minitorch/module.py minitorch/autodiff.py minitorch/module.py project/run_manual.py project/run_scalar.py project/run_tensor.py minitorch/operators.py minitorch/module.py minitorch/autodiff.py minitorch/tensor.py minitorch/datasets.py minitorch/testing.py minitorch/optim.py minitorch/tensor_ops.py minitorch/fast_ops.py minitorch/cuda_ops.py project/parallel_check.py tests/test_tensor_general.py


# Notes to self

for virtualenv:

```python3.12 -m venv .venv
source .venv/bin/activate
```

for streamlit
```python -m pip install -Ue .
streamlit run project/app.py -- 1
```

for pre-submission
```pre-commit run --all
```

to install minitorch
```pip install -e .
```

for running in bg
```screen -dm bash -c 'python -u project/run_mnist_multiclass.py > log-screen.txt 2>&1'
```

```
screen -dm bash -c 'python -u project/run_sentiment.py > log-screen.txt 2>&1'
```


# Submissions

See `mnist.txt` and `sentiment.txt` in the root directory of the repository.