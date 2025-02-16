# minitorch

This was done following https://minitorch.github.io/. Much was imported from prior repositories where I previously did earlier implementation tasks.

## Notes to self

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


## Evaluations

See `mnist.txt` and `sentiment.txt` in the root directory of the repository.
