# Unsupervised Sentiment Analysis for Code-mixed Data

We use embeddings techniques like MUSE, LASER, XLM, MutltiBPEemd, fasttext 
to efficiently transfer knowledge from monolingual test to code-mix text for sentiment analysis of code-mixed text.

## Environment

All the dependencies of the code are listed in `requirements.txt`. 

### pip
```
    pip install -r requirements.txt
    PYTHONIOENCODING=utf-8 python -m laserembeddings download-models
```

### docker

```
    # build the image 
    docker build -t unsacmt .
    
    # run the container
    nvidia-docker run -v $PWD:/app -p 8989:8989 unsacmt
    
    # launch a jupyter notebook
    jupyter notebook --ip 0.0.0.0 --port 8989 --allow-root
```
## Data 

The Sentiment Analysis data is present is `data/cm/`.    
The custom fastText embedding is provided [here]().  # TODO  
The aligned MUSE embedding is provided [here]().   # TODO

## Files Description

- `notebooks/archive/*.ipynb`: old notebooks with many more experiments than mentioned in the paper.
- `notebooks/Results.ipynb`: a notebook with all the experiments
- `src/utills.py`: code for reading raw data and f1 score
- `src/trainer.py`: code for following training curriculum given the model and data
- `src/models.py`: code for simple neural network models used by use
- `src/data_prep.py`: code for applying different kinds of embeddings on sentiment analysis dataset


 

