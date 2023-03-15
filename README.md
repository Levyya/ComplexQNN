# ComplexQNN

The ComplexQNN is a quantum-inspired complex-valued neural network for NLP tasks.

**Dependencies**

> 1. pytorch>=1.12
> 2. allennlp==2.10
> 3. complexPyTorch=0.4
>

## Allennlp train command example

```cmd
allennlp train config/complexqnn.jsonnet --include-package work -s ./result/mytrain2 -f --dry-run
```


## Some notes about allennlp
1. --dry-run  # load dataset but do not train the model
2. -f  # force training, this command will override the save path
3. -s  # save path
4. --include-package $path  # personal work path including model, classifier and so on
5. config/xxx.jsonnet  # config file with jsonnet format


## Our train command
### CNN
```
allennlp train config/cnn.jsonnet --include-package work -s ./result/cr_cnn -f --dry-run
```

### GRU
```
allennlp train config/gru.jsonnet --include-package work -s ./result/mpqa_gru -f --dry-run
```

### ELMo

```cmd
allennlp train config/elmo.jsonnet --include-package work -s ./result/sst2_emlo -f
```


### BERT
```cmd
allennlp train config/bert.jsonnet --include-package work -s ./result/subj_bert -f
```


### ComplexTextCNN
```cmd
allennlp train config/complexcnn.jsonnet --include-package work -s ./result/cr_complexcnn -f
```


### ComplexQNN
```cmd
allennlp train config/complexqdnn.jsonnet --include-package work -s ./result/subj_complexqnn -f
```

## Dataset
Please modify the variable "task_name" in model.jsonnet to change different datasets.

## Other command
```cmd
# delete training models
rm ./result/*/*_state_*.th -f
```