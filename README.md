# ComplexQNN

The ComplexQNN is a complex-valued quantum-inspired neural network for NLP downstreaming tasks.

**Dependencies**

> 1. pytorch>=1.12
> 2. allennlp==2.10
> 3. complexPyTorch=0.4
>

## Allennlp train command example

```cmd
allennlp train config/text_classifier_cnn.jsonnet --include-package myclassifier -s ./result/mytrain2 -f --dry-run
```


## Some notes about allennlp
1. --dry-run  # load dataset and not train
2. -f  # force trian, this command will override the save path
3. -s  # save path
4. --include-package  # personal work path including model, classifier and so on
5. config/xxx.jsonnet  # config file with jsonnet format


## Our train command
### CNN
```
allennlp train config/text_classifier_cnn.jsonnet --include-package myclassifier -s ./result/mytrain2 -f
```



### ELMo

```cmd
allennlp train config/text_classifier_elmo.jsonnet --include-package myclassifier -s ./result/sst_emlo -f
```


### BERT
```cmd
allennlp train config/text_classifier_bert.jsonnet --include-package myclassifier -s ./result/subj_bert -f
```


### ComplexTextCNN
```cmd
allennlp train config/text_classifier_complexcnn.jsonnet --include-package myclassifier -s ./result/cr_complexcnn -f
```


### ComplexQNN
```cmd
allennlp train config/complexqdnn.jsonnet --include-package work -s ./result/subj_complexqnn -f --dry-run
```



## Other command
```cmd
# delete training models
rm ./result/*/*_state_*.th -f
```