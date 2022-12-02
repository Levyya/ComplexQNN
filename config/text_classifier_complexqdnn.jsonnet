
local batch_size = 4;
local cuda_device = 2;
local num_epochs = 3;
local seed = 42;

local embedding_dim = 768;
local hidden_dim = 128;
local dropout = 0.2;
local lr = 0.0003;
local output_dim = 128;
local model_name = 'BERT';
local ptm_name = 
  if model_name == 'RoBERTa'
  then 'roberta-base'
  else 'distilbert-base-uncased';
  
local data_dir = '/workspace/Wei_lai/NLP/Mine_Project/NLP/Mine_Project/AllenNLP/Learning/Baseline_Allennlp/data/';
local get_train_path(task_name='SST') = 
  if task_name == 'SST' 
  then '/workspace/Wei_lai/NLP/data/SST/Binary/sentiment-train'
  else data_dir + task_name + '/' + task_name + '_train.txt';
local get_val_path(task_name='SST') = 
  if task_name == 'SST' 
  then '/workspace/Wei_lai/NLP/data/SST/Binary/sentiment-dev'
  else data_dir + task_name + '/' + task_name + '_test.txt';

local task_name = 'SUBJ';
local train_path = get_train_path(task_name);
local val_path = get_val_path(task_name);

{
  numpy_seed: seed,
  pytorch_seed: seed,
  random_seed: seed,
  
  dataset_reader: {
    type: 'my_dataset_reader',
    tokenizer: {
      type: 'pretrained_transformer',
      model_name: ptm_name
    },
    token_indexers: {
      tokens: {
        type: 'pretrained_transformer',
        model_name: ptm_name
      },
    },
  },
  datasets_for_vocab_creation: ['train'],
  train_data_path: train_path,
  validation_data_path: val_path,
  
  model: {
    type: 'complex_text_classifier',
    embedder_real: {
      token_embedders: {
        tokens: {
          type: 'pretrained_transformer',
          model_name: ptm_name
        },
      },
    },
    embedder_imag: {
      token_embedders: {
        tokens: {
          type: 'pretrained_transformer',
          model_name: ptm_name
        },
      },
    },
    encoder: {
      type: 'complexqnn',
      embedding_dim: embedding_dim,
    }
  },
  data_loader: {
    shuffle: true,
    batch_size: batch_size,
  },
  trainer: {
    cuda_device: cuda_device,
    num_epochs: num_epochs,
    optimizer: {
      lr: lr,
      type: 'adam',
    },
    validation_metric: '+f1',
  }
}
