
local batch_size = 64;
local cuda_device = 2;
local num_epochs = 4;
local seed = 42;

local dropout = 0.2;
local lr = 0.001;
local output_dim = 128;
local embedding_dim = 1024;
local hidden_dim = 128;

local data_dir = '/workspace/Wei_lai/NLP/Mine_Project/NLP/Mine_Project/AllenNLP/Learning/Baseline_Allennlp/data/';
local get_train_path(task_name='SST') = 
  if task_name == 'SST' 
  then '/workspace/Wei_lai/NLP/data/SST/Binary/sentiment-train'
  else data_dir + task_name + '/' + task_name + '_train.txt';
local get_val_path(task_name='SST') = 
  if task_name == 'SST' 
  then '/workspace/Wei_lai/NLP/data/SST/Binary/sentiment-dev'
  else data_dir + task_name + '/' + task_name + '_test.txt';

local SST2_train_path = '/workspace/Wei_lai/NLP/data/SST/Binary/sentiment-train';
local SST2_dev_path = '/workspace/Wei_lai/NLP/data/SST/Binary/sentiment-dev';
local SST5_train_path = '/workspace/Wei_lai/NLP/data/SST/Fine-Grained/sentiment-train';
local SST5_dev_path = '/workspace/Wei_lai/NLP/data/SST/Fine-Grained/sentiment-dev';

// 如果数据集为CR, MPQA, MR, SUBJ, 使用get_train_path
/*
local task_name = 'SUBJ';
local train_path = get_train_path(task_name);
local val_path = get_val_path(task_name);
*/

// 如果数据集为SST2，SST5，使用根变量引用
// Note: 如果为多分类，要修改num_classes！
local train_path = SST5_train_path;
local val_path = SST5_dev_path;

{
  numpy_seed: seed,
  pytorch_seed: seed,
  random_seed: seed,
  dataset_reader: {
    type: 'my_dataset_reader',
    tokenizer: {
      type: 'whitespace',
    },
    token_indexers: {
      tokens: {
        type: 'elmo_characters'
      }
    }
  },
  datasets_for_vocab_creation: ['train'],
  train_data_path: train_path,
  validation_data_path: val_path,

  model: {
    type: 'text_classifier',
    embedder: {
      token_embedders: {
        tokens: {
          type: 'elmo_token_embedder'
        }
      }
    },
    encoder: {
      type: 'lstm',
      input_size: embedding_dim,
      hidden_size: hidden_dim
    },
    num_classes: 5
  },
  data_loader: {
    shuffle: true,
    batch_size: batch_size,
  },
  trainer: {
    cuda_device: cuda_device,
    optimizer: {
      lr: lr,
      type: 'adam',
    },
    num_epochs: num_epochs,
    patience: 10,
    validation_metric: '+fscore',
  }
}
