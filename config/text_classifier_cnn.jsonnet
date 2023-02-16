
local batch_size = 32;
local cuda_device = 2;
local num_epochs = 5;
local seed = 42;

local embedding_dim = 64;
local dropout = 0.2;
local lr = 0.001;
local max_filter_size = 4;
local num_filters = 64;
local output_dim = 64;
local ngram_filter_sizes = std.range(2, max_filter_size);

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
        type: 'single_id',
        lowercase_tokens: true,
      },
    },
  },
  datasets_for_vocab_creation: ['train'],
  train_data_path: train_path,
  validation_data_path: val_path,
  
  model: {
    type: 'text_classifier',
    embedder: {
      token_embedders: {
        tokens: {
          embedding_dim: embedding_dim,
        },
      },
    },
    encoder: {
      type: 'cnn',
      embedding_dim: embedding_dim,
      ngram_filter_sizes: ngram_filter_sizes,
      num_filters: num_filters,
      output_dim: output_dim,
    },
    num_classes: 5
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
      type: 'adamw',
    },
    validation_metric: '+fscore',
  },
}
