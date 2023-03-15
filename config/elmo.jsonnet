
local batch_size = 32;
local cuda_device = 2;
local num_epochs = 2;
local seed = 42;

local dropout = 0.2;
local lr = 0.001;
local output_dim = 128;
local embedding_dim = 1024;
local hidden_dim = 128;

local SST2_train_path = './data/SST/Binary/sentiment-train';
local SST2_val_path = './data/SST/Binary/sentiment-dev';
local SST5_train_path = './data/SST/Fine-Grained/sentiment-train';
local SST5_val_path = './data/SST/Fine-Grained/sentiment-dev';

local data_dir = './data/';
local get_train_path(task_name='SST-2') = 
  if task_name == 'SST-2' 
  then SST2_train_path
  else data_dir + task_name + '/' + task_name + '_train.txt';
local get_val_path(task_name='SST-2') = 
  if task_name == 'SST-2' 
  then SST2_val_path
  else data_dir + task_name + '/' + task_name + '_test.txt';

// Please choose dataset with task_name! ['CR', 'MPQA', 'MR', 'SST-2', 'SUBJ', 'SST-5']
local task_name = 'CR';
local num_classes = if task_name == 'SST-5' then 5 else 2;

local train_path = 
  if task_name == 'SST-5'
  then SST5_train_path
  else get_train_path(task_name);
local val_path = 
  if task_name == 'SST-5'
  then SST5_val_path
  else get_val_path(task_name);

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
    },
    task_name: task_name,
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
    num_classes: num_classes
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
