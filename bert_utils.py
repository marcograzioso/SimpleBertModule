import torch
import datetime
import random
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

class ClassifierParameters():
    """
    Class which defines the classifier parameters
    """
    def __init__(self, label_list, max_seq_length = 32, out_dropout_rate = 0.1, batch_size = 32, learning_rate = 2e-5, nll_loss = torch.nn.CrossEntropyLoss(ignore_index=-1), 
        output_model_name = "best_model.pickle", num_train_epochs = 10, apply_scheduler = True, warmup_proportion= 0.1, print_each_n_step = 6,
        log_result_summary= False, print_result_summary = False, num_heads = 1 ):
        
        self.print_result_summary = print_result_summary
        self.log_result_summary = log_result_summary
        #set the device
        ##Set random values
        seed_val = 213
        random.seed(seed_val)
        np.random.seed(seed_val)
        torch.manual_seed(seed_val)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed_val)
        
        # If there's a GPU available...
        if torch.cuda.is_available():    
            # Tell PyTorch to use the GPU.    
            self.device = torch.device("cuda:0")
            print('There are %d GPU(s) available.' % torch.cuda.device_count())
            print('We will use the GPU:', torch.cuda.get_device_name(0))
        # If not...
        else:
            print('No GPU available, using the CPU instead.')
            self.device = torch.device("cpu")

        # the maximum length to be considered in input
        self.max_seq_length = max_seq_length
        # dropout applied to the embedding produced by BERT before the classifiation
        self.out_dropout_rate = out_dropout_rate
        # number of linear heads   
        self.num_heads = num_heads
        # --------------------------------
        # Training parameters
        # --------------------------------

        self.nll_loss = nll_loss

        # the batch size
        self.batch_size = batch_size

        # the learning rate used during the training process
        # learning_rate = 2e-5
        #learning_rate = 5e-6
        self.learning_rate = learning_rate

        # if you use large models (such as Bert-large) it is a good idea to use 
        # smaller values, such as 5e-6

        # name of the fine_tuned_model
        self.output_model_name = output_model_name

        # number of training epochs
        self.num_train_epochs = num_train_epochs

        # ADVANCED: Schedulers allow to define dynamic learning rates.
        # You can find all available schedulers here
        # https://huggingface.co/transformers/main_classes/optimizer_schedules.html
        self.apply_scheduler = apply_scheduler
        # Here a `Constant schedule with warmup`can be activated. More details here
        # https://huggingface.co/transformers/main_classes/optimizer_schedules.html#transformers.get_constant_schedule_with_warmup
        self.warmup_proportion = warmup_proportion
        if type(label_list) is list and type(label_list[0]) is not list:  
            self.num_labels = len(label_list)
            self.label_to_id , self.id_to_label = generate_label_maps(label_list)
            self.labels = label_list
        elif type(label_list) is list and type(label_list[0]) is list:
            self.num_labels = []
            self.label_to_id = []
            self.id_to_label = []
            self.labels = []
            for l in label_list:
                self.num_labels.append(len(l))
                self.labels.append(l)
                l_to_id, id_to_l = generate_label_maps(l)
                self.label_to_id.append(l_to_id)
                self.id_to_label.append(id_to_l)

        # --------------------------------
        # Log parameters
        # --------------------------------

        # Print a log each n steps
        self.print_each_n_step = print_each_n_step
        self.optimizer = None
        self.scheduler = None


def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


def generate_label_maps(label_list):
    # Initialize a map to associate labels to the dimension of the embedding 
    # produced by the classifier
    label_to_id_map = {}
    id_to_label_map = {}
    for (i, label) in enumerate(label_list):
        label_to_id_map[label] = i
        id_to_label_map[i] = label
    return label_to_id_map, id_to_label_map



def generate_data_loader(examples, label_map, tokenizer, max_seq_length, batch_size, do_shuffle = False):
    '''
    Generate a Dataloader given the input examples

    examples: a list of pairs (input_text, label)
    label_mal: a dictionary used to assign an ID to each label
    tokenize: the tokenizer used to convert input sentences into word pieces
    do_shuffle: a boolean parameter to shuffle input examples (usefull in training) 
    ''' 
    #-----------------------------------------------
    # Generate input examples to the Transformer
    #-----------------------------------------------
    input_ids = []
    input_mask_array = []
    label_id_array = []

    # Tokenization 
    for (text, label) in examples:
        # tokenizer.encode_plus is a crucial method which:
        # 1. tokenizes examples
        # 2. trims sequences to a max_seq_length
        # 3. applies a pad to shorter sequences
        # 4. assigns the [CLS] special wor-piece such as the other ones (e.g., [SEP])
        encoded_sent = tokenizer.encode_plus(text, add_special_tokens=True, max_length=max_seq_length, padding='max_length', truncation=True)
        # convert input word pieces to IDs of the corresponding input embeddings
        input_ids.append(encoded_sent['input_ids'])
        # store the attention mask to avoid computations over "padded" elements
        input_mask_array.append(encoded_sent['attention_mask'])

        # converts labels to IDs
        id = -1
        if label in label_map:
            id = label_map[label]
        label_id_array.append(id)
        
    # Convert to Tensor which are used in PyTorch
    input_ids = torch.tensor(input_ids) 
    input_mask_array = torch.tensor(input_mask_array)
    label_id_array = torch.tensor(label_id_array, dtype=torch.long)

    # Building the TensorDataset
    dataset = TensorDataset(input_ids, input_mask_array, label_id_array)

    if do_shuffle:
        # this will shuffle examples each time a new batch is required
        sampler = RandomSampler
    else:
        sampler = SequentialSampler

    # Building the DataLoader
    return DataLoader(
                dataset,  # The training samples.
                sampler = sampler(dataset), # the adopted sampler
                batch_size = batch_size) # Trains with this batch size.


def generate_data_loader_for_multihead(examples, label_map, tokenizer, max_seq_length, batch_size, do_shuffle = False):
    '''
    Generate a Dataloader given the input examples

    examples: a list of pairs (input_text, label)
    label_mal: a dictionary used to assign an ID to each label
    tokenize: the tokenizer used to convert input sentences into word pieces
    do_shuffle: a boolean parameter to shuffle input examples (usefull in training) 
    ''' 
    #-----------------------------------------------
    # Generate input examples to the Transformer
    #-----------------------------------------------
    input_ids = []
    input_mask_array = []
    label_id_array = []

    # Tokenization 
    for (text, labels) in examples:
        # tokenizer.encode_plus is a crucial method which:
        # 1. tokenizes examples
        # 2. trims sequences to a max_seq_length
        # 3. applies a pad to shorter sequences
        # 4. assigns the [CLS] special wor-piece such as the other ones (e.g., [SEP])
        encoded_sent = tokenizer.encode_plus(text, add_special_tokens=True, max_length=max_seq_length, padding='max_length', truncation=True)
        # convert input word pieces to IDs of the corresponding input embeddings
        input_ids.append(encoded_sent['input_ids'])
        # store the attention mask to avoid computations over "padded" elements
        input_mask_array.append(encoded_sent['attention_mask'])

        # converts labels to IDs
        label_ids = []
        for i, label in enumerate(labels):
            id = -1
            if label in label_map[i]:
                id = label_map[i][label]
            label_ids.append(id)
        label_id_array.append(label_ids)
    # Convert to Tensor which are used in PyTorch
    input_ids = torch.tensor(input_ids) 
    input_mask_array = torch.tensor(input_mask_array)
    label_id_array = torch.tensor(label_id_array, dtype=torch.long)

    # Building the TensorDataset
    dataset = TensorDataset(input_ids, input_mask_array, label_id_array)

    if do_shuffle:
        # this will shuffle examples each time a new batch is required
        sampler = RandomSampler
    else:
        sampler = SequentialSampler

    # Building the DataLoader
    return DataLoader(
                dataset,  # The training samples.
                sampler = sampler(dataset), # the adopted sampler
                batch_size = batch_size) # Trains with this batch size.