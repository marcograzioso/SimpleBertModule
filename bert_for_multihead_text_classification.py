from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, f1_score
import sys
import torch
import io
import torch.nn.functional as F
import numpy as np
import time
import math
import torch.nn as nn
from transformers import *
from bert_utils import *
from tqdm import tqdm


class MultiHeadClassifier(nn.Module):
    """
    Class which implements an exlcusive multiclass classifier 
    """
    def __init__(self, model_name, params = None):
        super(MultiHeadClassifier, self).__init__()
        self.params = params
        # Load the BERT-based encoder
        self.encoder = AutoModel.from_pretrained(model_name, cache_dir="new_cache_dir/")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="new_cache_dir/")
        # The AutoConfig allows to access the encoder configuration. 
        # The configuration is needed to derive the size of the embedding, which 
        # is produced by BERT (and similar models) to encode the input elements. 
        config = AutoConfig.from_pretrained(model_name, cache_dir="new_cache_dir/")
        self.cls_size = int(config.hidden_size)
        # Dropout is applied before the final classifier
        self.input_dropout = nn.Dropout(p=params.out_dropout_rate)
        # Final linear classifiers
        heads = []
        for i in range(params.num_heads):
            heads.append(nn.Linear(self.cls_size,params.num_labels[i]))
        self.fully_connected_layers = nn.ModuleList(heads)

    def forward(self, input_ids, attention_mask):
        # encode all outputs
        model_outputs = self.encoder(input_ids, attention_mask)
        # select the vector associated to the [CLS] symbol used as
        # first token for ALL sentences
        encoded_cls = model_outputs.last_hidden_state[:,0]
        # apply dropout
        encoded_cls_dp = self.input_dropout(encoded_cls)
        # apply the linear classifier
        logits = []
        for fully_connected_layer in self.fully_connected_layers:
            logits.append(fully_connected_layer(encoded_cls_dp))
        # return the logits
        return logits, encoded_cls
    
    def start_training(self, train_examples, dev_examples, params):
        # Define the Optimizer. Here the ADAM optimizer (a sort of standard de-facto) is
        # used. AdamW is a variant which also adopts Weigth Decay.
        params.optimizer = torch.optim.AdamW(self.parameters(), lr=params.learning_rate)
        # More details about the Optimizers can be found here:
        # https://huggingface.co/transformers/main_classes/optimizer_schedules.html
        
        # Define the scheduler
        if params.apply_scheduler:
            # Estimate the numbers of step corresponding to the warmup.
            num_train_examples = len(train_examples)
            num_train_steps = int(num_train_examples / params.batch_size * params.num_train_epochs)
            num_warmup_steps = int(num_train_steps * params.warmup_proportion)
            # Initialize the scheduler
            params.scheduler = get_constant_schedule_with_warmup(params.optimizer, num_warmup_steps = num_warmup_steps)
        train_dataloader = generate_data_loader_for_multihead(train_examples, params.label_to_id, self.tokenizer, params.max_seq_length, params.batch_size, do_shuffle = True)
        dev_dataloader = generate_data_loader_for_multihead(dev_examples, params.label_to_id, self.tokenizer, params.max_seq_length, params.batch_size, do_shuffle = True)
        start_training(self, params, train_dataloader, dev_dataloader)





def start_training(model, params, train_dataloader, dev_dataloader):
    training_stats = []

    # Measure the total training time for the whole run.
    total_t0 = time.time()

    # NOTICE: the measure to be maximized should depends on the task. 
    # Here accuracy is used.
    best_dev_accuracy = -1
    best_dev_f1_weighted = -1

    # For each epoch...
    for epoch_i in range(0, params.num_train_epochs):
        # ========================================
        #               Training
        # ========================================
        # Perform one full pass over the training set.
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, params.num_train_epochs))
        print('Training...')

        # Measure how long the training epoch takes.
        t0 = time.time()

        # Reset the total loss for this epoch.
        train_loss = 0

        # Put the model into training mode.
        model.train() 

        # For each batch of training data...
        for step, batch in enumerate(tqdm(train_dataloader)):
            loss = 0
            if step % params.print_each_n_step == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = format_time(time.time() - t0)
                # Report progress.
                # print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))
                
             # Unpack this training batch from our dataloader. 
            b_input_ids = batch[0].to(params.device)
            b_input_mask = batch[1].to(params.device)
            b_labels = batch[2].to(params.device)
            
            # clear the gradients of all optimized variables
            params.optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            train_logits, _ = model(b_input_ids, b_input_mask)
            # calculate the loss        
            for i in range(params.num_heads):
                loss += params.nll_loss(train_logits[i], b_labels[:,i])      
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward() 
            # perform a single optimization step (parameter update)
            params.optimizer.step()
            # update running training loss
            train_loss += loss.item()
        
            # Update the learning rate with the scheduler, if specified
            if params.apply_scheduler:
                params.scheduler.step()
       
        # Calculate the average loss over all of the batches.
        avg_train_loss = train_loss / len(train_dataloader)
        
        # Measure how long this epoch took.
        training_time = format_time(time.time() - t0)

        print("")
        print("  Average training loss: {0:.3f}".format(avg_train_loss))
        print("  Training epcoh took: {:}".format(training_time))
            
        # ========================================
        #     Evaluate on the Development set
        # ========================================
        # After the completion of each training epoch, measure our performance on
        # our test set.
        print("")
        print("Running Test...")

        t0 = time.time()

        # Put the model in evaluation mode--the dropout layers behave differently
        # during evaluation.
        model.eval()

        # Apply the evaluate_method defined above to estimate 
        avg_dev_loss, dev_accuracy, dev_f1_weighted = evaluate(dev_dataloader, model, params, epoch=epoch_i+1)

        # Measure how long the validation run took.
        test_time = format_time(time.time() - t0)
        for i_h in range(params.num_heads):
            print("  Accuracy head {}: {:.3f}".format(i_h,dev_accuracy[i_h]))
            print("  F1 weightedhead {}: {:.3f}".format(i_h,dev_f1_weighted[i_h]))
        print("  Test Loss: {:.3f}".format(avg_dev_loss))
        print("  Test took: {}".format(test_time))

        # Record all statistics from this epoch.
        training_stats.append(
            {
                'epoch': epoch_i + 1,
                'Training Loss': avg_train_loss,
                'Valid. Loss': avg_dev_loss,
                'Valid. Accur.': dev_accuracy,
                'Training Time': training_time,
                'Test Time': test_time
            }
        )

        # Save the model if the performance on the development set increases
        mean_accuracy = sum(dev_accuracy)/params.num_heads
        if mean_accuracy > best_dev_accuracy:
            best_dev_accuracy = mean_accuracy
            torch.save(model, params.output_model_name)
            print("\n  Saving the model during epoch " + str(epoch_i + 1))
            print("  Actual Best Validation Accuracy: {:.3f} - from these accuracies: {}".format(mean_accuracy, dev_accuracy))






def evaluate(dataloader, model, params, print_classification_output=False, epoch=0):
    '''
    Evaluation method which will be applied to development and test datasets.
    It returns the pair (average loss, accuracy)

    dataloader: a dataloader containing examples to be classified
    classifier: the BERT-based classifier
    print_classification_output: to log the classification outcomes 
    ''' 
    total_loss = 0
    gold_classes = [[] for i in range(params.num_heads)] 
    system_classes = [[] for i in range(params.num_heads)]
    accuracy = [None] * params.num_heads
    preds = [None] * params.num_heads
    output = [None] * params.num_heads
    tokenizer = model.tokenizer
    if print_classification_output:
        print("\n------------------------")
        print("  Classification outcomes")
        print("is_correct\tgold_label\tsystem_label\ttext")
        print("------------------------")
    
    # For each batch of examples from the input dataloader
    for batch in dataloader:
        # Unpack this training batch from our dataloader. Notice this is populated 
        # in the method `generate_data_loader`
        b_input_ids = batch[0].to(params.device)
        b_input_mask = batch[1].to(params.device)
        b_labels = batch[2].to(params.device)
        
        # Tell pytorch not to bother with constructing the compute graph during
        # the forward pass, since this is only needed for backprop (training).
        with torch.no_grad():
            loss = 0
            # Each batch is classifed        
            logits, _ = model(b_input_ids, b_input_mask)
            # Evaluate the loss. 
            for i in range(params.num_heads):
                loss += params.nll_loss(logits[i], b_labels[:,i])
            total_loss += loss

        # Accumulate the predictions and the input labels
        for i in range(params.num_heads):
            _, preds[i] = torch.max(logits[i], 1)
            system_classes[i] += preds[i].detach().cpu()
            gold_classes[i] += b_labels[:,i].detach().cpu()

        # Print the output of the classification for each input element
        if print_classification_output:
            for ex_id in range(len(b_input_mask)):
                input_strings = tokenizer.decode(b_input_ids[ex_id], skip_special_tokens=True)
                # convert class id to the real label
                for i in range(params.num_heads):
                    predicted_label = params.id_to_label[i][preds[i][ex_id].item()]
                    gold_standard_label = "UNKNOWN"
                    # convert the gold standard class ID into a real label
                    if gold_classes[i][ex_id].item() in params.id_to_label[i]:
                        gold_standard_label = params.id_to_label[i][gold_classes[i][ex_id].item()]
                    # put the prefix "[OK]" if the classification is correct
                    output[i] = '[OK]' if predicted_label == gold_standard_label else '[NO]'
                # print the output
                print("\t".join(output) + "\t" + gold_standard_label + "\t" + predicted_label + "\t" + input_strings)
    
    # Calculate the average loss over all of the batches.
    avg_loss = total_loss / len(dataloader)
    avg_loss = avg_loss.item()

    # Report the final accuracy for this validation run.
    for i_h in range(params.num_heads):
        system_classes[i_h] = torch.stack(system_classes[i_h]).numpy()
        gold_classes[i_h] = torch.stack(gold_classes[i_h]).numpy()
        accuracy[i_h] = np.sum(system_classes[i_h] == gold_classes[i_h]) / len(system_classes[i_h])

        if params.print_result_summary or params.log_result_summary: 
            summary = "\n------------------------"
            summary += "  Summary " + str(epoch)
            summary += "------------------------\n"
            #remove unused classes in the test material
            filtered_label_list = []
            for i in range(len(params.labels[i_h])):
                if i in gold_classes[i_h] or i in system_classes[i_h]:
                    filtered_label_list.append(params.id_to_label[i_h][i])
            summary += classification_report(gold_classes[i_h], system_classes[i_h], digits=3, target_names=filtered_label_list)

            summary += "\n------------------------"
            summary += "  Confusion Matrix"
            summary += "------------------------\n"
            conf_mat = confusion_matrix(gold_classes[i_h], system_classes[i_h])
            for row_id in range(len(conf_mat)):
                summary += filtered_label_list[row_id] + "\t" +str(conf_mat[row_id]) + "\n"
            if params.print_result_summary:
                print(summary)
            if params.log_result_summary:
                with open("training_log.log", mode="a", encoding="utf-8") as f:
                    f.write(summary)
                f.close()

    return avg_loss, accuracy, [f1_score(gold_classes[i_h], system_classes[i_h], average='weighted') for i_h in range(params.num_heads)]


def getClassProbabilities(classifier, dataloader, device):
    '''
    Propagate the data through the model and obtain the probabilities
    ''' 
    tokenizer = classifier.tokenizer
    examples = []
    probs = []
    # For each batch of examples from the input dataloader
    for batch in dataloader:   
        # Unpack this training batch from our dataloader. Notice this is populated 
        # in the method `generate_data_loader`
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        
        # Tell pytorch not to bother with constructing the compute graph during
        # the forward pass, since this is only needed for backprop (training).
        with torch.no_grad():
            # Each batch is classifed        
            logits, _ = classifier(b_input_ids, b_input_mask)
            # get the prediction and use the softmax to obtain a probability
            out = torch.nn.functional.softmax(logits, dim=1)
            classes = out.detach().cpu()
            for ex_id in range(len(b_input_mask)):
                input_strings = tokenizer.decode(b_input_ids[ex_id], skip_special_tokens=True)
                # convert class id to the real label
                predicted_prob = classes[ex_id].tolist()
                a = classes[ex_id]
                examples.append(input_strings)
                probs.append(predicted_prob)
    return examples, probs

def getClassProbabilitiesForMultihead(classifier, dataloader, device):
    '''
    Propagate the data through the model and obtain the probabilities
    ''' 
    tokenizer = classifier.tokenizer
    examples = []
    total_probs = []
    # For each batch of examples from the input dataloader
    for batch in tqdm(dataloader, desc="passing data through the model"):  
        
        # Unpack this training batch from our dataloader. Notice this is populated 
        # in the method `generate_data_loader`
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        # b_labels = batch[2].to(device)
        
        # Tell pytorch not to bother with constructing the compute graph during
        # the forward pass, since this is only needed for backprop (training).
        with torch.no_grad():
            # Each batch is classifed        
            logits, _ = classifier(b_input_ids, b_input_mask)
            # get the prediction and use the softmax to obtain a probability
            out = [None for x in range(classifier.params.num_heads)]
            classes = [None for x in range(classifier.params.num_heads)]
            for i, l in enumerate(logits):
                out[i] = torch.nn.functional.softmax(l, dim=1)
                classes[i] = out[i].detach().cpu()
            for ex_id in range(len(b_input_mask)):
                probs = [None for x in range(classifier.params.num_heads)]
                input_strings = tokenizer.decode(b_input_ids[ex_id], skip_special_tokens=False)
                for i, cl in enumerate(classes):
                    predicted_prob = cl[ex_id].tolist()
                    probs[i] = predicted_prob
                total_probs.append(probs)
                examples.append(input_strings)
    return examples, total_probs


def getSentenceEmbeddings(classifier, dataloader, params):
    '''
    Propagate the data through the model and obtain the vectors
    ''' 
    device = params.device
    tokenizer = classifier.tokenizer
    examples = []
    embeddings = []
    labels = []
    # For each batch of examples from the input dataloader
    for batch in dataloader:   
        # Unpack this training batch from our dataloader. Notice this is populated 
        # in the method `generate_data_loader`
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        
        # Tell pytorch not to bother with constructing the compute graph during
        # the forward pass, since this is only needed for backprop (training).
        with torch.no_grad():
            # Each batch is classifed        
            logits, embedding = classifier(b_input_ids, b_input_mask)
            emb_det = embedding.detach().cpu()
            for ex_id in range(len(b_input_mask)):
                input_strings = tokenizer.decode(b_input_ids[ex_id], skip_special_tokens=True)
                examples.append(input_strings)
                embeddings.append(emb_det[ex_id].numpy())
                gold_standard_label = params.id_to_label[b_labels[ex_id].item()]
                labels.append(gold_standard_label)
    return examples, embeddings, labels