import sys
sys.path.append('./')
from bert_for_text_classification import *
from bert_utils import *
from sklearn.model_selection import train_test_split
import csv


topic_ids_map = {}
def load_examples(input_file):
    examples = []
    labels = []
    question_set = []
    is_first_line = True
    with open(input_file, 'r', encoding='utf-8', errors='ignore') as f:
        contents = f.read()
        file_as_list = contents.splitlines()
        for line in file_as_list:
            if is_first_line:
                is_first_line = False
                continue
            split = line.split("\t")
            question = split[1]
            label = split[3]
            if(question not in question_set):
                topic_ids_map[question] = split[0]
                question_set.append(question)
                labels.append(label)
                examples.append((question, label))
        f.close()

    return examples, list(set(labels))


def load_qc_examples(input_file, use_fine_grained_classes = False):
    """Creates examples for the training and dev sets."""
    examples = []
    labels = []

    with open(input_file, 'r', encoding='utf-8', errors='ignore') as f:
        contents = f.read()
        file_as_list = contents.splitlines()
        for line in file_as_list:
            split = line.split(" ")
            question = ' '.join(split[1:])
            if not use_fine_grained_classes:
                label = split[0].split(":")[0]
            else:
                label = split[0]
            labels.append(label)
            examples.append(question)
        f.close()

    return examples, labels


if __name__ == "__main__":
    #--------------------------------
    #  Read the QC dataset
    #--------------------------------

    train_filename = "examples/data_qc/train_5500.label"
    test_filename = "examples/data_qc/TREC_10.label"
    examples, labels = load_qc_examples(train_filename, False)
    t_examples, t_labels = load_qc_examples(test_filename, False)
    print("labels: " + str(set(labels)))
    print("Number of training examples: " + str(len(examples)))
    print("Some training examples:\n")
    for i in range(1, 50):
        print(examples[i], labels[i])
    
    # split the dataset keeping the label distribution (stratify parameter)
    x_train, x_dev, y_train, y_dev = train_test_split(examples, labels, test_size=0.2, random_state=42, stratify=labels)
    
    train_examples = []
    dev_examples = []
    test_examples = []
    train_labels = list(set(labels))
    train_labels.sort()

    #generate tuples 
    for i in range(0,len(x_train)):
        train_examples.append((x_train[i], y_train[i]))
    for i in range(0,len(x_dev)):
        dev_examples.append((x_dev[i], y_dev[i]))
    for i in range(0,len(t_examples)):
        test_examples.append((t_examples[i], t_labels[i]))
    # train_examples, train_labels = load_examples(train_filename)
    # dev_examples, dev_labels = load_examples(dev_filename)
    # test_examples, test_labels = load_examples(test_filename)
    params = ClassifierParameters(train_labels)
    model = Classifier(model_name = "roberta-base", params = params)
    # Put everything in the GPU if available
    if torch.cuda.is_available():    
        model.cuda()
    
    #start training
    model.start_training(train_examples, dev_examples, params)

    #after training load the best model
    model_to_load = "best_model.pickle"
    model = torch.load(model_to_load)
    model.eval()

    test_dataloader = generate_data_loader(test_examples, params.label_to_id, model.tokenizer, params.max_seq_length, params.batch_size, do_shuffle=False)
    evaluate(test_dataloader, model, params, print_classification_output=True)
    ex, probs = getClassProbabilities(model, test_dataloader, params.device)
    
    for i in range(0, len(ex)):
        print ("{} - {} - {}".format(ex[i], max(probs[i]), params.id_to_label[probs[i].index(max(probs[i]))]))






    