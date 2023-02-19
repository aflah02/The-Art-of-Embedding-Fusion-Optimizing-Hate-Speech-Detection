import pickle, os
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_auc_score

dynahate_dataset_path = "..\\Data_Preprocessing\\PreProcessed_Data\\DynaHate\\"
latenthatred_dataset_path = "..\\Data_Preprocessing\\PreProcessed_Data\\Latent_Hatred\\"
olid_dataset_path = "..\\Data_Preprocessing\\PreProcessed_Data\\OLID\\"

def read_labels(dataset = "dynahate", task = "train"):
    dataset_path = None
    curr_task = None
    labels = []
    
    if dataset == "dynahate":
        dataset_path = dynahate_dataset_path
        if task == "train":
            curr_task = "DynaHate_Training"
        elif task == "dev":
            curr_task = "DynaHate_Val"
        else:
            curr_task = "DynaHate_Test"
    elif dataset == "latenthatred":
        dataset_path = latenthatred_dataset_path
        if task == "train":
            curr_task = "LatentHatred_Training"
        elif task == "dev":
            curr_task = "LatentHatred_Val"
        else:
            curr_task = "LatentHatred_Test"
    else:
        dataset_path = olid_dataset_path
        if task == "train":
            curr_task = "OLID_Training"
        elif task == "dev":
            curr_task = "OLID_Val"
        else:
            curr_task = "OLID_Test"
    dataset_path = os.path.join(dataset_path, curr_task + ".txt")
    with open(dataset_path, "r", encoding="utf8") as file:
        temp = file.readlines()
    file.close()

    for each in temp[1:]:
        curr = each.split()
        labels.append(curr[-1])
    
    return labels

def read_from_pickle(file_name):
    with open(file_name, 'rb') as f:
        temp = pickle.load(f)
    f.close()
    return temp

def process_labels(labels, dataset = "dynahate"):
    if dataset == "dynahate":
        labels = [0 if each == "nothate" else 1 for each in labels]
    return labels

def save_model(model, file_name):
    with open(file_name, 'wb') as f:
        pickle.dump(model, f)
    f.close()

def load_model(file_name):
    with open(file_name, 'rb') as f:
        temp = pickle.load(f)
    f.close()
    return temp

def computeAllScores(y_pred_train, y_pred_dev, y_pred_test, train_labels, dev_labels, test_labels):
    print("Accuracy Train: ", accuracy_score(train_labels, y_pred_train))
    print("Accuracy Dev: ", accuracy_score(dev_labels, y_pred_dev))
    print("Accuracy Test: ", accuracy_score(test_labels, y_pred_test))
    print("Weighted F1 Train: ", f1_score(train_labels, y_pred_train, average="weighted"))
    print("Weighted F1 Dev: ", f1_score(dev_labels, y_pred_dev, average="weighted"))
    print("Weighted F1 Test: ", f1_score(test_labels, y_pred_test, average="weighted"))
    print("Macro F1 Train: ", f1_score(train_labels, y_pred_train, average="macro"))
    print("Macro F1 Dev: ", f1_score(dev_labels, y_pred_dev, average='macro'))
    print("Macro F1 Test: ", f1_score(test_labels, y_pred_test, average='macro'))
    print("Micro F1 Train: ", f1_score(train_labels, y_pred_train, average="micro"))
    print("Micro F1 Dev: ", f1_score(dev_labels, y_pred_dev, average='micro'))
    print("Micro F1 Test: ", f1_score(test_labels, y_pred_test, average='micro'))
    print("Weighted Recall Train: ", recall_score(train_labels, y_pred_train, average="weighted"))
    print("Weighted Recall Dev: ", recall_score(dev_labels, y_pred_dev, average='weighted'))
    print("Weighted Recall Test: ", recall_score(test_labels, y_pred_test, average='weighted'))
    print("Macro Recall Train: ", recall_score(train_labels, y_pred_train, average="macro"))
    print("Macro Recall Dev: ", recall_score(dev_labels, y_pred_dev, average='macro'))
    print("Macro Recall Test: ", recall_score(test_labels, y_pred_test, average='macro'))
    print("Micro Recall Train: ", recall_score(train_labels, y_pred_train, average="micro"))
    print("Micro Recall Dev: ", recall_score(dev_labels, y_pred_dev, average='micro'))
    print("Micro Recall Test: ", recall_score(test_labels, y_pred_test, average='micro'))
    # Confusion Matrix
    print("Confusion Matrix Train: ")
    print(confusion_matrix(train_labels, y_pred_train))
    print("Confusion Matrix Dev: ")
    print(confusion_matrix(dev_labels, y_pred_dev))
    print("Confusion Matrix Test: ")
    print(confusion_matrix(test_labels, y_pred_test))

# Read all the embeddings

bert_dynahate_train_embeddings = read_from_pickle('..\Embeddings\Model_Embeddings\BERT\DynaHate\DynaHate_Training')
bert_dynahate_dev_embeddings = read_from_pickle('..\Embeddings\Model_Embeddings\BERT\DynaHate\DynaHate_Val')
bert_dynahate_test_embeddings = read_from_pickle('..\Embeddings\Model_Embeddings\BERT\DynaHate\DynaHate_Test')

bert_latenthatred_train_embeddings = read_from_pickle('..\Embeddings\Model_Embeddings\BERT\Latent_Hatred\LatentHatred_Training')
bert_latenthatred_dev_embeddings = read_from_pickle('..\Embeddings\Model_Embeddings\BERT\Latent_Hatred\LatentHatred_Val')
bert_latenthatred_test_embeddings = read_from_pickle('..\Embeddings\Model_Embeddings\BERT\Latent_Hatred\LatentHatred_Test')

bert_olid_train_embeddings = read_from_pickle('..\Embeddings\Model_Embeddings\BERT\OLID\OLID_Training')
bert_olid_dev_embeddings = read_from_pickle('..\Embeddings\Model_Embeddings\BERT\OLID\OLID_Val')
bert_olid_test_embeddings = read_from_pickle('..\Embeddings\Model_Embeddings\BERT\OLID\OLID_Test')

bertweet_dynahate_train_embeddings = read_from_pickle('..\Embeddings\Model_Embeddings\BERTweet\DynaHate\DynaHate_Training')
bertweet_dynahate_dev_embeddings = read_from_pickle('..\Embeddings\Model_Embeddings\BERTweet\DynaHate\DynaHate_Val')
bertweet_dynahate_test_embeddings = read_from_pickle('..\Embeddings\Model_Embeddings\BERTweet\DynaHate\DynaHate_Test')

bertweet_latenthatred_train_embeddings = read_from_pickle('..\Embeddings\Model_Embeddings\BERTweet\Latent_Hatred\LatentHatred_Training')
bertweet_latenthatred_dev_embeddings = read_from_pickle('..\Embeddings\Model_Embeddings\BERTweet\Latent_Hatred\LatentHatred_Val')
bertweet_latenthatred_test_embeddings = read_from_pickle('..\Embeddings\Model_Embeddings\BERTweet\Latent_Hatred\LatentHatred_Test')

bertweet_olid_train_embeddings = read_from_pickle('..\Embeddings\Model_Embeddings\BERTweet\OLID\OLID_Training')
bertweet_olid_dev_embeddings = read_from_pickle('..\Embeddings\Model_Embeddings\BERTweet\OLID\OLID_Val')
bertweet_olid_test_embeddings = read_from_pickle('..\Embeddings\Model_Embeddings\BERTweet\OLID\OLID_Test')

hatebert_dynahate_train_embeddings = read_from_pickle('..\Embeddings\Model_Embeddings\HateBERT\DynaHate\DynaHate_Training')
hatebert_dynahate_dev_embeddings = read_from_pickle('..\Embeddings\Model_Embeddings\HateBERT\DynaHate\DynaHate_Val')
hatebert_dynahate_test_embeddings = read_from_pickle('..\Embeddings\Model_Embeddings\HateBERT\DynaHate\DynaHate_Test')

hatebert_latenthatred_train_embeddings = read_from_pickle('..\Embeddings\Model_Embeddings\HateBERT\Latent_Hatred\LatentHatred_Training')
hatebert_latenthatred_dev_embeddings = read_from_pickle('..\Embeddings\Model_Embeddings\HateBERT\Latent_Hatred\LatentHatred_Val')
hatebert_latenthatred_test_embeddings = read_from_pickle('..\Embeddings\Model_Embeddings\HateBERT\Latent_Hatred\LatentHatred_Test')

hatebert_olid_train_embeddings = read_from_pickle('..\Embeddings\Model_Embeddings\HateBERT\OLID\OLID_Training')
hatebert_olid_dev_embeddings = read_from_pickle('..\Embeddings\Model_Embeddings\HateBERT\OLID\OLID_Val')
hatebert_olid_test_embeddings = read_from_pickle('..\Embeddings\Model_Embeddings\HateBERT\OLID\OLID_Test')