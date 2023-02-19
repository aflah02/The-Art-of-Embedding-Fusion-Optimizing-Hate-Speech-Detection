import pickle, os
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_auc_score
import json

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
    # print("Reading from pickle file: ", file_name)
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

def computeAllScores(y_pred_train_dev, y_pred_test, train_dev_labels, test_labels, save_name):
    print("Accuracy Train Dev: ", accuracy_score(train_dev_labels, y_pred_train_dev))
    print("Accuracy Test: ", accuracy_score(test_labels, y_pred_test))
    print("Weighted F1 Train Dev: ", f1_score(train_dev_labels, y_pred_train_dev, average="weighted"))
    print("Weighted F1 Test: ", f1_score(test_labels, y_pred_test, average="weighted"))
    print("Macro F1 Train Dev: ", f1_score(train_dev_labels, y_pred_train_dev, average="macro"))
    print("Macro F1 Test: ", f1_score(test_labels, y_pred_test, average='macro'))
    print("Micro F1 Train Dev: ", f1_score(train_dev_labels, y_pred_train_dev, average="micro"))
    print("Micro F1 Test: ", f1_score(test_labels, y_pred_test, average='micro'))
    print("Weighted Recall Train Dev: ", recall_score(train_dev_labels, y_pred_train_dev, average="weighted"))
    print("Weighted Recall Test: ", recall_score(test_labels, y_pred_test, average='weighted'))
    print("Macro Recall Train Dev: ", recall_score(train_dev_labels, y_pred_train_dev, average="macro"))
    print("Macro Recall Test: ", recall_score(test_labels, y_pred_test, average='macro'))
    print("Micro Recall Train Dev: ", recall_score(train_dev_labels, y_pred_train_dev, average="micro"))
    print("Micro Recall Test: ", recall_score(test_labels, y_pred_test, average='micro'))
    # Confusion Matrix
    print("Confusion Matrix Train Dev: ")
    print(confusion_matrix(train_dev_labels, y_pred_train_dev))
    print("Confusion Matrix Test: ")
    print(confusion_matrix(test_labels, y_pred_test))
    # Json with all the scores
    scores = {}
    scores["Accuracy Train Dev"] = accuracy_score(train_dev_labels, y_pred_train_dev)
    scores["Accuracy Test"] = accuracy_score(test_labels, y_pred_test)
    scores["Weighted F1 Train Dev"] = f1_score(train_dev_labels, y_pred_train_dev, average="weighted")
    scores["Weighted F1 Test"] = f1_score(test_labels, y_pred_test, average="weighted")
    scores["Macro F1 Train Dev"] = f1_score(train_dev_labels, y_pred_train_dev, average="macro")
    scores["Macro F1 Test"] = f1_score(test_labels, y_pred_test, average='macro')
    scores["Micro F1 Train Dev"] = f1_score(train_dev_labels, y_pred_train_dev, average="micro")
    scores["Micro F1 Test"] = f1_score(test_labels, y_pred_test, average='micro')
    scores["Weighted Recall Train Dev"] = recall_score(train_dev_labels, y_pred_train_dev, average="weighted")
    scores["Weighted Recall Test"] = recall_score(test_labels, y_pred_test, average='weighted')
    scores["Macro Recall Train Dev"] = recall_score(train_dev_labels, y_pred_train_dev, average="macro")
    scores["Macro Recall Test"] = recall_score(test_labels, y_pred_test, average='macro')
    scores["Micro Recall Train Dev"] = recall_score(train_dev_labels, y_pred_train_dev, average="micro")
    scores["Micro Recall Test"] = recall_score(test_labels, y_pred_test, average='micro')
    scores["Confusion Matrix Train Dev"] = confusion_matrix(train_dev_labels, y_pred_train_dev).tolist()
    scores["Confusion Matrix Test"] = confusion_matrix(test_labels, y_pred_test).tolist()
    with open(save_name, 'w') as f:
        json.dump(scores, f)
    

# Read all the embeddings

bert_dynahate_train_embeddings = read_from_pickle(r'..\Embeddings\Numpy_Model_Embeddings\BERT\DynaHate\DynaHate_Training')
bert_dynahate_dev_embeddings = read_from_pickle(r'..\Embeddings\Numpy_Model_Embeddings\BERT\DynaHate\DynaHate_Val')
bert_dynahate_test_embeddings = read_from_pickle(r'..\Embeddings\Numpy_Model_Embeddings\BERT\DynaHate\DynaHate_Test')

bert_latenthatred_train_embeddings = read_from_pickle(r'..\Embeddings\Numpy_Model_Embeddings\BERT\Latent_Hatred\LatentHatred_Training')
bert_latenthatred_dev_embeddings = read_from_pickle(r'..\Embeddings\Numpy_Model_Embeddings\BERT\Latent_Hatred\LatentHatred_Val')
bert_latenthatred_test_embeddings = read_from_pickle(r'..\Embeddings\Numpy_Model_Embeddings\BERT\Latent_Hatred\LatentHatred_Test')

bert_olid_train_embeddings = read_from_pickle(r'..\Embeddings\Numpy_Model_Embeddings\BERT\OLID\OLID_Training')
bert_olid_dev_embeddings = read_from_pickle(r'..\Embeddings\Numpy_Model_Embeddings\BERT\OLID\OLID_Val')
bert_olid_test_embeddings = read_from_pickle(r'..\Embeddings\Numpy_Model_Embeddings\BERT\OLID\OLID_Test')

bertweet_dynahate_train_embeddings = read_from_pickle(r'..\Embeddings\Numpy_Model_Embeddings\BERTweet\DynaHate\DynaHate_Training')
bertweet_dynahate_dev_embeddings = read_from_pickle(r'..\Embeddings\Numpy_Model_Embeddings\BERTweet\DynaHate\DynaHate_Val')
bertweet_dynahate_test_embeddings = read_from_pickle(r'..\Embeddings\Numpy_Model_Embeddings\BERTweet\DynaHate\DynaHate_Test')

bertweet_latenthatred_train_embeddings = read_from_pickle(r'..\Embeddings\Numpy_Model_Embeddings\BERTweet\Latent_Hatred\LatentHatred_Training')
bertweet_latenthatred_dev_embeddings = read_from_pickle(r'..\Embeddings\Numpy_Model_Embeddings\BERTweet\Latent_Hatred\LatentHatred_Val')
bertweet_latenthatred_test_embeddings = read_from_pickle(r'..\Embeddings\Numpy_Model_Embeddings\BERTweet\Latent_Hatred\LatentHatred_Test')

bertweet_olid_train_embeddings = read_from_pickle(r'..\Embeddings\Numpy_Model_Embeddings\BERTweet\OLID\OLID_Training')
bertweet_olid_dev_embeddings = read_from_pickle(r'..\Embeddings\Numpy_Model_Embeddings\BERTweet\OLID\OLID_Val')
bertweet_olid_test_embeddings = read_from_pickle(r'..\Embeddings\Numpy_Model_Embeddings\BERTweet\OLID\OLID_Test')

hatebert_dynahate_train_embeddings = read_from_pickle(r'..\Embeddings\Numpy_Model_Embeddings\HateBERT\DynaHate\DynaHate_Training')
hatebert_dynahate_dev_embeddings = read_from_pickle(r'..\Embeddings\Numpy_Model_Embeddings\HateBERT\DynaHate\DynaHate_Val')
hatebert_dynahate_test_embeddings = read_from_pickle(r'..\Embeddings\Numpy_Model_Embeddings\HateBERT\DynaHate\DynaHate_Test')

hatebert_latenthatred_train_embeddings = read_from_pickle(r'..\Embeddings\Numpy_Model_Embeddings\HateBERT\Latent_Hatred\LatentHatred_Training')
hatebert_latenthatred_dev_embeddings = read_from_pickle(r'..\Embeddings\Numpy_Model_Embeddings\HateBERT\Latent_Hatred\LatentHatred_Val')
hatebert_latenthatred_test_embeddings = read_from_pickle(r'..\Embeddings\Numpy_Model_Embeddings\HateBERT\Latent_Hatred\LatentHatred_Test')

hatebert_olid_train_embeddings = read_from_pickle(r'..\Embeddings\Numpy_Model_Embeddings\HateBERT\OLID\OLID_Training')
hatebert_olid_dev_embeddings = read_from_pickle(r'..\Embeddings\Numpy_Model_Embeddings\HateBERT\OLID\OLID_Val')
hatebert_olid_test_embeddings = read_from_pickle(r'..\Embeddings\Numpy_Model_Embeddings\HateBERT\OLID\OLID_Test')