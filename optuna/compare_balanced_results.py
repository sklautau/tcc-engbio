'''
Finds all files trainHistoryDict.pickle, which store the history, including data from training and test stages.
Compares curves
'''
import os
import pickle
import argparse
import matplotlib.pyplot as plt
import numpy as np

# use to disable steps
run_step = [False, True, True, True]
# run_step = [False, False, False, True]

def get_roc(history):
    fpr = history['roc_fpr']
    tpr = history['roc_tpr']
    #thresholds = history['roc_thresholds']
    auc = history['auc']
    return fpr, tpr, auc

def get_precision_recall_curve(history):
    if not 'pr_precision' in history:
        #skip because there is no info about PR curve in history
        return list(), list()
    pr_precision = history['pr_precision']
    pr_recall = history['pr_recall']
    # pr_thresholds = history['pr_thresholds']
    return pr_precision, pr_recall

def find_pkl_files(folder):
    pkl_files = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith("trainHistoryDict.pickle"):
                pkl_files.append(os.path.join(root, file))
    return pkl_files

def curve_metric_over_num_examples(root_folder, metric):
    pkl_files = find_pkl_files(root_folder)
    
    abscissa = list()
    ordinate = list()
    counter = 0
    for file in pkl_files:
        #print(file)
        counter += 1
        with open(file, "rb") as file_pi:
            history = pickle.load(file_pi)

            if not 'num_desired_train_examples' in history:
                N = counter
            else:
                N = int(history['num_desired_train_examples'])

            if False: # not needed anymore
                # obtain number of examples from folder name
                tokens = file.split("tr_ex_")
                tokens = tokens[1].split("_")
                N2 = int(tokens[0])
                print("achei = ", N2)            
                if N != N2:
                    raise Exception("N != N2")
            
            if not metric in history:
                return None, None
            y = history[metric]
            if not np.isscalar(y): #(len(y)) > 1:
                #for metrics such as 'val_accuracy', we have a series of numbers and will get the last one (from last model and epoch)
                y = y[-1]
            abscissa.append(N) 
            ordinate.append(y) #this is a single number
    return abscissa, ordinate

def curve_metric_over_epochs(root_folder, metric):
    pkl_files = find_pkl_files(root_folder)
        
    for file in pkl_files:
        #print(file)
        this_path = os.path.dirname(file)
        with open(file, "rb") as file_pi:
            history = pickle.load(file_pi)
            y = history[metric]
            x = 1 + np.arange(len(y))                        
        plt.close("all")
        create_plot(x, y, metric, 'epoch #', id, uselog=False)
        output_file_name = os.path.join(this_path, metric + '.png')
        plt.savefig(output_file_name)
        print("Wrote", output_file_name)
                
def curve_compare_metric_different_models(root_folder, metric, minimum_N=0):
    pkl_files = find_pkl_files(root_folder)
    
    minimum_N = minimum_N
    x = list()
    y = list()
    train_examples_numbers = list()
    plt.close("all")       
    counter = 0 
    for file in pkl_files:
        counter += 1
        #print(file)
        #this_path = os.path.dirname(file)
        with open(file, "rb") as file_pi:
            history = pickle.load(file_pi)
            ordinate = history[metric]
            abscissa = 1 + np.arange(len(ordinate))
            if not 'num_desired_train_examples' in history:
                N = minimum_N + counter
            else:
                N = int(history['num_desired_train_examples'])
            if N >= minimum_N:
                y.append(ordinate)
                x.append(abscissa)
                train_examples_numbers.append(N)
                plt.plot(abscissa, ordinate)
    plt.ylabel(metric)
    plt.xlabel('epochs')
    plt.legend(train_examples_numbers)
    output_file_name = os.path.join(root_folder, 'models_' + metric + '.png')
    plt.savefig(output_file_name)
    print("Wrote", output_file_name)

def curve_compare_metric_different_ids(root_folder, metric, desired_N=648):
    pkl_files = find_pkl_files(root_folder)
    
    desired_N = desired_N # choose the values of number of training examples that will be used
    x = list()
    y = list()
    ids = list()
    legend_entries = list()
    plt.close("all")        
    counter = 0
    for file in pkl_files:
        #print(file)
        counter += 1
        this_path = os.path.dirname(file)
        id = os.path.split(this_path)[0].split('id_')[1]
        with open(file, "rb") as file_pi:
            history = pickle.load(file_pi)
            if not 'num_desired_train_examples' in history:
                N = counter
            else:
                N = int(history['num_desired_train_examples'])
            if N == desired_N:
                if metric == 'roc':
                    abscissa, ordinate, auc = get_roc(history)
                    legend_entries.append('id_' + id + ' AUC= %0.3f' % float(auc))
                elif metric == 'pr':
                    ordinate, abscissa = get_precision_recall_curve(history)
                    if len(ordinate) < 1:
                        continue #skip because there is no info about PR curve in history                    
                    legend_entries.append('id_' + id)
                else:
                    ordinate = history[metric]
                    abscissa = 1 + np.arange(len(ordinate))                
                    legend_entries.append('id_' + id)
                ids.append('id_' + id)
                y.append(ordinate)
                x.append(abscissa)
                plt.plot(abscissa, ordinate)
    if metric == 'roc':
        plt.title('Receiver Operating Characteristic for N=' + str(desired_N) + ' examples')
        plt.plot([0, 1], [0, 1],'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
    elif metric == 'pr':
        #plt.fill_between(recall, precision)
        plt.ylabel("Precision")
        plt.xlabel("Recall")
        plt.title("Train Precision-Recall curve for N=" + str(desired_N) + ' examples')
    else:
        plt.ylabel(metric)
        plt.xlabel('epochs')
        plt.title("for N=" + str(desired_N) + ' examples')
    plt.legend(legend_entries)
    #plt.legend(ids, fontsize="x-large")
    #plt.legend(loc = 'lower right')

    all_ids = '_'.join(ids)
    all_ids = all_ids.replace('id_','_')
    all_ids = all_ids.replace('__','_')
    #print(all_ids)

    output_file_name = os.path.join(root_folder, 'ids' + all_ids + '_' + metric + '_' + str(desired_N) + '.png')
    plt.savefig(output_file_name)
    #plt.show()
    print("Wrote", output_file_name)
    
def create_plot(x, y, metric, xlabel, id, uselog=False):
    x = np.array(x)
    y = np.array(y)
    sorting_indices = np.argsort(x)
    #print(sorting_indices)
    #print(x[sorting_indices])
    #print(y[sorting_indices])
    if uselog:
        plt.semilogx(x[sorting_indices], y[sorting_indices], '-x')
    else:
        plt.plot(x[sorting_indices], y[sorting_indices], '-x')
    plt.xlabel(xlabel)
    plt.ylabel(metric)
    plt.title('ID ' + id)
    #plt.legend(["trained with AUC","trained with Accuracy"])    
    #plt.show()
    
def get_folder_for_given_id(folder, id):    
    id_string = 'id_' + str(id)
    all_folders = [x[0] for x in os.walk(folder)]
    for folder in all_folders:
        if id_string in folder:
            parent_folder = folder.split(id_string)[0]
            parent_folder = os.path.join(parent_folder, id_string)
            return parent_folder 

if __name__ == '__main__':

    print("=====================================")
    print("Collect results")

    parser = argparse.ArgumentParser()
    #required arguments    
    parser.add_argument('--root_folder', help='Parent folder', required=True)
    parser.add_argument('--ids', help='List of ids separated by commas and without blank spaces among them. Example: 1,4,67', required=False)

    args = parser.parse_args()
    root_folder = args.root_folder
    if args.ids:
        ids = args.ids.split(',')
                        
    #Process metrics that correspond to a curve over epochs
    #In this case, the abscissa is the epoch number
    #The folder is the actual model folder. Example:
    #D:\sofia\ufpa\tcc\outputs\balanced\id_2\effnetb1_no_augment_testeC2_50-50_tr_ex_250_output
    #Training stage already generates such curves (integrated in one)
    # recover all history's
    if run_step[0]:
        print("############### Step 0 ###############")
        curve_metrics = ['loss', 'val_loss', 'accuracy', 'val_accuracy']
        for metric in curve_metrics:
            for id in ids:
                print("Processing ID:", id,"for metric", metric)
                this_path = os.path.join(root_folder , "id_" + id)
                #this_path = get_folder_for_given_id(root_folder, id)          
                curve_metric_over_epochs(this_path, metric)

    #Compare models of same ID trained with different number of examples for metrics using curves over epochs
    #In this case, the abscissa is the epoch number
    #The folder is the ID folder, that has subfolder for each num train examples. Example:
    #D:\sofia\ufpa\tcc\outputs\balanced\id_2\
    #Curves are similar to previous step, but among different models (different number of training examples)
    # recover all history's
    if run_step[1]:    
        print("############### Step 1 ###############")
        min_num_examples = 0 # if you want to eliminate curves less than 1000 examples, use min_num_examples = 1000
        models_curve_metrics = ['loss', 'val_loss', 'accuracy', 'val_accuracy']
        for metric in models_curve_metrics:
            for id in ids:
                print("Processing ID:", id,"for metric", metric)
                this_path = os.path.join(root_folder , "id_" + id)
                #this_path = get_folder_for_given_id(root_folder, id)          
                curve_compare_metric_different_models(this_path, metric, minimum_N=min_num_examples)


    #Process metrics that correspond to a single number per experiment
    #or a sequence of numbers. If it is a sequence, only the last number is used.
    #In this case, the abscissa is the number of ***training examples***. 
    #The folder is the ID folder, that has subfolder for each num train examples. Example:
    #D:\sofia\ufpa\tcc\outputs\balanced\id_2\
    # recover all history's
    if run_step[2]:
        print("############### Step 2 ###############")
        number_metrics = ['val_accuracy','val_auc','val_loss','accuracy','loss','test_accuracy', 'test_loss', 'auc', 'recall', 'f1', 'auc']
        for metric in number_metrics:
            for id in ids:
                print("Processing ID:", id,"for metric", metric)
                this_path = os.path.join(root_folder , "id_" + id)                
                #this_path = get_folder_for_given_id(root_folder, id)          
                x, y = curve_metric_over_num_examples(this_path, metric)
                if x==None and y==None:
                    print(metric, "is not in history")
                    continue
                plt.close("all")
                create_plot(x, y, metric, 'number of training examples', id, uselog=True)
                file_name = os.path.join(this_path, metric + '.png')
                plt.savefig(file_name)
                print("Wrote", file_name)


    #Compare models of different ID for largest number of examples
    #The folder is the parent of the ID folders. Example:
    #D:\sofia\ufpa\tcc\outputs\balanced\
    # recover all history's
    if run_step[3]:    
        print("############### Step 3 ###############")
        desired_N = 648 # 30000 # all models to be compared were trained with this number of training examples
        models_curve_metrics = ['loss', 'val_loss', 'accuracy', 'val_accuracy', 'roc', 'pr']
        for metric in models_curve_metrics:
            curve_compare_metric_different_ids(root_folder, metric, desired_N=desired_N)
