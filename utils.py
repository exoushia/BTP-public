import matplotlib
matplotlib.use("agg")

import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, multilabel_confusion_matrix, precision_recall_fscore_support
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import numpy as np
import pandas as pd
import argparse, time



def print_info(df,category):
    print('----------------------------------')
    print("Details for {} : \n".format(category))
    print("Number of answerable and non answerable questions : " , len(df.loc[df['is_answerable']==1]) , len(df.loc[df['is_answerable']==0]))
    print("Number of descriptive and non yesno questions : " , len(df.loc[df['questionType']=='descriptive']) , len(df.loc[df['questionType']=='yesno']))
    print("\n yesno Type: ")
    print("Number of answerable and non answerable questions in yesno type : " , len(df.loc[(df['is_answerable']==1) & (df['questionType']=='yesno')]) , len(df.loc[(df['is_answerable']==0) & (df['questionType']=='yesno')]))
    print('\n Descriptive type : ')
    print("Number of answerable and non answerable questions in Descriptive type : " , len(df.loc[(df['is_answerable']==1) & (df['questionType']=='descriptive')]) , len(df.loc[(df['is_answerable']==1) & (df['questionType']=='descriptive')]))
    print("---------------------------------")
    print("\n\n")


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

def print_classification_report(y_pred, y_true, title, target_names=['no_answerable','unanswerable', 'yes_answerable'],
								save_result_path="Expt_results/results.csv"):

	str_title = "Printing Classification Report : " + title + " \n\n"
	print(str_title)
	report = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)
	df = pd.DataFrame(report)
	df.to_csv(save_result_path)
	print(report)

	str_title = "\n\n Printing Multilabel Confusion Matrix : " + title + " \n\n"
	print(str_title)
	print(multilabel_confusion_matrix(y_true, y_pred))

	print("\n All Results Printed !! \n")

def save_object(obj, filename):
	with open(filename, 'wb') as output:  # Overwrites any existing file.
		pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


def plot_results(train_losses_plot,val_accuracies_plot,figname):
	
	plt.style.use('classic')
	fig = plt.figure(figsize=(10, 8))
	print("Starting to plot figures.... \n\n")

	ax = plt.subplot(2, 1, 1)
	ax.plot(train_losses_plot)
	ax.set(title="training set loss")
	ax.grid()
	if figname is not None:
		fig.savefig(figname[0])

	ax = plt.subplot(2, 1, 2)
	ax.plot(self.val_accuracies_plot, color='green')
	ax.set(title="validation accuracy")
	ax.grid()
	if figname is not None:
		fig.savefig(figname[1])
	plt.show()

