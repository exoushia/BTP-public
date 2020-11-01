import numpy as np
import pandas as pd


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

def quantify_info(df,to_save=True,filepath="results/InfoTable.csv")
	dict_columns = {'category':[],'#yesno':[],'#descriptive':[],'#non_answerable':[],'#answerable':[],'#yesno-answerable':[],'#descriptive-answerable':[],'#yesno-non_answerable':[],'#descriptive-non_answerable':[]}
	category = list(set(df['category']))

	for cat in category:
	    temp = df.loc[df['category']==cat]
	    dict_columns['category'].append(cat)
	    dict_columns['#yesno'].append(len(temp.loc[temp['questionType']=='yesno']))
	    dict_columns['#descriptive'].append(len(temp.loc[temp['questionType']=='descriptive']))
	    dict_columns['#non_answerable'].append(len(temp.loc[temp['is_answerable']==0]))
	    dict_columns['#answerable'].append(len(temp.loc[temp['is_answerable']==1]))
	    dict_columns['#yesno-answerable'].append(len(temp.loc[(temp['is_answerable']==1) & (temp['questionType']=='yesno')]))
	    dict_columns['#descriptive-answerable'].append(len(temp.loc[(temp['is_answerable']==1) & (temp['questionType']=='descriptive')]))
	    dict_columns['#yesno-non_answerable'].append(len(temp.loc[(temp['is_answerable']==0) & (temp['questionType']=='yesno')]))
	    dict_columns['#descriptive-non_answerable'].append(len(temp.loc[(temp['is_answerable']==0) & (temp['questionType']=='descriptive')]))
	del temp

	df_table = pd.DataFrame(dict_columns)
	del temp

	if to_save:
		df_table.to_csv(filepath,index=None)
		
	return df_table

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
