import json
import pickle
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

def load_jsonl(input_path,question_type,category_type,column_list=['asin','category','questionText','review_snippets','label']) -> list:
    """
    Read list of objects from a JSON lines file.
    """
    data = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.rstrip('\n|\r')))
    print('Loaded {} records from {}'.format(len(data), input_path))
    df = pd.DataFrame(data)
    df = df[column_list]
    if question_type!=None:
    	df= df.loc[df['questionType'].isin(question_type)]
    if category_type != None:
    	df = df.loc[df['category'].isin(category_type)]
    del data
    return df

def read_csv_cols(input_path,question_type,category_type,column_list=['asin','category','questionText','review_snippets','label']):
    df = pd.read_csv(input_path)
    df = df[column_lists]
    if question_type!=None:
    	df= df.loc[df['questionType'].isin(question_type)]
    if category_type != "all" :
    	df = df.loc[df['category'].isin(category_type)]
    return df

def drop_null(df):

    df = df.loc[~df.questionText.isnull()]
    df = df.loc[~df.review_snippets_total.isnull()]

    print("\n Final distribution of classes after removing null values: " )
    print(df[["label"]].value_counts())
    return df

def target_class(answers):
    class1 = "yes_answerable"
    class2 = "no_answerable"
    class4 = "yes_no_equal"

    yes=0
    no=0
    NA = 0
    for dict_ in answers:
        if dict_['answerType'] == 'N' : 
            no = no+1
        elif dict_['answerType'] == 'Y' : 
            yes = yes+1
        elif dict_['answerType'] == 'NA' :
            NA = NA+1 
    if yes>no : 
        num_yes = num_yes+1
        return str(yes)+"#"+str(no)+"#"+str(NA)+"#"+class1
    elif no>yes:
        num_no = num_no+1
        return str(yes)+"#"+str(no)+"#"+str(NA)+"#"+class2
    else:
        no_class4 = no_class4+1
        return str(yes)+"#"+str(no)+"#"+str(NA)+"#"+class4

def assign_class(df,cutoff,col1="answers",col2="is_answerable"):
	classes = []
	labels = []
	dict_num_classes = {}
	list_keywords = ['no_class4_train' , 'num_yes_train' , 'num_no_train' , 'num_unans_train']

	no_class4 = 0
	num_yes = 0
	num_no = 0
	num_unans=0

	for i in range(len(df)):
		if df.iloc[i][col2] == 0:
			classes.append("unanswerable")
			labels.append("unanswerable")
			continue
		else:
			classes.append(target_class(df.iloc[i][col1]))
			#labels.append(target_class(df.iloc[i][col1]).split("#")[3])
			yes = target_class(df.iloc[i][col1]).split('#')[0]
			no = target_class(df.iloc[i][col1]).split('#')[1]
			if target_class(df.iloc[i][col1]).split("#")[3] == "yes_answerable" and yes!='0' : 
				yes = int(yes)/(int(yes)+int(no))
				no = int(no)/(int(yes)+int(no))
				if yes>=cutoff: labels.append("yes_answerable")
				else: labels.append("yes_no_equal")
			elif target_class(df.iloc[i][col1]).split("#")[3] == "no_answerable" and no!='0' :
				yes = int(yes)/(int(yes)+int(no))
				no = int(no)/(int(yes)+int(no))
				if no>=cutoff: labels.append("no_answerable")
				else: labels.append("yes_no_equal")
			else:
				labels.append("yes_no_equal")
			
	
	for i in range(len(labels)):
		if labels[i] == 'unanswerable' : num_unans = num_unans + 1
		elif labels[i] == "yes_answerable" : num_yes = num_yes + 1
		elif labels[i] == "yes_no_equal" : no_class4 = no_class4 + 1
		elif labels[i] == "no_answerable" : num_no = num_no + 1

	df["target"] = classes
	df["label"] = labels
	   
	dict_num_classes[list_keywords[0]] = no_class4
	dict_num_classes[list_keywords[1]] = num_yes
	dict_num_classes[list_keywords[2]] = num_no
	dict_num_classes[list_keywords[3]] = num_unans

	print("# of observations in data set with equal yes and no as answertype [for answerable] : {} out of total {} obs".format(dict_num_classes[list_keywords[0]],len(df)))
	print("# of observations in data set with label = yes : {}".format(dict_num_classes[list_keywords[1]]))
	print("# of observations in data set with label = no : {}".format(dict_num_classes[list_keywords[2]]))
	print("# of observations in data set with label = unanswerable : {}".format(dict_num_classes[list_keywords[3]]))

	df = df.loc[df["label"]!="yes_no_equal"]
	del labels, classes
	return df , dict_num_classes

def encode_label(df,label_column):
    le = LabelEncoder()
    df[label_column] = le.fit_transform(df[label_column])
    print(dict(zip(le.classes_, le.transform(le.classes_))))
    return df , le 

# def encode_label_test(df,label_column,label_encoder):
#     df[label_column] = label_encoder.transform(df[label_column])
#     print("Should be similar to train : \n")
#     print(dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))))
#     return df 

def drop_column(df,colname_list):
    df = df.drop(colname_list,axis=1)
    return df

def sample_from_class(df,sample_ratio_list,label):
    df = pd.concat([ df.loc[df['label']==label[0]].sample(sample_ratio_list[0]) , df.loc[df['label']==label[1]].sample(sample_ratio_list[1]) , df.loc[df['label']==label[2]].sample(sample_ratio_list[2])]).sample(frac=1) 
    return df

