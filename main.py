from transformers import BertForSequenceClassification, AdamW, BertConfig, get_linear_schedule_with_warmup
from transformers import BertTokenizer
import time
import datetime
import tensorflow as tf
import random
from settings import *
from prepro.data_loading import *
from prepro.reading_files import *
from train import *
from network_architecture import model_bert
from test import *
from utils import *
import argparse, torch 
import numpy as np
import pandas as pd



timestamp = time.strftime("%d-%m-%Y_%H:%M")


class config_BERT():
	patience = 2
	delta = 0.02
	epochs = 2
	max_length = 20
	padding = 'post'
	truncation = 'post'
	test_size=0.1
	batch_size=4
	batch_size_test = None
	num_labels = 3
	cutoff = 0.7
	tokenizer_name="BertTokenizer"
	model_name  = 'bert-base-uncased'
	lr = 5e-5
	eps = 1e-8

class config_file():
	question_type=['yesno']
	category_type=['Home_and_Kitchen']
	column_list=['asin','questionType','category','questionText','review_snippets','answers','is_answerable']


def set_seed(seed_value=42):
	"""Set seed for reproducibility."""
	random.seed(seed_value)
	np.random.seed(seed_value)
	torch.manual_seed(seed_value)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed_value)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("-path_to_data", default='Data', type=str, help="path to data folder")
	parser.add_argument("-save_result_path", default="Expt_results/result_" + timestamp + ".csv",
						help="Path to save results on test")
	parser.add_argument("-path_to_ckpt", default='Expt_results/checkpoint_' + timestamp + '.pt',
						help="Path to where checkpoints will be stored")

	parser.add_argument("-filetype", default='csv', help="filetype")


	parser.add_argument("-name_train", default='train_sample .csv', help="Name of train file")
	parser.add_argument("-name_val", default='val_sample.csv', help="Name of val file or None: Train will be splitted")
	parser.add_argument("-name_test", default='test_sample .csv', type=str, help="Name of test file")

	parser.add_argument("-to_preprocess_train", default=True, type=bool,help="Boolean to preprocess train and val data")
	parser.add_argument("-to_preprocess_test", default=True, type=bool, help="Boolean to preprocess test data")
	parser.add_argument("-to_preprocess_val", default=True, type=bool, help="Boolean to preprocess val data")

	parser.add_argument("-to_train_split", default=False, type=bool, help="Boolean to split train for train and val")


	parser.add_argument("-mode", default='train_&_test', type=str, choices=['train_&_test', 'only_test'])
	args = parser.parse_args()

	set_seed(42)

	test_path = args.path_to_data + '/' + args.name_test
	config = config_BERT()
	config_2 = config_file()
	filetype = args.filetype


	print('Loading BERT tokenizer...')
	tokenizer = BertTokenizer.from_pretrained(config.model_name, do_lower_case=True)


	if args.mode == "train_&_test":

		train_path = args.path_to_data + '/' + args.name_train

		if filetype == 'jsonl':
			df_train = load_jsonl(train_path,config_2.question_type,config_2.category_type,config_2.column_list)
		elif filetype == 'csv':
			df_train = read_csv_cols(train_path,config_2.question_type,config_2.category_type,config_2.column_list)

		# df_train = drop_null(df_train)
		# df_train , _ = assign_class(df_train,config.cutoff,col1="answers",col2="is_answerable")

		if args.to_preprocess_train:
			preprocessed_filepath = args.path_to_data + '/' + 'preprocessed_' + args.name_train 
			df_train = run_preprocess(df_train,'train',config.cutoff,preprocessed_filepath)

		print("Value counts in train")
		print(df_train.shape)

		sample_ratio_list = [2,2,1]
		labels_list = [1,2,0]
		df_train = sample_from_class(df_train,sample_ratio_list,labels_list)
		print("Creating Data loaders for Train : \n")
		sentencesA = df_train['questionText'].values
		sentencesB = df_train['review_snippets_total'].values
		labels_tr = df_train['label'].values

		dl = Preprocess_dataloading_bert(sentencesA,sentencesB,labels_tr)
		input_ids_tr , attention_ids_tr , sent_ids_tr = dl.tokenize(tokenizer,config.tokenizer_name,config)

		if args.to_train_split :
			train , val = dl.train_test_split_dataloading(input_ids_tr, attention_ids_tr, sent_ids_tr, config.test_size)

		else : 
			val_path = args.path_to_data + '/' + args.name_val
			if filetype == 'jsonl':
				df_val = load_jsonl(val_path,config_2.question_type,config_2.category_type,config_2.column_list)
			elif filetype == 'csv':
				df_val = read_csv_cols(val_path,config_2.question_type,config_2.category_type,config_2.column_list)

			# df_val = drop_null(df_val)
			# df_val , _ = assign_class(df_val,config.cutoff,col1="answers",col2="is_answerable")
			
			if args.to_preprocess_train:
				preprocessed_filepath = args.path_to_data + '/' + 'preprocessed_' + args.name_val 
				df_val = run_preprocess(df_val,'Validation', config.cutoff,preprocessed_filepath)

			print("Creating Data loaders for Validation : \n")
			sentencesA = df_val['questionText'].values
			sentencesB = df_val['review_snippets_total'].values
			labels_val = df_val['label'].values

			dl = Preprocess_dataloading_bert(sentencesA,sentencesB,labels_val)
			input_ids_val , attention_ids_val , sent_ids_val = dl.tokenize(tokenizer,config.tokenizer_name,config)


			train = [torch.tensor(input_ids_tr),torch.tensor(attention_ids_tr),torch.tensor(labels_tr),torch.tensor(sent_ids_tr)]
			val = [torch.tensor(input_ids_val),torch.tensor(attention_ids_val),torch.tensor(labels_val),torch.tensor(sent_ids_val)]

		del df_train, df_val, input_ids_val , attention_ids_val , sent_ids_val, labels_val, labels_tr, input_ids_tr , attention_ids_tr , sent_ids_tr

		train_loader = dl.dataloading('train',config.batch_size, train[0],train[1],train[2],train[3])
		validation_dataloader = dl.dataloading('validation',config.batch_size, val[0],val[1],val[2],val[3]) 

		total_steps = len(train_loader) * config.epochs
		loss_values, val_accuracy, model = train_model_bert(train_loader,validation_dataloader,config,total_steps,args.path_to_ckpt)

		figname = [args.save_result_path+'train_loss_'+timestamp+'.png' ,args.save_result_path+'val_acc_'+timestamp+'.png' ]
		plot_results(loss_values,val_accuracy,figname)


		#Test:
		test_path = args.path_to_data + '/' + args.name_test

		if filetype == 'jsonl':
			df_test = load_jsonl(test_path,config_2.question_type,config_2.category_type,config_2.column_list)
		elif filetype == 'csv':
			df_test = read_csv_cols(test_path,config_2.question_type,config_2.category_type,config_2.column_list)

		# df_test , _ = assign_class(df_train,config.cutoff,col1="answers",col2="is_answerable")

		if args.to_preprocess_test:
			preprocessed_filepath = args.path_to_data + '/' + 'preprocessed_' + args.name_test 
			df_test = run_preprocess(df_test,'test',config.cutoff,preprocessed_filepath)

		print("Creating Data loaders for Test : \n")
		sentencesA = df_test['questionText'].values
		sentencesB = df_test['review_snippets_total'].values
		labels_ts = df_test['label'].values

		dl_test = Preprocess_dataloading_bert(sentencesA,sentencesB,labels_ts)
		input_ids_test , attention_ids_test , sent_ids_test = dl_test.tokenize(tokenizer,config.tokenizer_name,config)

		test = [torch.tensor(input_ids_test),torch.tensor(attention_ids_test),torch.tensor(labels_ts),torch.tensor(sent_ids_test)]

		if config.batch_size_test == None:
			config.batch_size_test = len(df_test)

		test_loader = dl_test.dataloading('test',config.batch_size_test, test[0],test[1],test[2],test[3])

		y_pred, y_true = model.evaluate_bert(test_loader,'Test')
		title = 'Results ' + str(timestamp) 
		print_classification_report(y_pred, y_true, title, target_names=['no_answerable','unanswerable', 'yes_answerable'],
								save_result_path=args.save_result_path)

		print("Saving model...")
		torch.save(model.model.state_dict(), args.path_to_ckpt)
		# model.save_pretrained('./my_mrpc_model/')

	elif args.mode == "only_test":
		pass

