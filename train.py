import time
import torch
from utils import *
from prepro.data_loading import *
from prepro.reading_files import *
from network_architecture import *
from settings import * 
from test import *
import torch.optim as optim


def run_preprocess(df,filename,cutoff,preprocessed_filepath):
	print("----------------------File Info: [{}] -------------------------\n".format(filename))

	print("-------------Assigning Label ----------------\n")
	df, dict_num_classes = assign_class(df,cutoff,"answers","is_answerable")
	print("------------------------------------------------------\n")


	print("-------------Concatenating reviews----------------\n")
	df['review_snippets_total'] = df['review_snippets'].apply(lambda x : ' '.join(x))
	df = drop_column(df,["review_snippets"])
	print("-------------------------------------------------\n")


	print("-------------Encoding Label-------------------\n")
	df , label_encoder = encode_label(df,"label")
	print("------------------------------------------------------------\n")


	print("-------------Removing null rows----- --------\n")
	df = drop_null(df)
	print("---------------------------------------------------------------")


	print("-------------INITIAL PREPROCESSING COMPLETED-----------------")
	print("-------------------------------------------------------------\n\n")


	print("-------------Text preprocessing------------------------------\n")

	preprocess = Preprocessing_text()
	df = preprocess.main_df_preprocess(df,['questionText' , 'review_snippets_total'])
	print("Saving the preprocessed file.......")
	df.to_csv(preprocessed_filepath, index=None)

	print("-------------TEXT PREPROCESSING COMPLETED-----------------")

	return df


def train_model_bert(train_loader,validation_dataloader,config,total_steps,path_to_cpt):
	# Store the average loss after each epoch so we can plot them.
	model_name = config.model_name
	num_labels = config.num_labels
	loss_values = []
	val_accuracy = []

	print("Loading model ...")

	model = model_bert(model_name,num_labels,config,total_steps)

	# initialize the early_stopping object
	# early_stopping = EarlyStopping(patience=config.patience, verbose=True, delta=config.delta, path_to_cpt=path_to_cpt)

	start_of_training = time.time()

	# For each epoch...
	for epoch_i in range(0, config.epochs):
		# ========================================
		#               Training
		# ========================================
		training_loss = model.forward(epoch_i,train_loader)
		loss_values.append(training_loss)

		if epoch_i%1==0:
			# ========================================
			#               Validation
			# ========================================

			eval_acc = model.evaluate_bert(validation_dataloader)
			val_accuracy.append(eval_acc)

	print("\n")
	elapsed = format_time(time.time() - start_of_training)
	print("Training complete!")
	print("Total Training time: {}".format(elapsed))

	return loss_values , val_accuracy, model 



def train_model_custom_bert(mode,train_loader,validation_dataloader,config,total_steps,_):

	# Store the average loss after each epoch so we can plot them.
	model_name = config.model_name
	num_labels = config.num_labels
	loss_values = []
	val_accuracy = []

	print("Loading model ...")

	if mode == 'lstm':
		model = CustomBERTModel(model_name,num_labels,config,total_steps)
	else :
		model = CustomBERTModel_hidden(model_name,num_labels,config,total_steps)

	model.to(torch.device(device)) 

	optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))

	start_of_training = time.time()

	# For each epoch...
	for epoch_i in range(0, config.epochs):
		# ========================================
		#               Training
		# ========================================
		model.train()
		optimizer.zero_grad()
		training_loss = model(epoch_i,train_loader)
		loss_values.append(training_loss)
		optimizer.step()

		if epoch_i%1==0:
			# ========================================
			#               Validation
			# ========================================
			model.eval()
			if mode=='lstm':
				eval_acc = evaluate(model,validation_dataloader)
			else : 
				eval_acc = evaluate2(model,validation_dataloader)

			val_accuracy.append(eval_acc)

	print("\n")
	elapsed = format_time(time.time() - start_of_training)
	print("Training complete!")
	print("Total Training time: {}".format(elapsed))

	return loss_values , val_accuracy, model 

