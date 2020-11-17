import time
import torch
from utils import *
from prepro.data_loading import *
from prepro.reading_files import *
from network_architecture import model_bert
from settings import * 



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



