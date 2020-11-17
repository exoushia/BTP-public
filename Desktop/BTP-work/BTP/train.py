import time
import torch
import utils
from prepro.data_loading import *
from prepro.reading_files import *
import settings 


def evaluate_bert(model,validation_dataloader,filename='Validation'):
	print("\n Running Validation...")
	t0 = time.time()

	model.eval()
	eval_loss, eval_accuracy = 0, 0
	nb_eval_steps, nb_eval_examples = 0, 0

	for batch in validation_dataloader:
		
		# Add batch to GPU
		batch = tuple(t.to(device) for t in batch)
		b_input_ids, b_input_mask, b_labels, b_segment = batch

		with torch.no_grad():        
			outputs = model(b_input_ids, token_type_ids=b_segment, attention_mask=b_input_mask)
		
		# Get the "logits" output by the model. The "logits" are the output
		# values prior to applying an activation function like the softmax.
		logits = outputs.logits
		# Move logits and labels to CPU
		logits = logits.detach().cpu().numpy()
		label_ids = b_labels.to('cpu').numpy()
		
		# Calculate the accuracy for this batch of test sentences.
		tmp_eval_accuracy = flat_accuracy(logits, label_ids)
		
		# Accumulate the total accuracy.
		eval_accuracy += tmp_eval_accuracy
		# Track the number of batches
		nb_eval_steps += 1

	# Report the final accuracy for this validation run.
	print("  Validation Accuracy: {0:.2f}".format(eval_accuracy/nb_eval_steps))
	print("  Validation took: {:}".format(format_time(time.time() - t0)))

	return eval_accuracy/nb_eval_steps

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

	model = model_bert(model_name,num_labels,config,total_steps)

	# initialize the early_stopping object
	# early_stopping = EarlyStopping(patience=config.patience, verbose=True, delta=config.delta, path_to_cpt=path_to_cpt)

	if torch.cuda.is_available():
		model.to(device)

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

			eval_acc = evaluate_bert(model,validation_dataloader)
			val_accuracy.append(eval_acc)

	print("\n")
	elapsed = format_time(time.time() - start_of_training)
	print("Training complete!")
	print("Total Training time: {}".format(elapsed))

	return loss_values , val_accuracy, model 



