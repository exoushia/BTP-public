import time
import torch
from utils import *
from prepro.data_loading import *
from prepro.reading_files import *
from network_architecture import model_bert
from settings import * 



def evaluate(model,validation_dataloader,filename="Validation"):
	print("\n Running {}...".format(filename))
	t0 = time.time()

	model.eval()
	eval_loss, eval_accuracy = 0, 0
	nb_eval_steps, nb_eval_examples = 0, 0

	pred = []
	true = []

	for batch in validation_dataloader:
		
		# Add batch to GPU
		batch = tuple(t.to(device) for t in batch)
		b_input_ids, b_input_mask, b_labels, b_segment = batch

		with torch.no_grad():   
			sequence_output, pooled_output = model.bert(b_input_ids, 
														token_type_ids=b_segment, 
														attention_mask=b_input_mask)
		sequence_output = sequence_output[:,0,:].view(-1,1,768) 
		_ , (lstm_output,_) = model.lstm(sequence_output)
		lstm_output = lstm_output.permute(2,0,1).reshape(b_labels.shape[0],256)
		linear_output = model.net(lstm_output)
		output = linear_output.detach().cpu().numpy()
		
		
		tmp_eval_accuracy = flat_accuracy(output, b_labels)
		eval_accuracy += tmp_eval_accuracy

		# Track the number of batches
		nb_eval_steps += 1

		if filename == 'Test':
			y_pred = np.argmax(output, axis=1).flatten()
			y_true = b_labels.to('cpu').numpy().flatten()
			pred.extend(y_pred)
			true.extend(y_true)

	# Report the final accuracy for this validation run.
	print("  {} Accuracy: {}".format(filename,eval_accuracy/nb_eval_steps))
	print("  {} took: {}".format(filename,format_time(time.time() - t0)))

	if filename == "Validation":
		return eval_accuracy/nb_eval_steps
	else: 
		return pred,true


def evaluate2(model,validation_dataloader,filename="Validation"):
	print("\n Running {}...".format(filename))
	t0 = time.time()

	model.eval()
	eval_loss, eval_accuracy = 0, 0
	nb_eval_steps, nb_eval_examples = 0, 0

	pred = []
	true = []

	for batch in validation_dataloader:
		
		# Add batch to GPU
		batch = tuple(t.to(device) for t in batch)
		b_input_ids, b_input_mask, b_labels, b_segment = batch

		with torch.no_grad():   
			sequence_output, pooled_output, hidden_states = model.bert(b_input_ids, 
														token_type_ids=b_segment, 
														attention_mask=b_input_mask)
		print("hidden states")
		print(hidden_states[0].shape)
		print(hidden_states[1].shape)

		h12 = hidden_states[-1][:, 0].reshape((-1, 1, 768))
		h11 = hidden_states[-2][:, 0].reshape((-1, 1, 768))
		h10 = hidden_states[-3][:, 0].reshape((-1, 1, 768))
		h9  = hidden_states[-4][:, 0].reshape((-1, 1, 768))

		all_h = torch.cat([h9, h10, h11, h12], 1) 
		mean_pool = torch.mean(all_h, 1)

		print("shape of mean pool : {}".format(mean_pool.shape))

		linear_output = model.net(mean_pool)
		output = linear_output.detach().cpu().numpy()
		
		
		tmp_eval_accuracy = flat_accuracy(output, b_labels)
		eval_accuracy += tmp_eval_accuracy

		# Track the number of batches
		nb_eval_steps += 1

		if filename == 'Test':
			y_pred = np.argmax(output, axis=1).flatten()
			y_true = b_labels.to('cpu').numpy().flatten()
			pred.extend(y_pred)
			true.extend(y_true)

	# Report the final accuracy for this validation run.
	print("  {} Accuracy: {}".format(filename,eval_accuracy/nb_eval_steps))
	print("  {} took: {}".format(filename,format_time(time.time() - t0)))

	if filename == "Validation":
		return eval_accuracy/nb_eval_steps
	else: 
		return pred,true


	
