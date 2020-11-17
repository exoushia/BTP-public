import torch
from transformers import BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
import tensorflow as tf
import time
from settings import device
from utils import *
import datetime

class model_bert:
	def __init__(self,model_name,num_labels,config,total_steps):
		#Instantiating the model
		self.config = config
		self.model = BertForSequenceClassification.from_pretrained(
		    model_name, 
		    num_labels = num_labels, 
		    output_attentions = False, 
		    output_hidden_states = False, 
		    return_dict=True

		)

		self.optimizer = AdamW(self.model.parameters(),
                  lr = config.lr ,
                  eps = config.eps
                )

		self.scheduler = get_linear_schedule_with_warmup(self.optimizer, 
                                            num_warmup_steps = 0, 
                                            num_training_steps = total_steps)


		# Telling pytorch to run this model on the GPU.
		if torch.cuda.is_available():
			self.model.cuda()

		params = list(self.model.named_parameters())
		print('The BERT model has {:} different named parameters.\n'.format(len(params)))
		print('==== Embedding Layer ====\n')
		for p in params[0:5]:
		    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))
		print('\n==== First Transformer ====\n')
		for p in params[5:21]:
		    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))
		print('\n==== Output Layer ====\n')
		for p in params[-4:]:
		    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

	def eval_switch(self):
		self.model.eval()

	def forward(self,epoch_i,train_loader):
		# ========================================
	    #               Training
	    # ========================================
	    
	    # Performing one full pass over the training set.
	    print("")
	    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1,self.config.epochs))
	    print('Training...')

	    t0 = time.time()

	    # Reset the total loss for this epoch.
	    total_loss = 0
	    self.model.train()
	    # For each batch of training data...
	    for step, batch in enumerate(train_loader):
	        # Progress update every 40 batches.
	        if step % 10 == 0 and not step == 0:
	            # Calculate elapsed time in minutes.
	            elapsed = format_time(time.time() - t0)
	            # Report progress.
	            print('Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_loader), elapsed))

	        # As we unpack the batch, we'll also copy each tensor to the GPU using the 
	        # `to` method.
	        #
	        # `batch` contains three pytorch tensors:
	        #   [0]: input ids 
	        #   [1]: attention masks
	        #   [2]: labels 
	        b_input_ids = batch[0].to(device)
	        b_input_mask = batch[1].to(device)
	        b_labels = batch[2].to(device)
	        b_segment = batch[3].to(device)

	        self.model.zero_grad()        
	        outputs = self.model(b_input_ids, 
	                    token_type_ids=b_segment, 
	                    attention_mask=b_input_mask, 
	                    labels=b_labels)
	        
	        # The call to `model` always returns a tuple, so we need to pull the 
	        # loss value out of the tuple.
	        loss = outputs.loss
	        total_loss += loss.item()
	        loss.backward()

	        # Clip the norm of the gradients to 1.0.
	        # This is to help prevent the "exploding gradients" problem.
	        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

	        self.optimizer.step()
	        self.scheduler.step()

	    # Calculate the average loss over the training data.
	    avg_train_loss = total_loss / len(train_loader)            

	    print("\n")
	    print("  Average training loss: {0:.2f}".format(avg_train_loss))
	    print("  Training epcoh took: {:}".format(format_time(time.time() - t0)))

	    return avg_train_loss


	def evaluate_bert(self,validation_dataloader,filename='Validation'):
		print("\n Running {}...".format(filename))
		t0 = time.time()

		self.eval_switch()
		eval_loss, eval_accuracy = 0, 0
		nb_eval_steps, nb_eval_examples = 0, 0

		pred = []
		true = []

		for batch in validation_dataloader:
			
			# Add batch to GPU
			batch = tuple(t.to(device) for t in batch)
			b_input_ids, b_input_mask, b_labels, b_segment = batch

			with torch.no_grad():        
				outputs = self.model(b_input_ids, token_type_ids=b_segment, attention_mask=b_input_mask)
			
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

			if filename == 'Test':
				y_pred = np.argmax(logits, axis=1).flatten()
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






