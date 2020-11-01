import time
import settings 
import utils
import torch

def evaluate(model,validation_dataloader):
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
        logits = outputs[0]
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

def train_model():
	# Store the average loss after each epoch so we can plot them.
	loss_values = []
	val_accuracy = []

	model = model_bert(model_name,num_labels,config,total_steps)

    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=config.patience, verbose=True, delta=config.delta, path_to_cpt=path_to_cpt)

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

    		eval_acc = evaluate(model,validation_dataloader)
    		val_accuracy.append(eval_acc)

	print("\n")
	elapsed = format_time(time.time() - start_of_training)
	print("Training complete!")
	print("Total Training time: {}".format(elapsed))






