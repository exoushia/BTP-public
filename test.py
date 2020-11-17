import time
import settings 
import utils
import torch
from settings import * 


def run_test(model,test_dataloader):
	print("\nTesting the model...")
	t0 = time.time()
	pred = []
	true = []

	model.eval_switch()
	for batch in test_dataloader:
		batch = tuple(t.to(device) for t in batch)
		b_input_ids, b_input_mask, b_labels, b_segment = batch

		with torch.no_grad():        
			outputs = model(b_input_ids, token_type_ids=b_segment, attention_mask=b_input_mask)
		
		logits = outputs.logits
		logits = logits.detach().cpu().numpy()
		y_pred = np.argmax(logits, axis=1).flatten()
		y_true = b_labels.to('cpu').numpy().flatten()
		pred = pred + y_pred
		true = true + y_true

	print("Testing took : {}".format(format_time(time.time() - t0)))
	return pred , true 



	