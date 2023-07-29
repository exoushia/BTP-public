# Measuring and ranking Question - Question relatedness to remove duplicate questions on Stackoverflow
Repository for BTP - amazonQA data

Refer to main.py 

	parser.add_argument("-path_to_data", default='Data', type=str, help="path to data folder")
	parser.add_argument("-save_result_path", default="Expt_results/result_" + timestamp + ".csv",
						help="Path to save results on test")
	parser.add_argument("-path_to_ckpt", default='Expt_results/checkpoint_' + timestamp + '.pt',
						help="Path to where checkpoints will be stored")

	parser.add_argument("-filetype", default='csv', help="filetype")


	parser.add_argument("-name_train", default='train_preprocessed.csv', help="Name of train file")
	parser.add_argument("-name_val", default='val_preprocessed.csv', help="Name of val file or None: Train will be splitted")
	parser.add_argument("-name_test", default='test_preprocessed.csv', type=str, help="Name of test file")

	parser.add_argument("-to_preprocess_train", default=False, type=bool,help="Boolean to preprocess train and val data")
	parser.add_argument("-to_preprocess_test", default=False, type=bool, help="Boolean to preprocess test data")
	parser.add_argument("-to_preprocess_val", default=False, type=bool, help="Boolean to preprocess val data")

	parser.add_argument("-to_train_split", default=False, type=bool, help="Boolean to split train for train and val")


	parser.add_argument("-mode", default='train_&_test', type=str, choices=['train_&_test', 'only_test'])

