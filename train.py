from input_args import get_input_args








args = get_input_args()
image_dir = args.image_dir
checkpoint_dir = args.save_dir
train_model = args.arch
train_lr = args.learning_rate
train_epochs = args.epochs
train_input = args.input_layer
train_output = args.output_layer
train_hidden = args.hidden_layer
