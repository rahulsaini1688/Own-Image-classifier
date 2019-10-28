import argparse
def get_input_args():
    
    
    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--data_dir',type=str,help='Image Folder path',default = 'ImageClassifier/flowers')
    parser.add_argument('--arch',type=str,help='CNN Model',default = 'vgg11')
    parser.add_argument('--save_dir',type=str,help='Save directory path',default = 'ImageClassifier')
    
    parser.add_argument('--learning_rate',type=float,help='Learning rate to be used for training',default = '0.005')
    parser.add_argument('--epochs',type=int,help='Epochs to be used for training',default = '5')
    
    
    parser.add_argument('--hidden_layer',type=int,help='Hidden layer size to be used for training',default = '512')
    parser.add_argument('--gpu',type=str,help='Use GPU or CPU',default = 'gpu')
    return parser.parse_args()
    

