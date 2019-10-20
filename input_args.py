import argparse
def get_input_args():
    
    
    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser()
    # Create 3 command line arguments as mentioned above using add_argument() from ArguementParser method
    parser.add_argument('--image_dir',type=str,help='Image Folder path',default = 'ImageClassifier/flowers')
    parser.add_argument('--arch',type=str,help='CNN Model',default = 'vgg11')
    parser.add_argument('--save_dir',type=str,help='Save directory path',default = 'ImageClassifier')
    
    parser.add_argument('--learning_rate',type=str,help='Learning rate to be used for training',default = '.005')
    parser.add_argument('--epochs',type=str,help='Epochs to be used for training',default = '.005')
    parser.add_argument('--input_layer',type=str,help='Input layer size to be used for training',default = '25088')
    parser.add_argument('--output_layer',type=str,help='Output layer size to be used for training',default = '102')
    parser.add_argument('--hidden_layer',type=str,help='Hidden layer size to be used for training',default = '512')

    print (parser.parse_args())
get_input_args()