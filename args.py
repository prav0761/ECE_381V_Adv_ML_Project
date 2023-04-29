
# coding: utf-8

# In[5]:


import argparse

def args_project():
    parser = argparse.ArgumentParser(description='Image denoiser Training')
    
    # Model hyperparameters
    parser.add_argument('--trial_number', type=int, default=3, help='Trial number for the experiment')
    parser.add_argument('--intra_projection_dim', type=int, default=128, help='Intra-attention projection dimension')
    parser.add_argument('--hidden_dim', type=int, default=512, help='Intra-attention projection dimension')
    parser.add_argument('--image_learning_rate', type=float, default=0.0001, help='Learning rate for image encoder')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for SGD optimizer')
    parser.add_argument('--temperature', type=float, default=0.07, help='Temperature for softmax in contrastive loss')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='Weight decay for SGD optimizer')
    parser.add_argument('--optimizer_type', type=str, default='sgd', help='Optimizer type (sgd or adam)')
    parser.add_argument('--total_epochs', type=int, default=50, help='Total number of training epochs')
    parser.add_argument('--reconstruction_tradeoff', type=float, default=1, help='Trade-off for reconstruction_tradeoff')
    parser.add_argument('--contrastive_tradeoff', type=float, default=1, help='Trade-off for contrastive_tradeoff')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--image_layers_to_train', nargs='+', default=['layer4'], help='Image encoder layers to train')
    parser.add_argument('--noise_amount', type=float, default=0.2 ,help='noise_amount')


    # Paths and directories
    parser.add_argument('--flickr30k_images_dir_path', 
                        type=str, default='/work/08629/pradhakr/maverick2/cv_project/flickr30k-images', 
                        help='Directory path for Flickr30k images')
    parser.add_argument('--flickr30k_tokens_dir_path', type=str, 
                        default='/work/08629/pradhakr/maverick2/cv_project/flickr30k_captions/results_20130124.token',
                        help='Directory path for Flickr30k captions')
    parser.add_argument('--graph_save_dir', type=str, default='/home1/08629/pradhakr/advml_project/graphs')
    parser.add_argument('--logresults_save_dir_path', type=str, default='/work/08629/pradhakr/maverick2/advml_project/train_results')
                        
                        
    args = parser.parse_args()

    return args

