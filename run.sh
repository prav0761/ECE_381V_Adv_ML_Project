#!/bin/bash

#SBATCH -J train5  # Job name
#SBATCH -o output5.txt       # Name of stdout output file
#SBATCH -e error5.txt      # Name of stderr error file
#SBATCH -p p100           # Queue (partition) name
#SBATCH -N 1               # Total # of nodes (must be 1 for serial)
#SBATCH -n 1               # Total # of mpi tasks (should be 1 for serial)
#SBATCH -t 23:30:00        # Run time (hh:mm:ss)
#SBATCH --mail-type=all    # Send email at begin and end of job
#SBATCH -A Senior-Design_UT-ECE       # Project/Allocation name (req'd if you have more than 1)
#SBATCH --mail-user=praveenradhakrishnan076@utexas.edu


python3 /home1/08629/pradhakr/advml_project/main_ml.py --trial_number 5 --intra_projection_dim 128 --hidden_dim 512 --image_learning_rate 0.001 --momentum 0.9 --temperature 0.07 --weight_decay 0.0001 --optimizer_type sgd --total_epochs 50 --reconstruction_tradeoff 1 --contrastive_tradeoff 1 --batch_size 32 --image_layers_to_train layer3 layer4 --noise_amount 0.2 --flickr30k_images_dir_path '/work/08629/pradhakr/maverick2/cv_project/flickr30k-images' --flickr30k_tokens_dir_path '/work/08629/pradhakr/maverick2/cv_project/flickr30k_captions/results_20130124.token' --graph_save_dir '/home1/08629/pradhakr/advml_project/graphs' --logresults_save_dir_path '/work/08629/pradhakr/maverick2/advml_project/train_results'

