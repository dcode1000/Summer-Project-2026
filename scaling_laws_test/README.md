## scaling_laws README

This folder contains all the salt configs, a representative checkpoint file for each trained model (usually that with the lowest validation loss) and metadata and performance metrics used to train the various models needed when testing for the presence of scaling laws in the trained models. Analysis script and output is in the analysis folder. The models used are:

- fd_fm: full dataset, full model - trains on the entire dataset and uses the full model
- sd_fm: small dataset, full model - trains on a slice of the dataset and uses full model
- sd_sm: small dataset, small model - trains on a slice of the dataset using a model whose number of trainable parameters is roughly scaled down by a factor similar to that of the size of the dataset slice compared to the full model
- dd_sd_fm: double descent, small dataset, full model - small test for the presence of double descent behaviour in the model using the same config as the sd_fm setup but trained over 100 epochs instead of 40
 
Note: All data taken comes from file scaling_laws_test_2 on the cognition server. Max learning rate was tuned down 1e-4 on the fd_fm training to prevent crashes, this may alter the usefulness of the data.
