# MEnKF-ANN-PUL

## Data Description

The supervised dataset of sequences can be found at [supervised_data](https://github.com/Ved-Piyush/MEnKF-ANN-PUL/blob/main/Data_Generation/Data/supervised.csv) and the unsupervised dataset of sequences can be found at [unsupervised_data](https://github.com/Ved-Piyush/MEnKF-ANN-PUL/blob/main/Data_Generation/Data/all_unsupervised.csv). 

The train, validation, and test sequences corresponding to the 50 simulation repetitions can be found at [train_valid_test](https://github.com/Ved-Piyush/MEnKF-ANN-PUL/blob/main/Data_Generation/Data/train_valid_test_splits_50.pkl), the 50 sets of training, validation, and testing sequences are in the a pickle format generated by the code [train_valid_test_generating_code](https://github.com/Ved-Piyush/MEnKF-ANN-PUL/blob/main/Data_Generation/Scripts/Generate_Train_Valid_Test_50_Random_Samples.ipynb). 

The code to train Word2Vec and Doc2Vec embedding models using the unsupervised sequences can be found at [Word2Vec_Doc2Vec_Training](https://github.com/Ved-Piyush/MEnKF-ANN-PUL/blob/main/Data_Generation/Scripts/Train_Word2Vec_Doc2Vec.ipynb). 

## Data Generation for Simulations and Applications

In this section, we detail the codes that were used to train the True and the Fitted LSTMs for the 50 sets of training, validation, and testing sequences. 

The code to train the True LSTM (with two dropout layers) can be found at [True_LSTM_Training_Two_Dropout_Layers](https://github.com/Ved-Piyush/MEnKF-ANN-PUL/blob/main/Data_Generation/LSTM_Heavy_Dropout/Generate_Heavy_Dropout_First_LSTM.ipynb). The code to train the Fitted LSTMs (with two dropout layers) can be found at [Fitted_LSTM_Training_Two_Dropout_Layers](https://github.com/Ved-Piyush/MEnKF-ANN-PUL/blob/main/Data_Generation/LSTM_Heavy_Dropout/Generate_Heavy_Dropout_Second_LSTM.ipynb). 

The code to train the True LSTM (with one dropout layer) can be found at [True_LSTM_Training_One_Dropout_Layer](https://github.com/Ved-Piyush/MEnKF-ANN-PUL/blob/main/Data_Generation/LSTM_Low_Dropout/Generate_Low_Dropout_First_LSTM.ipynb). The code to train the Fitted LSTMs (with one dropout layer) can be found at [Fitted_LSTM_Training_One_Dropout_Layer](https://github.com/Ved-Piyush/MEnKF-ANN-PUL/blob/main/Data_Generation/LSTM_Low_Dropout/Generate_Low_Dropout_Second_LSTM.ipynb). 

The code to train the ANN model using backpropagation that uses the Doc2Vec representation for the sequences can be found at [Doc2Vec_with_ANN_Training](https://github.com/Ved-Piyush/MEnKF-ANN-PUL/blob/main/Data_Generation/Doc2Vec_ANN/Generate_Low_Dropout_First_LSTM.ipynb) 

## Reproducing the Tables and Figures in the Manuscript
The table below links the Jupyter Notebooks and the resulting output files from the notebooks to verify the results in the Tables 1 through Table 6 of the manuscript. 

| Table From the Paper  | Python Code |  Output Files        |
| ------------- | ------------- | ------------- |
| 1  | [First_Row](https://github.com/Ved-Piyush/MEnKF-ANN-PUL/blob/main/Simulations_EnKF_LSTM_Doc2Vec_Heavy_Dropout/Simulations_EnKF_Old_Strategy_Doc2Vec_lstm_extract_var_16_size_ens_216.ipynb), [Second_Row](https://github.com/Ved-Piyush/MEnKF-ANN-PUL/blob/main/Simulations_EnKF_LSTM_Doc2Vec_Heavy_Dropout/Simulations_EnKF_Old_Strategy_Doc2Vec_lstm_extract_var_32_size_ens_216.ipynb)   | [First_Row](https://github.com/Ved-Piyush/MEnKF-ANN-PUL/blob/main/Simulations_EnKF_LSTM_Doc2Vec_Heavy_Dropout/mean_metrics_EnKF_LSTM_Doc2Vec_var_weights_16_num_ens_216.csv), [Second_Row](https://github.com/Ved-Piyush/MEnKF-ANN-PUL/blob/main/Simulations_EnKF_LSTM_Doc2Vec_Heavy_Dropout/mean_metrics_EnKF_LSTM_Doc2Vec_var_weights_32_num_ens_216.csv)               |
| 2  | [First_Row](https://github.com/Ved-Piyush/MEnKF-ANN-PUL/blob/main/Simulations_MC_Dropout_Just_LSTM_Heavy_Dropout/Simulations_MC_Dropout_Old_Strategy_Just_LSTM_extract_rate_0.5_bnn_reps_50.ipynb), [Second_Row](https://github.com/Ved-Piyush/MEnKF-ANN-PUL/blob/main/Simulations_MC_Dropout_Just_LSTM_Heavy_Dropout/Simulations_MC_Dropout_Old_Strategy_Just_LSTM_extract_rate_0.5_bnn_reps_200.ipynb)  | [First_Row](https://github.com/Ved-Piyush/MEnKF-ANN-PUL/blob/main/Simulations_MC_Dropout_Just_LSTM_Heavy_Dropout/mean_metrics_MCD_Just_LSTMrate0.5_bnn_reps_50.csv), [Second_Row](https://github.com/Ved-Piyush/MEnKF-ANN-PUL/blob/main/Simulations_MC_Dropout_Just_LSTM_Heavy_Dropout/mean_metrics_MCD_Just_LSTMrate0.5_bnn_reps_200.csv) |
| 3  | [First_Row](https://github.com/Ved-Piyush/MEnKF-ANN-PUL/blob/main/Simulations_EnKF_Doc2Vec_Word2Vec_Heavy_Dropout/Simulations_EnKF_Old_Strategy_Word2Vec_lstm_extract_var_16_size_ens_216.ipynb), [Second_Row](https://github.com/Ved-Piyush/MEnKF-ANN-PUL/blob/main/Simulations_EnKF_Doc2Vec_Word2Vec_Heavy_Dropout/Simulations_EnKF_Old_Strategy_Word2Vec_lstm_extract_var_32_size_ens_216.ipynb)  | [First_Row](https://github.com/Ved-Piyush/MEnKF-ANN-PUL/blob/main/Simulations_EnKF_Doc2Vec_Word2Vec_Heavy_Dropout/mean_metrics_EnKF_LSTM_Doc2Vec_var_weights_16_num_ens_216.csv), [Second_Row](https://github.com/Ved-Piyush/MEnKF-ANN-PUL/blob/main/Simulations_EnKF_Doc2Vec_Word2Vec_Heavy_Dropout/mean_metrics_EnKF_LSTM_Doc2Vec_var_weights_32_num_ens_216.csv)|
| 4 | [First_Row](https://github.com/Ved-Piyush/MEnKF-ANN-PUL/blob/main/Real_World_EnKF_LSTM_Doc2Vec_Heavy_Dropout/Real_World_EnKF_Old_Strategy_Doc2Vec_lstm_extract_var_16_size_ens_216.ipynb), [Second_Row](https://github.com/Ved-Piyush/MEnKF-ANN-PUL/blob/main/Real_World_EnKF_LSTM_Doc2Vec_Heavy_Dropout/Real_World_EnKF_Old_Strategy_Doc2Vec_lstm_extract_var_32_size_ens_216.ipynb)  | [First_Row](https://github.com/Ved-Piyush/MEnKF-ANN-PUL/blob/main/Real_World_EnKF_LSTM_Doc2Vec_Heavy_Dropout/Real_World_mean_metrics_EnKF_LSTM_Doc2Vec_var_weights_16_num_ens_216.csv), [Second_Row](https://github.com/Ved-Piyush/MEnKF-ANN-PUL/blob/main/Real_World_EnKF_LSTM_Doc2Vec_Heavy_Dropout/Real_World_mean_metrics_EnKF_LSTM_Doc2Vec_var_weights_32_num_ens_216.csv)|
| 5 | [LSTM_1](https://github.com/Ved-Piyush/MEnKF-ANN-PUL/blob/main/Real_World_LSTM_Models_Heavy_Dropout/Real_World_Just_One_LSTM_Old_Strategy_Heavy_Dropout.ipynb), [LSTM_2](https://github.com/Ved-Piyush/MEnKF-ANN-PUL/blob/main/Real_World_LSTM_Models_Low_Dropout/Real_World_Just_One_LSTM_Old_Strategy_Low_Dropout.ipynb), [MEnKF-ANN11](https://github.com/Ved-Piyush/MEnKF-ANN-PUL/blob/main/Real_World_EnKF_LSTM_Doc2Vec_Heavy_Dropout/Real_World_EnKF_Old_Strategy_Doc2Vec_lstm_extract_var_16_size_ens_216.ipynb), [MEnKF-ANN12](https://github.com/Ved-Piyush/MEnKF-ANN-PUL/blob/main/Real_World_EnKF_LSTM_Doc2Vec_Heavy_Dropout/Real_World_EnKF_Old_Strategy_Doc2Vec_lstm_extract_var_32_size_ens_216.ipynb), [MEnKF-ANN21](https://github.com/Ved-Piyush/MEnKF-ANN-PUL/blob/main/Real_World_EnKF_LSTM_Doc2Vec_Low_Dropout/Real_World_EnKF_Old_Strategy_Doc2Vec_lstm_extract_var_16_size_ens_216.ipynb), [MEnKF-ANN22](https://github.com/Ved-Piyush/MEnKF-ANN-PUL/blob/main/Real_World_EnKF_LSTM_Doc2Vec_Low_Dropout/Real_World_EnKF_Old_Strategy_Doc2Vec_lstm_extract_var_32_size_ens_216.ipynb)| [LSTM_1](https://github.com/Ved-Piyush/MEnKF-ANN-PUL/blob/main/Real_World_LSTM_Models_Heavy_Dropout/mean_metrics_real_world_just_lstm_heavy_dropout_drop_rate_0.5_bnn_reps_200.csv), [LSTM_2](https://github.com/Ved-Piyush/MEnKF-ANN-PUL/blob/main/Real_World_LSTM_Models_Low_Dropout/mean_metrics_real_world_just_lstm_low_dropout_drop_rate_0.5_bnn_reps_200.csv), [MEnKF-ANN11](https://github.com/Ved-Piyush/MEnKF-ANN-PUL/blob/main/Real_World_EnKF_LSTM_Doc2Vec_Heavy_Dropout/Real_World_mean_metrics_EnKF_LSTM_Doc2Vec_var_weights_16_num_ens_216.csv), [MEnKF-ANN12](https://github.com/Ved-Piyush/MEnKF-ANN-PUL/blob/main/Real_World_EnKF_LSTM_Doc2Vec_Heavy_Dropout/Real_World_mean_metrics_EnKF_LSTM_Doc2Vec_var_weights_32_num_ens_216.csv), [MEnKF-ANN21](https://github.com/Ved-Piyush/MEnKF-ANN-PUL/blob/main/Real_World_EnKF_LSTM_Doc2Vec_Low_Dropout/Real_World_mean_metrics_EnKF_LSTM_Doc2Vec_var_weights_16_num_ens_216.csv), [MEnKF-ANN22](https://github.com/Ved-Piyush/MEnKF-ANN-PUL/blob/main/Real_World_EnKF_LSTM_Doc2Vec_Low_Dropout/Real_World_mean_metrics_EnKF_LSTM_Doc2Vec_var_weights_32_num_ens_216.csv) |
| 6  | [First_Row](https://github.com/Ved-Piyush/MEnKF-ANN-PUL/blob/main/Real_World_ANN_LSTM_Generate_EnKF_LSTM_Doc2Vec/Real_World_ANN_LSTM_EnKF_LSTM_Doc2Vec_var_2_size_ens_433.ipynb), [Second_Row](https://github.com/Ved-Piyush/MEnKF-ANN-PUL/blob/main/Real_World_ANN_LSTM_Generate_EnKF_LSTM_Doc2Vec/Real_World_ANN_LSTM_EnKF_LSTM_Doc2Vec_var_4_size_ens_433.ipynb)   | [First_Row](https://github.com/Ved-Piyush/MEnKF-ANN-PUL/blob/main/Real_World_ANN_LSTM_Generate_EnKF_LSTM_Doc2Vec/Real_World_mean_metrics_ANN_LSTM_EnKF_LSTM_Doc2Vec_var_weights_2_num_ens_433.csv), [Second_Row](https://github.com/Ved-Piyush/MEnKF-ANN-PUL/blob/main/Real_World_ANN_LSTM_Generate_EnKF_LSTM_Doc2Vec/Real_World_mean_metrics_ANN_LSTM_EnKF_LSTM_Doc2Vec_var_weights_4_num_ens_433.csv) |

The table below links the Jupyter Notebooks that were used to generate Figures 3 through Figure 6 of the manuscript. 

| Figure from the Paper  | Python Code | Comments|
| ------------- | ------------- | ------------- |
| 3  | [Code](https://github.com/Ved-Piyush/MEnKF-ANN-PUL/blob/main/Boxplots_Just_LSTMs_with_Dropout.ipynb)| Scroll to the end of the script to see the plot|
| 4  | [Code](https://github.com/Ved-Piyush/MEnKF-ANN-PUL/blob/main/Simulations_EnKF_LSTM_Doc2Vec_Heavy_Dropout/Simulations_EnKF_Old_Strategy_Doc2Vec_lstm_extract_var_32_size_ens_216.ipynb)| Scroll to Jupyter cell [79] to see the plot| 
| 5  | [Code](https://github.com/Ved-Piyush/MEnKF-ANN-PUL/blob/main/Real_World_EnKF_LSTM_Doc2Vec_Heavy_Dropout/Real_World_EnKF_Old_Strategy_Doc2Vec_lstm_extract_var_16_size_ens_216.ipynb)|Scroll to the end of the script to see the plot| 
| 6  | [Code](https://github.com/Ved-Piyush/MEnKF-ANN-PUL/blob/main/Boxplots_EnKFs.ipynb)|Scroll to the end of the script to see the plot|
