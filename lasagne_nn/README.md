# lasange_nn

This is the keiserlab package for neural networks written in [Lasagne](https://github.com/Lasagne/Lasagne).

## train hdf5_basic_nn.py
To train hdf5_basic_nn using a pickled dataset
([like this one](https://drive.google.com/a/keiserlab.org/file/d/0Bzy2bqh5kfl0aHNpZ0tGdXFQSVU/view?usp=drive_web))
type this in the command line from the lasagne_nn directory:

    python hdf5_basic_nn.py -o output/example_output_dir \
    -d ../datasets/STchembl20_800maxmw_10moleculesmin_medianval_mid_4096bvECFP.hdf5

you could optionally create a file with the arguments and pass that to hdf5_basic_nn.py

create a file like [example_args/basic_nn_args.txt](example_args/basic_nn_args.txt):

    --output_directory
    output/example_output
    --dataset
    ../data/STchembl20_800maxmw_10moleculesmin_medianval_mid_4096bvECFP.hdf5
    --num_epochs
    3


then in the command line:

    python hdf5_basic_nn.py @example_args/basic_nn_args.txt


#### resume training on existing trained network
To resume training or to train more epochs, pass the existing argument file and add the `resume` argument.
You can also pass the 'num_epochs' argument to override the value in the args file.

     python hdf5_basic_nn.py @example_args/basic_nn_args.txt --num_epochs 4000 --resume


## run_nn.py
To get a prediction using a neural network and trained parameters use run_nn.py like this
(also take a look at [run_nn_args.txt](example_args/run_nn_args.txt)) in example_args)

    python run_nn.py -o output/example_output_dir/drug_matrix_prediction \
    -f ../data/drugmatrix_id_fingerprint.csv \
    -t data/chmbl20_cta_ijv_m10cpt_gtrel_sub_23l.pcl \
    -w output/example_output_dir/model_at_epoch_490.npz \
    -n hdf5_basic_nn.py


## useful scripts in neural-nets/common
### k-fold cross validation
To train using k-fold cross validation use [neural-nets/common/kfold_cl.py](../common)
 which creates pickled test indices, output directories and bash scripts for each fold.
 Training hdf5_basic_nn.py using 5 fold cross validation and the `args.txt` file from above would go like this:

Create an arguments like [kfold_args.txt](example_args/kfold_args.txt) in example_args (you might want to use absolute paths to avoid confusion):

    --output_directory
    output/5fold_hdf5_basic_nn_example_output
    --dataset
    ../data/STchembl20_800maxmw_10moleculesmin_medianval_mid_4096bvECFP.hdf5
    --n_folds
    5
    --command-line
    python hdf5_basic_nn.py @example_args/hdf5_basic_nn_args.txt --output_directory {fold_dir} --target_dataset {index_file}


Then in the command line:

`python ../common/kfold_cl.py @5fold_train_args.txt`


Follow the instructions to train each fold, also run `python kfold_cl.py -h` to see more options like automatically running each script in parallel or one after the other


### prediction_analysis.py
This script compares the prediction from run_nn.py with known truths. As of now it's hard-coded to do drug-matrix
comparisons but we should probably move the non-plotting functions from nn_reporter to this script.
To compare the prediction from run_nn.py against datamatrix provide a prediction file and the datamatrix export csv.

[prediction_analysis_args.txt](example_args/prediction_analysis_args.txt)

    --prediction_path
    ../lasagne_nn/output/5fold_example_output_dir/fold_0/drugmatrix_prediction/nn_predicted_targets.csv
    --dm_export_path
    ../data/drugmatrix_chembl20_export.csv

