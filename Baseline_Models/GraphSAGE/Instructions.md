# Graphsage Instructions
## Step 1: Data Creation
- To prepare the data for graphsage, please refer to the input_creation.py file.
## Step 2: Data Prep
- Next, run the Graphsage_data_prep.py file to get the data into the correct directory.
## Step 3: Docker Creation
- Next, CD into Baseline_Models/GraphSAGE and run the following command to create a docker container:
    ```
    docker build -t graphsage .
    ```
- Once that docker container is built, enter the container by running this command:
    ```
    docker run -it graphsage bash
    ```
(Note: any time the graphsage code is modified, you must rebuild the container to see the changes reflected.)
## Step 4: Running the Model
To run the model, run the following code:
```
python -m graphsage.supervised_train \
--train_prefix ./graph/max_pygraphs \
--model graphsage_mean \
--learning_rate 0.001 \
--epochs 10 \
--dropout 0.5 \
--weight_decay 0.0005 \
--max_degree 128 \
--samples_1 25 \
--samples_2 10 \
--samples_3 0 \
--dim_1 128 \
--dim_2 128 \
--batch_size 512 \
--identity_dim 16 \
--verbose
```
(Note: modifying some of these values may be necessary to better optimize these values)