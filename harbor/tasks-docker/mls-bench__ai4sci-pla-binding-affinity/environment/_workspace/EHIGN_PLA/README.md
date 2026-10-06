# Interaction-Based Inductive Bias in Graph Neural Networks: Enhancing Protein-Ligand Binding Affinity Predictions from 3D Structures

## Note
- Implementation of other baselines can be found on [GIGN](https://github.com/guaguabujianle/GIGN).  
- This repository contains the source code for PLA prediction. For structure-based virtual screening (SBVS), please refer to our dedicated repository at [EHIGN_SBVS on GitHub](https://github.com/guaguabujianle/EHIGN_SBVS).

## Dataset
All data used in this paper are publicly available at the following locations:
- **PDBbind v2016 and v2019:** [pdbbind](http://www.pdbbind.org.cn/download.php)
- **2013 and 2016 core sets:** [casf](http://www.pdbbind.org.cn/casf.php)

The preprocessed data can be downloaded from [Graphs](https://drive.google.com/file/d/1oGUP4z7htNXyxTqx95HNSDLsaoxa3fX7/view?usp=share_link).

## Requirements  
dgl==0.9.0  
networkx==2.5  
numpy==1.19.2  
pandas==1.1.5  
pymol==0.1.0  
rdkit==2022.3.5  
scikit_learn==1.1.2  
scipy==1.5.2  
torch==1.10.2  
tqdm==4.63.0  
openbabel==3.3.1 (conda install -c conda-forge openbabel)  

Alternatively, install the environment using the provided YAML file at `./environment.yaml`.


## Descriptions of Folders and Files
- **`./data`:** Contains information about various datasets. Download and organize preprocessed datasets as described.
- **`./config`:** Parameters used in EHIGN.
- **`./log`:** Logger.
- **`./model`:** Contains model checkpoints and training records.
- **Scripts and Implementations:** Various Python files implementing models, preprocessing, training, and testing.

## Step-by-step Running

### 1. Model Training
- Download the preprocessed datasets and organize them in the `./data` folder.
- Run `python train.py`.

### 2. Model Testing
- Run `python test.py` (modify file paths in the source code if necessary).

### 3. Process Raw Data
- Run a demo using provided examples:
  - `python preprocess_complex.py`
  - `python graph_constructor.py`
  - `python train_example.py`

### 4. Test the Trained Model in Other External Test Sets
- Organize the data like:
  -data  
  &ensp;&ensp;-external_test  
  &ensp; &ensp;&ensp;&ensp; -pdb_id  
  &ensp; &ensp; &ensp;&ensp;&ensp;&ensp;-pdb_id_ligand.mol2  
  &ensp; &ensp; &ensp;&ensp;&ensp;&ensp;-pdb_id_protein.pdb
  
- Execute the following commands:
  - `python preprocess_complex.py`
  - `python graph_constructor.py`
  - `python test.py`
  - (Modify file paths in the source code if necessary)

### 5. Cold Start Settings
- Use datasets found in the `./cold_start_data` folder.
- Execute scripts `train_random.py`, `train_scaffold.py`, and `train_sequence.py` if the original training set has been processed.




