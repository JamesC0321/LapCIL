# LapCIL
Codes for molecular classification of endometrial cancer. 
Citation: Jian Chen, Ziyuan Chen, Geng Chen, Mengyu Liu, Sohaib Asif, He Zhang, Jun Jin, Laplacian-guided contextual instance learning for whole slide image classification, Engineering Applications of Artificial Intelligence, Volume 176, Part 1, 2026, 114680.

The file structure is as follows:

1、LapCIL_train_CM16.py
Training script.

2、train_labels.zip
Files containing detailed information on training/validation/testing data for the CAMELYON16 dataset.

3、model/
Folder for Network model.

4、LapCIL_CAMELYON16/
Folder containing experiment results on the CAMELYON16 dataset.

5、cupy_layers/
Third-party library dependencies.

🛠️ Instructions

Preprocessing: The preprocessing step utilizes the code from CALM. 
For specific steps, please refer to the data processing stage in the official repository: https://github.com/mahmoodlab/CLAM.
Data Preparation: Split the data and place it in the train_labels folder.
Execution: Modify the data path and save folder addresses in LapCIL_train_CM16.py, then run the script.
