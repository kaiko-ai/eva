# TIGER (Tumor Infiltrating Lymphocytes in breast cancER)

TIGER contains digital pathology images of Her2 positive (Her2+) and Triple Negative (TNBC) breast cancer whole-slide images, together with manual annotations. Training data comes from multiple sources. A subset of Her2+ and TNBC cases is provided by the Radboud University Medical Center (RUMC) (Nijmegen, Netherlands). A second subset of Her2+ and TNBC cases is provided by the Jules Bordet Institut (JB) (Bruxelles, Belgium). A third subset of TNBC cases only is derived from the TCGA-BRCA archive obtained from the Genomic Data Commons Data Portal.

It contains 3 different datasets and thus 3 different tasks to add to eva. 

WSIBULK - WSI level classification task: Detecting tumour presence in patches of a given slide.
WSITILS - Regression task: predicting "TIL" score of a whole slide image. 
WSIROIS - Cell level segmentation task: predicting boundaries of TIL cells. 

However only WSIBULK and WSITILS are currently implemented.

Source: https://tiger.grand-challenge.org/Data/ 


## Raw data

### Key stats

|                           |                                                          |
|---------------------------|----------------------------------------------------------|
| **Modality**              | Vision (WSI)                                             |
| **Tasks**                 | Binary Classification / Regression                       |
| **Cancer type**           | Breast                                                   |
| **Data size**             | 182 GB                                                   |
| **Image dimension**       | ~20k x 20k x 3                                           |
| **Magnification (μm/px)** | 20x (0.5) - Level 0                                      |
| **Files format**          | `.tif`                                                   |
| **Number of images**      | 178 WSIs (96 for WSIBULK and 82 for WSITILS)             |


### Organization

The data `tiger.zip` from [grand challenge](https://tiger.grand-challenge.org/) is organized as follows:

training/
	|_wsibulk/                                      * Used for classification task
	|	|__annotations-tumor-bulk/                  * Manual annotations of "tumor bulk" regions
	|	|	|___masks/	                            * Binary masks in TIF format					
	|	|	|___xmls/                               * Not used in eva
	|	|__images/									* Whole-Slide Images
    |   │   ├── 103S.tiff
    │   |   └── ...						
	|	|__tissue-masks/                            * Not used in eva	
	|
	|_wsirois/                                      * Not used in eva currently
	|
	|_wsitils/	                                    * Used for regression task
	|	|__images/									* Whole-slide images
    |   │   ├── 104S.tiff
    │   |   └── ...									
	|	|__tissue-masks/                            * Not used in eva
	|	|__tiger-tils-scores-wsitils.csv            * Target variable file


## Download and preprocessing

The `TIGER` dataset class doesn't download the data during runtime and must be downloaded manually as follows:

- Make sure that the latest version of the AWS CLI is installed on your system by following [these instructions](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html)

With the AWS CLI installed, you can download the official training set (no AWS account required) by running:

`aws s3 cp s3://tiger-training/ /path/to/destination/ --recursive --no-sign-request`

These instructions can also be found on the official challenge page [here](https://tiger.grand-challenge.org/Data/)

We then generate random stratified train / validation and test splits using a 0.7 / 0.15 / 0.15 ratio.




