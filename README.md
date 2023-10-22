# TempCRCIBD

Welcome to the TempCRCIBD repository. Follow the instructions below to set up and run the notebooks.

## Setup

1. **Install Dependencies**:
   - First, ensure you have installed all required packages from `requirements.txt` using:
     ```bash
     pip install -r requirements.txt
     ```

2. **Datasets**: 
   - Download the datasets `.rar` file [here](https://www.mediafire.com/file/7m0f8p3b80atei7/Datasets.rar/file).
   - Extract its contents.
   - Place the `Datasets` folder at the same directory level as the other folders (`1_CRC`, `2_IBD`, etc.).

## Running Order

**For CRC**:
1. Navigate to `1_CRC/` and run:
   - `CRC_FS.ipynb`
   - `CRC_Classification.ipynb`

**For IBD**:
1. Navigate to `2_IBD/` and run:
   - `IBD_FS.ipynb`
   - `IBD_Classification.ipynb`
