# SIGN-MMKG
Multimodal Sign Language Knowledge Graph and Representation: Text, Video KeyFrames, and Motion Trajectories
## Installation
```bash
$ git clone https://github.com/Lzq19/SIGN-MMKG.git
$ cd SIGM-MMKG
$ conda env create -f sign-mmkg.yml
$ conda activate sign-mmkg
```
## Dataset
```
https://drive.google.com/file/d/1AP12UvwF1giPJzN1HR11DZ9wp30hiODU/view?usp=drive_link
```
First, get the dataset from the link above. Then place the data according to the following structure.
```
├── SIGN-MMKG/
│   └── data/
│       ├── img
│       ├── motion
│       └── kg_T_datasource_number.txt
```
## Train
```bash
$ pyhton SIGN-MMKG/main.py
```
## Test
```bash
$ pyhton SIGN-MMKG/test.py
```