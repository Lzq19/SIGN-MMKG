# SIGN-MMKG
Multimodal Sign Language Knowledge Graph and Representation: Text, Video KeyFrames, and Motion Trajectories
## Installation
```bash
$ git clone https://github.com/yourusername/SLKG.git
$ cd SIGM-MMKG
$ conda env create -f sign-mmkg.yml
```
## Dataset
```bash
$ link：https://pan.baidu.com/s/172flnNFWnecRyqJfRzixug?pwd=hiaz
$ code：hiaz
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