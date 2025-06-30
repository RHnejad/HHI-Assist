<div align="center">
<h1> HHI-Assist <br>  </h1>



This is the official code for the paper "HHI-Assist: A Dataset and Benchmark of Human-Human Interaction in Physical Assistance Scenario", accepted and published in IEEE Robotics and Automation Letters (RA-L), 2025.

[[arXiv]()] [[webpage](https://sites.google.com/view/hhi-assist/home)]

<image src="docs/hhi2.jpg" width="600">
 

</div>

</br>


## Requirements
The code requires Python 3.10 or later. The file [requirements.txt](requirements.txt) contains the full list of required Python modules.
```
pip install -r requirements.txt
```
## Data directory
Data can be downloaded from [here](https://huggingface.co/datasets/jose-barreiros-tri/hhi-assist).
- Data of Task1 and Task2 is split per caregiver and carereceiver sequences. (```AA-RM, AB-JB, ...```).
- Data of Task3
```
|-- AA
|-- AB
|-- ...
|-- SR
|-- AA-RM
|-- AB-JB
|-- ...
|-- SR-LPR
|-- Task3
```

Convert bvh files to csv files by calling [bvh-converter](https://github.com/tekulvw/bvh-converter).


## Training

To train a model, use the following command:
```python3 main.py --mode train --data-dir PATH_TO_DATA  --output_dir PATH_TP_OUTPUT --batch 256 --two --h 2 --joints 21 --epochs 50```

## Test

To evaluate a model, use the following command:
```python3 main.py --mode test --data-dir PATH_TO_DATA  --output_dir PATH_TP_OUTPUT --batch 256 --two --h 2 --joints 21```





## Acknowledgments

The overall code framework (dataloading, training, testing etc.) was adapted from [DePOSit](https://github.com/vita-epfl/DePOSit/).

### Citation


```
@article{saadatnejad2025hhiassist,
  title={HHI-Assist: A Dataset and Benchmark of Human-Human Interaction in Physical Assistance Scenario},
  author={Saadatnejad, Saeed and Hosseininejad, Reyhaneh and Barreiros, Jose and Tsui, Katherine and Alahi, Alexandre},
  journal={IEEE Robotics and Automation Letters (RA-L)},
  year={2025},
  publisher={IEEE}
}