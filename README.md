# NeurIPS 2019 Reproducibilty Challenge "Zero-shot Knowledge Transfer via Adversarial Belief Matching"

Our code for the [Reproducibility Challenge](https://openreview.net/group?id=NeurIPS.cc/2019/Reproducibility_Challenge) of the 2019 Conference on Neural Information Processing Systems.

We participate in the Baseline Track with the paper [Zero-Shot Knowledge Transfer via Adversarial Belief Matching
](https://arxiv.org/abs/1905.09768). 

Update: Our work was accepted from the NeurIPS 2019 Reproducibility Challenge Program Chairs and is under publication!
[Journal Publication](http://rescience.github.io/bibliography/Ferles_2020.html) [Open Review](https://openreview.net/forum?id=B1gZ8acM6S)

## Wide Residual Networks, Few-Shot Knowledge Distillation and Zero-Shot Knowledge Transfer

The main steps of this work include re-implementation of:

1. [Wide Residual Networks](https://arxiv.org/abs/1605.07146) (WRNs) that are solely used as our teacher and student networks on all knowledge transfer settings. 
2. A spatial-attention knowledge transfer [setting](https://arxiv.org/abs/1612.03928) which we used as our main source of comparison with the paper's main method.  
3. Zero-shot knowledge transfer as described on the paper.
4. A method that measures the degree of belief matching of zero-shot and few-shot as it was also introduced on the main paper. 

On top of these, we manipulate the main method on what we name as 'modified zero-shot training' where we further exploit the generator network that was used on the main paper. 

![](figs/overview.png?raw=true)


## Run our experiments from scratch

For all the experiments listed on the main report and its 'Appendix' section, we provide the corresponding configuration files in JSON format so that you can re-train a Wide ResNet of your choice under the same circumstances as we did. 
Feel free to manipulate the config files to run your own experiments. 

Additionally, we have provided all necessary pre-trained documents needed for the experiments (WRN scratches, zero-shot and few-shot trained models for extra experiments and adversarial belief matching) in the 'PreTrainedModels' folder.  The folder structure has already been taken care of, so please note that changing a file location requires an assignment of different file paths in the training codes. 

### Example runs

Train a scratch of Wide ResNet-16-1 on CIFAR: 
```
python train_scratches.py --config configs/scratch_configs/WRN-16-1-scratch-cifar.json
```

Train a no-teacher WRN-16-1 with M=10 samples per class on SVHN:
```
python train_no_teacher.py --config configs/no_teacher_configs/SVHN_KD_ATT_M10.json
```

Train a WRN-16-1 on CIFAR with M=25 samples per class and few-shot knowledge distillation from WRN-40-2:
```
python train_kd_att.py --config configs/kd_att_configs/CIFAR10_KD_ATT_M25.json
```

Train a WRN-40-2, WRN-40-1 teacher-student pair on CIFAR with M=200 samples per class:

```
python train_kd_att.py --config configs/kd_att_configs/CIFAR10_KD_ATT_TEACHER_WRN_40_2_STUDENT_WRN_40_1.json
```

Repeat the above pair in a zero-shot knowledge transfer setting:

```
python train_reproducibility_zero_shot.py --config configs/zero_shot_configs/CIFAR10_ZERO_SHOT_TEACHER_WRN_40_2_STUDENT_WRN_40_1.json
```

Or use our own method to do so:
```
python train_modified_zero_shot.py --config configs/zero_shot_configs/CIFAR10_ZERO_SHOT_TEACHER_WRN_40_2_STUDENT_WRN_40_1.json
```

Update a zero-shot pre-trained WRN-16-1 on SVHN with few-shot attention transfer from its WRN-40-2 teacher (also used on the zero-shot pre-training) with M=50 samples per class:

```
python train_with_extra_M_samples_zero_shot.py --config configs/extra_M_configs/SVHN_ZERO_SHOT_M50.json
```

And again, try the same update setting on a network trained by our method:
```
python train_with_extra_M_samples_zero_shot.py --config configs/extra_M_configs/SVHN_ZERO_SHOT_M50_MODIFIED.json
```

Finally, measure belief matching between Zero-Shot and Few-Shot WRNs-16-1 with their WRN-40-2 teacher:
```
python adversarial_belief_matching.py --config configs/abm_configs/ABM_ZERO_SHOT_CIFAR.json && python adversarial_belief_matching.py --config configs/abm_configs/ABM_ZERO_SHOT_SVHN.json
``` 

## Use our pre-trained models 

We publicly share a Dropbox [folder](https://www.dropbox.com/sh/xuk69az4dlw26uu/AAAG2v_tgXrivL_dSHd486a9a?dl=0) with several checkpoint files for our trained model. We have also compiled a few [Google Colaboratory](https://colab.research.google.com) notebooks that present some of our most interesting results based on these checkpoints. You can upload and use them (remember to enable GPU acceleration!) for inference.   

## Read our reproducibility paper

Our paper [[RE] Zero-Shot Knowledge Transfer via Adversarial Belief Matching](https://github.com/AlexandrosFerles/NIPS_2019_Reproducibilty_Challenge_Zero-shot_Knowledge_Transfer_via_Adversarial_Belief_Matching/blob/master/%5BRE%5D%20Zero-Shot%20Knowledge%20Transfer%20via%20Adversarial%20Belief%20Matching.pdf) highlights the key points of our work towards the reproducibility of all the methods and experiments presented in the [original work](https://arxiv.org/abs/1905.09768) along with our own modifications. 

## Acknowledgements

This work was initiated as a project of our master's level [course](https://www.kth.se/student/kurser/kurs/DD2412?l=en) titled 'Deep Learning, Advanced Course' @ KTH Stockholm, Sweden. We would like to thank the course staff for providing us with the necessary Google Cloud tickets to run our (several) experiments. 

Most importantly, we would like to thank the authors of the original paper, [Paul Micaelli](https://github.com/polo5) and [Amos Storkey](https://homepages.inf.ed.ac.uk/amos/) for answering immediately all the questions that came up during this work. 

## Members

This work could not have been completed without the help and collaboration of [Alexander Nöu
](https://github.com/AlexLacson) and [Leonidas Valavanis](https://github.com/valavanisleonidas).   
