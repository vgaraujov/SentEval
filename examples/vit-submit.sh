#!/bin/bash


sbatch vit-finetuning.slurm dep UD_Arabic-PADT
sbatch vit-finetuning.slurm dep UD_Korean-GSD
sbatch vit-finetuning.slurm dep UD_Japanese-GSD
sbatch vit-finetuning.slurm dep UD_Chinese-GSD
sbatch vit-finetuning.slurm dep UD_Tamil-TTB
sbatch vit-finetuning.slurm dep UD_Hindi-HDTB

sbatch vit-finetuning.slurm ner ner/amh
sbatch vit-finetuning.slurm ner ner/hau
sbatch vit-finetuning.slurm ner ner/ibo
sbatch vit-finetuning.slurm ner ner/kin
sbatch vit-finetuning.slurm ner ner/lug
sbatch vit-finetuning.slurm ner ner/luo
sbatch vit-finetuning.slurm ner ner/pcm
sbatch vit-finetuning.slurm ner ner/swa
sbatch vit-finetuning.slurm ner ner/yor
sbatch vit-finetuning.slurm ner ner/wol

