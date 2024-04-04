#!/bin/bash


sbatch random-model.slurm dep UD_Arabic-PADT
sbatch random-model.slurm dep UD_Chinese-GSD
sbatch random-model.slurm dep UD_Coptic-Scriptorium
sbatch random-model.slurm dep UD_Hindi-HDTB
sbatch random-model.slurm dep UD_Japanese-GSD
sbatch random-model.slurm dep UD_Korean-GSD
sbatch random-model.slurm dep UD_Tamil-TTB
sbatch random-model.slurm dep UD_Vietnamese-VTB


sbatch vit-finetuning.slurm pos UD_Arabic-PADT
sbatch vit-finetuning.slurm pos UD_Chinese-GSD
sbatch vit-finetuning.slurm pos UD_Coptic-Scriptorium
sbatch vit-finetuning.slurm pos UD_Hindi-HDTB
sbatch vit-finetuning.slurm pos UD_Japanese-GSD
sbatch vit-finetuning.slurm pos UD_Korean-GSD
sbatch vit-finetuning.slurm pos UD_Tamil-TTB
sbatch vit-finetuning.slurm pos UD_Vietnamese-VTB
