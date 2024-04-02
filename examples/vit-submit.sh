#!/bin/bash


sbatch vit-finetuning.slurm dep UD_Vietnamese-VTB
sbatch vit-finetuning.slurm dep UD_Coptic-Scriptorium
sbatch vit-finetuning.slurm pos UD_Coptic-Scriptorium

sbatch random-model.slurm pos UD_Vietnamese-VTB
sbatch random-model.slurm pos UD_Coptic-Scriptorium

