#!/bin/bash


sbatch -o pixel-words.out probe_h100.slurm English pixel-words
sbatch -o pixel-bigrams.out probe_h100.slurm English pixel-bigrams
sbatch -o pixel-bigrams-r.out probe_h100.slurm English pixel-bigrams-r



# sbatch -o visual_bert probe_h100.slurm Visual bert
# sbatch -o visual_pixel probe_h100.slurm Visual pixel

