#!/bin/bash


sbatch genius_probe.slurm English pixel 14768
sbatch genius_probe.slurm English bert 14768
sbatch genius_probe.slurm English vit-mae 14768

sbatch genius_probe.slurm English pixel 74629
sbatch genius_probe.slurm English bert 74629
sbatch genius_probe.slurm English vit-mae 74629

