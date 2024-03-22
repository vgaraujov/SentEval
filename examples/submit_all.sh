#!/bin/bash

sbatch probe.slurm English pixel 8446
sbatch probe.slurm English bert 8446
sbatch probe.slurm English vit-mae 8446

sbatch probe.slurm English pixel 32873
sbatch probe.slurm English bert 32873
sbatch probe.slurm English vit-mae 32873

