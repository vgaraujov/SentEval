#!/bin/bash

sbatch genius_probe.slurm English bert 8446
sbatch genius_probe.slurm English vit-mae 8446

sbatch genius_probe.slurm English pixel 32873
sbatch genius_probe.slurm English bert 32873
sbatch genius_probe.slurm English vit-mae 32873

