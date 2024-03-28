#!/bin/bash

sbatch vit-finetuning.slurm dep UD_Arabic-PADT 2f9506e16930f137abbd18a3fb16f6b31840a830
sbatch vit-finetuning.slurm dep UD_Korean-GSD 2f9506e16930f137abbd18a3fb16f6b31840a830
sbatch vit-finetuning.slurm dep UD_Japanese-GSD 2f9506e16930f137abbd18a3fb16f6b31840a830
sbatch vit-finetuning.slurm dep UD_Chinese-GSD 2f9506e16930f137abbd18a3fb16f6b31840a830
sbatch vit-finetuning.slurm dep UD_Tamil-TTB 2f9506e16930f137abbd18a3fb16f6b31840a830
sbatch vit-finetuning.slurm dep UD_Hindi-HDTB 2f9506e16930f137abbd18a3fb16f6b31840a830

sbatch vit-finetuning.slurm ner ner/amh 2f9506e16930f137abbd18a3fb16f6b31840a830
sbatch vit-finetuning.slurm ner ner/hau 2f9506e16930f137abbd18a3fb16f6b31840a830
sbatch vit-finetuning.slurm ner ner/ibo 2f9506e16930f137abbd18a3fb16f6b31840a830
sbatch vit-finetuning.slurm ner ner/kin 2f9506e16930f137abbd18a3fb16f6b31840a830
sbatch vit-finetuning.slurm ner ner/lug 2f9506e16930f137abbd18a3fb16f6b31840a830
sbatch vit-finetuning.slurm ner ner/luo 2f9506e16930f137abbd18a3fb16f6b31840a830
sbatch vit-finetuning.slurm ner ner/pcm 2f9506e16930f137abbd18a3fb16f6b31840a830
sbatch vit-finetuning.slurm ner ner/swa 2f9506e16930f137abbd18a3fb16f6b31840a830
sbatch vit-finetuning.slurm ner ner/yor 2f9506e16930f137abbd18a3fb16f6b31840a830
sbatch vit-finetuning.slurm ner ner/wol 2f9506e16930f137abbd18a3fb16f6b31840a830

