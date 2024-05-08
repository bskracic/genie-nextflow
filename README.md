# GENIE
Graph gEnetic Network Inference and Evaluation.
Bioinformatics project for discovering impact of different gene interactions represented as a graph on tumor creation.
\
This repository includes everything required to run GENIE using Nextflow and Docker.

## How to run

Firstly, pull the image:

```docker pull bskracic/genie-runtime:latest```

Check the configuraton in `scripts/genie_config.json`:

```json
{
  "genes": [
    "C6orf150",
    "CCL5",
    "CXCL10",
    "TMEM173",
    "CXCL9",
    "CXCL11",
    "NFKB1",
    "IKBKE",
    "IRF3",
    "TREX1",
    "ATM"
  ],
  "survival_data_path": "/home/jugo/research/genie-nextflow/resources/TCGA_survival_data_clean.txt",
  "cancers": {
    "BRCA": {
      "expression": "/home/jugo/research/genie-nextflow/resources/BRCA_expression.csv"
    },
    "LUAD": {
      "expression": "/home/jugo/research/genie-nextflow/resources/LUAD_expression.csv"
    },
    "KIRC": {
      "expression": "/home/jugo/research/genie-nextflow/resources/KIRC_expression.csv"
    },
    "LUNG": {
      "expression": "/home/jugo/research/genie-nextflow/resources/LUNG_expression.csv"
    },
    "THCA": {
      "expression": "/home/jugo/research/genie-nextflow/resources/THCA_expression.csv"
    }
  }
}
```

Run the nextflow script:

```./nextflow -log nextflow-logs/log run main.nf --genie_config_path=$(pwd)/scripts/genie_config.json --wandb_api_key=<api key> --cancers=<list of cancers> --target=<target>```