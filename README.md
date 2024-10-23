# Text to HPO Pipeline

Welcome to the **Text to HPO Pipeline** project! This tool is designed to map free text descriptions of phenotypic abnormalities to Human Phenotype Ontology (HPO) terms, providing a standardized way to capture and analyze phenotypic data.

> **⚠️ Warning:** This project is currently under development. The repository will be fully functional and publicly available soon. Please check back later for updates.

## Overview
The text to HPO Pipeline is a Python-based tool that enables the conversion of unstructured clinical descriptions into structured, standardized HPO terms.

## Installation
Install metamap from https://github.com/AnthonyMRios/pymetamap.

Install pymetamap from https://github.com/AnthonyMRios/pymetamap.

## Config
To run `main.py`, the `text_hpo_mapping/config/config.yaml` file needs to be adapted:

`clinical_data_path` contains the path to the input data. A fake example is provided for illustration.

`base_dir` must contain the base_dir of the metamap install (Note the other metamap settings might need to be adapted as well)

`api_key` must be set to the user's private UMLS api key. Instruction on where to find the API key can be found here: https://documentation.uts.nlm.nih.gov/rest/authentication.html


<!--
## Table of Contents

- [Overview](#overview)

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Input Data](#input-data)
- [Output Data](#output-data)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)

## Overview

The text to HPO Pipeline is a Python-based tool that enables the conversion of unstructured clinical descriptions into structured, standardized HPO terms. This is particularly useful in clinical and research settings where phenotypic data needs to be analyzed, shared, or integrated with other datasets.

### What is HPO?

The Human Phenotype Ontology (HPO) is a standardized vocabulary of phenotypic abnormalities encountered in human disease. Each HPO term describes a phenotypic abnormality, and the ontology covers a wide range of phenotypic abnormalities, from congenital disorders to common symptoms.

## Features

- **Text Normalization**: Cleans and pre-processes free text to facilitate accurate mapping.
- **HPO Term Mapping**: Leverages natural language processing (NLP) techniques to map text to HPO terms.
- **Customizable**: Allows customization of the mapping process, including adding custom dictionaries or adjusting matching algorithms.
- **Scalable**: Can handle large datasets, making it suitable for clinical databases and large research studies.
- **Extensible**: Modular design allows for easy integration with other tools and databases.

## Installation

To install the pipeline, clone the repository and install the required dependencies:

```bash
git clone https://github.com/yourusername/hpo-term-mapping-pipeline.git
cd hpo-term-mapping-pipeline
pip install -r requirements.txt

--->