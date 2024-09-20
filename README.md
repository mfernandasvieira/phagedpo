# PhageDPO

PhageDPO is a tool designed to predict the depolymerase content of given DNA sequences using a pre-trained machine
learning model. The script includes a command line interface (CLI) to process multiple FASTA files and save the
prediction results in HTML format.

## Installation
To install PhageDPO, follow these steps:

1. **Clone Repository**: Clone the PhageDPO repository to your local machine.
    ```
    git clone https://github.com/mfernandasvieira/phagedpo.git
    ```

2. **Navigate to PhageDPO Directory**: Move into the PhageDPO directory.
    ```
    cd phagedpo
    ```

3. **Set Up Environment**: Navigate to the cloned repository and create the Conda environment using the provided
environment configuration file.
    ```
    conda env create -f environment.yml
    ```

4. **Activate Environment**: Activate the Conda environment.
    ```
    conda activate phagedpo
    ```

## Usage
Run the script from the command line with the required arguments:

```
python phagedpo_cli.py -i genomes
```