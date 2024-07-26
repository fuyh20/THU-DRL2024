[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/xRGOQOFk)
# Homework 2

This assignment consists of two tasks. The detailed requirements of each task can be found in `hw2.pdf`.

Please fill in the blanks inside the "YOUR IMPLEMENTATION HERE" comments, any changes outside the area might be **ignored**.

## Getting Started

Since we are using PyTorch for `hw2`, we recommend using conda to manage the environment. Please refer to the [miniconda](https://docs.conda.io/en/latest/miniconda.html) or [miniforge](https://github.com/conda-forge/miniforge) homepage for a compact conda (or mamba) installation.

You have two options for creating the environment of hw2. For users without a CUDA device, please remove the `pytorch-cuda` term either way for a CPU-only installation.
* To create a new conda environment, simply run `conda env create -f environment.yml`
* If you want to install the package within the environment you created with `hw1`, please follow the below steps:

  ```bash
  conda activate <hw1-env-name>
  # remove the pytorch-cuda=12.1 term if you want a cpu-only installation
  conda install pytorch==2.2.1 pytorch-cuda=12.1 -c pytorch -c nvidia
  pip install gymnasium[classic_control]==0.29.1
  pip install matplotlib
  # for hyperparameter management
  pip install hydra-core==1.3.2
  # for video recording
  pip install moviepy==1.0.3
  # for attribute-like config access
  pip install dotmap
  # for tabular output of `test_main`
  pip install tabulate
  ```

That's it! If you encounter any trouble creating the environment, please let us know :-)


## Submitting the Results

After finishing your homework, please remember to commit the changes and push them to the GitHub repo.

TAs have created a GitHub workflow to evaluate your code automatically, and a reference score of your homework can be found on the Actions page of your repository. (The total points of the auto-test is 30, and 70 points are assigned to the report.)

You can also use the following command to quickly test your code locally.
```
python test_main.py
```
If your code is correct, you will see the following output:
```bash
Test Name                 Result    Score
------------------------  --------  -------
test_get_Q                Passed    10 pts
test_get_Q_target         Passed    5 pts
test_get_double_Q_target  Passed    5 pts
test_get_action           Passed    5 pts
test_dueling_forward      Passed    5 pts
------------------------  --------  -------
Total                     5/5       30 pts
```

However, the local test does not guarantee the final score you will get from the GitHub classroom. Please make sure your last commit can pass the GitHub workflow.

## NOTICE: DO NOT MODIFY AUTO-GRADING CODE!

The auto-grading relies on the contents within the `.github`, `test` folders, and the `test_main.py` file. These files should not be altered. You will not be able to modify your commit history as all changes are saved permanently in the GitHub classroom records.

Attempts to bypass the auto-grading system, either by altering grading scripts or creating functions intended to manipulate the test cases, will be recorded. Teaching assistants can easily verify any changes to these files.

Modifying auto-grading code or attempting to manipulate it will result in a zero mark for the assignment.