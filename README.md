[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/FAEG5PME)
# Installation

This assignment consists of three tasks. The detailed requirements of each task can be found in `hw3.pdf`.

Please fill in the blanks inside the "YOUR IMPLEMENTATION HERE" comments, any changes outside the area might be **ignored**.

## Getting Started

You can use basically the same environment as in hw2, with just one modification.

```bash
conda activate <hw2-env-name>
pip install dotmap gymnasium[box2d]==0.29.1 jaxtyping beartype
```

if you are using zsh as your shell, use the following command instead:

```zsh
conda activate <hw2-env-name>
pip install gymnasium\[box2d\]==0.29.1 jaxtyping beartype
```

We use the `dotmap` package to get a "dot-able" config dictionary as a substitution for the default one, as we find the original dictionary from hydra is slow. And we use `jaxtyping` and `beartype` to better check the input and output argument types and **shapes** of the functions to be completed.

You may encounter several errors when installing `gynmasium[box2d]` depending on your system and requirements installed previously, here's an incomplete list of how to get over them:

1. `error: command 'swig.exe' failed: None` or `command 'swig' failed: No such file or directory`

   In that case, you can install `swig` via `conda install swig` in your `<hw2-env-name>` environment and try again. If the same error persists, try the guide in this [link](https://open-box.readthedocs.io/en/latest/installation/install_swig.html) to install `swig` manually.

2. On the Windows platform, you may encounter the following problem: `error`: Microsoft Visual C++ 14.0 or greater is required. Get it with "Microsoft C++ Build Tools"` 
   
   In that case, you can follow this [link](https://visualstudio.microsoft.com/zh-hans/visual-cpp-build-tools/) which the error message provided, download the build tool file, and run it. You'll need to select the "Desktop development with C++" checkbox in the "Workloads", and you may remove the optional requirements in the right sidebar (MSVC, Windows SDK, CMake tools, etc.) 
   
   After the installation is finished, restart your computer and try installing `gymnasium[box2d]` again (if you encounter the same error, try to select the optional dependencies and try again). There may be other issues concerning the installation of `gymnasium[box2d]`, please contact us if you find yourself in a different situation.

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
test_get_Qs_ddpg          Passed    10 pts
test_get_actor_loss_ddpg  Passed    5 pts
test_get_Qs_td3           Passed    5 pts
test_get_alpha_loss_sac   Passed    5 pts
test_forward_sac          Passed    5 pts
------------------------  --------  -------
Total                     5/5       30 pts
```

However, the local test does not guarantee the final score you will get from the GitHub classroom. Please make sure your last commit can pass the GitHub workflow.

## NOTICE: DO NOT MODIFY AUTO-GRADING CODE!

The auto-grading relies on the contents within the `.github`, `test` folders, and the `test_main.py` file. These files should not be altered. You will not be able to modify your commit history as all changes are saved permanently in the GitHub classroom records.

Attempts to bypass the auto-grading system, either by altering grading scripts or creating functions intended to manipulate the test cases, will be recorded. Teaching assistants can easily verify any changes to these files.

Modifying auto-grading code or attempting to manipulate it will result in a zero mark for the assignment.