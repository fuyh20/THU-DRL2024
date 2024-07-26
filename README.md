[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/wtr50jJI)
# Installation

This assignment consists of three tasks. The detailed requirements of each task can be found in `hw4.pdf`.

Please fill in the blanks inside the "YOUR IMPLEMENTATION HERE" comments, any changes outside the area might be **ignored**.

## Getting Started

You can use exactly the same environment as in hw3, without any modification. If you encounter any trouble creating the environment, please let us know :-)

## Submitting the Results

After finishing your homework, please remember to commit the changes and push them to the GitHub repo.

TAs have created a GitHub workflow to evaluate your code automatically, and a reference score of your homework can be found on the Actions page of your repository. (The total points of the auto-test is 30, and 70 points are assigned to the report.)

You can also use the following command to quickly test your code locally.

```
python test_main.py
```

If your code is correct, you will see the following output:

```bash
Test Name              Result    Score
---------------------  --------  -------
test_get_policy_loss   Passed    10 pts
test_get_value_loss    Passed    10 pts
test_get_entropy_loss  Passed    10 pts
---------------------  --------  -------
Total                  3/3       30 pts
```

However, the local test does not guarantee the final score you will get from the GitHub classroom. Please make sure your last commit can pass the GitHub workflow.

## NOTICE: DO NOT MODIFY AUTO-GRADING CODE!

The auto-grading relies on the contents within the `.github`, `test` folders, and the `test_main.py` file. These files should not be altered. You will not be able to modify your commit history as all changes are saved permanently in the GitHub classroom records.

Attempts to bypass the auto-grading system, either by altering grading scripts or creating functions intended to manipulate the test cases, will be recorded. Teaching assistants can easily verify any changes to these files.

Modifying auto-grading code or attempting to manipulate it will result in a zero mark for the assignment.