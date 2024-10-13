To create the composite using the proposed fuzzy AHP-based approach, please run the code, select the path of the initial dataset and the target variable, in which the fuzzy-AHP method is centered around.
Please note, that the code is also designed to handle the alphanumeric variables and identify if any alphanumeric variable refers to classes or is a generic, commentary-like variable.
Specifically, the code tests the different values identified in all of the alphanumeric variables. If the different values within a variable are finite (i.e. 10 different values), the code handles these variables as class-related, thus it reconfigurates them in a numeric variables by assigning discrete values. 
To better perform this operator, please assess all of the alphanumeric variables in your dataset and find out which of them are class-related, as well as the distinct values exist. Then, go to the corresponding part of the code and callibrate the threshold, to be alligned with your specific problem.

After completing the process of generating the composite variables, you can run the k-NN, or any other ML-based algotirthm in both initial and composite datasets and compare the results obtained.
