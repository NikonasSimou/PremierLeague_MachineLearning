# Study on outcome prediction of Premier League using Machine Learning

This project was developed for the Machine Learning course of the University of Crete (CS-577). The goal is to try various algorithms on a public dataset provided by http://www.football-data.co.uk/ (Seasons 2007-2017).
Included is the report of the project too.



### Prerequisites

After cloning the repository you will need to download and install Python 3.6.3 or preferably Anaconda 3.6 version https://www.anaconda.com/download/.<br /> If Anaconda is not installed the Libraries needed are: <br />
-Scipy https://www.scipy.org/ <br />
-Numpy http://www.numpy.org/ <br />
-Pandas https://pandas.pydata.org/ <br />
-Matplotlib https://matplotlib.org/ <br />


### A few things about the implementation

First of, in order to create a dataset that can be used for prediction we need to make sure that each time we are relying on past matches only (you can't predict a match using the number of fouls comitted). The approach taken here is to create
individual datasets for each team. Which means that we select a team and look to it's 'K' most recent matches and measure performance for both home and away teams. This feature extraction method is refered as "K-Team Statistics" in this project.
Example of creating one row of dataset for team 'Arsenal'
![arsenal_kdata](https://user-images.githubusercontent.com/36821727/36682778-ab6561fa-1b24-11e8-9439-1ac63515a156.png)


Another important aspect is the use of time series cross validation in order to adress the time dependencies of the dataset (you can't use future matches to predict current ones) and the use of nested (time series) cross validation in order to perform Hyperparameter Tuning and Feature Selction without being biased (if you use the same data for tuning/feature selection and for accuracy estimation the results will be optimistic).

The classifiers used are Support Vector Machines, Random Forests, K-Nearest Neighbors.

For more information please read the comments in the code and the Report of the project.

## Running the tests

Run the [test.py] file.


## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details


