# Study on prediction of outcome of Premier League using Machine Learning

This project was developed for the Machine Learning course of the University of Crete (CS-577). The goal is to try various algorithms on a public dataset provided by http://www.football-data.co.uk/ (Seasons 2007-2017).



### Prerequisites

After cloning the repository you will need to download and install Python 3.6.3 or preferably Anaconda 3.6 version https://www.anaconda.com/download/.


### A few things about the implementation

First of, in order to create a dataset that can be used for prediction we need to make sure that each time we are relying on past matches only (you can't predict a match using the number of fouls comitted). The approach taken here is to create
individual datasets for each team. Which means that we select a team and look to it's 'K' most recent matches and measure performance for both home and away teams. This feature extraction method is refered as "K-Team Statistics" in this project.



## Running the tests

Explain how to run the automated tests for this system

### Break down into end to end tests

Explain what these tests test and why

```
Give an example
```

### And coding style tests

Explain what these tests test and why

```
Give an example
```

## Deployment

Add additional notes about how to deploy this on a live system

## Built With

* [Dropwizard](http://www.dropwizard.io/1.0.2/docs/) - The web framework used
* [Maven](https://maven.apache.org/) - Dependency Management
* [ROME](https://rometools.github.io/rome/) - Used to generate RSS Feeds

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## Authors

* **Billie Thompson** - *Initial work* - [PurpleBooth](https://github.com/PurpleBooth)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone who's code was used
* Inspiration
* etc
