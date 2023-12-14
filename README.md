# Model free localization with Deep Neural Architectures by means of an underwater WSN

## Introduction

This code implements the results published in Parras, J., Zazo, S, Pérez-Álvarez, I. A., & Sanz González, J. L. (2019). Model free localization with Deep Neural Architectures by means of an underwater WSN. Sensors, 19(16), 3530. [DOI](https://doi.org/10.3390/s19163530).

## Launch

This project requires Python 3.6. To run this project, create a `virtualenv` (recomended) and then install the requirements as:

```
$ pip install -r requirements.txt
```

To obtain the results in the paper, first run:
```
$ python channel_generator_main.py
```
and then, obtain the plots with:
```
$ python channel_generator_results.py
```
Note that, with the default parameters, a large number of files will be generated (~8000).
