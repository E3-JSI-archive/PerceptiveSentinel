# EOQMiner

## Prerequisites
* NodeJS installed (able to run QMiner) [[https://qminer.github.io/setup/]]

## PythonQMinerBridge
This class takes care about connectivity between Python based scripts (eo-learn) and Node.JS based scripts (QMiner).

## Roadmap
* per-time series feature extraction
* batch feature extraction
* implement additional measures in QMiner:
  * skewness [[https://en.wikipedia.org/wiki/Skewness]]
  * kurtosis \lambda [[https://en.wikipedia.org/wiki/Kurtosis]]
  * RMS [[https://en.wikipedia.org/wiki/Root_mean_square]]
  * all based on: Mohamed-Rafik Bouguelia, Alexander Karlsson, Sepideh Pashami, Sławomir Nowaczyk, Anders Holst, Mode tracking using multiple data streams, Information Fusion, Volume 43, 2018, Pages 33-46, ISSN 1566-2535.