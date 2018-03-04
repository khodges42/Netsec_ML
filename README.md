# Server Health ML

Demo/Proof of concept for predicting server load using Apache Spark ML and Scala. 

util/log-gen.py will generate some csv logs to train on, however this data is randomized and won't yield any worthwhile results.


## Usage

This is a demo, but if given real data it yields a model that can be worked with. Currently the label is based upon the cpu load average, but this can be modified to other things, as shown below.

```
// Use memory utilization as the label
data.select(data("memutil")
    .as("label"),$"hour",$"load",$"netutil")
```

Additionally datasets can be filtered by modifying the select on the dataframe, as shown below

```
// Train only on data from the abstract host taken at noon
data.filter("hostname = 'abstract' AND hour = 12")
    .select(data("load")
    .as("label"),$"hour",$"memutil",$"netutil")
```
