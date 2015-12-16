# Practical Machine Learning Project
J. Hartsfield  
December 10, 2015  
Can we predict whether or not someone has lifted weights correctly using data from accelerometers on the belt, forearm, arm and dumbbell?  We use data gathered on 6 participants who were asked to perform barbell lifts correctly and incorrectly in 5 different ways.  The results of their exercises were recorded and will be used to predict whether or not other lifting is done correctly or not.  If incorrect, the basic type of incorrect behavior will be predicted.  

More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset).

## Load and pre-process data

We load the libraries we need and download the datasets from the internet.  After reading in the datasets, we eliminate the first 7 columns since they are not applicable to predicting the result. We also eliminate all columns that are mostly NA or empty. We then scale all columns except the last one ( classe which is what we want to predict).

We analyze the covariates to see if we any have near zero variance.  They do not, so there is no need to remove columns on that basis. We then divide the training set into 4 folds (mutually exclusive random subsets of the observations).  Each fold is then divided into a training dataset and a testing dataset. 

```
##                      freqRatio percentUnique zeroVar   nzv
## roll_belt             1.101904     6.7781062   FALSE FALSE
## pitch_belt            1.036082     9.3772296   FALSE FALSE
## yaw_belt              1.058480     9.9734991   FALSE FALSE
## total_accel_belt      1.063160     0.1477933   FALSE FALSE
## gyros_belt_x          1.058651     0.7134849   FALSE FALSE
## gyros_belt_y          1.144000     0.3516461   FALSE FALSE
## gyros_belt_z          1.066214     0.8612782   FALSE FALSE
## accel_belt_x          1.055412     0.8357966   FALSE FALSE
## accel_belt_y          1.113725     0.7287738   FALSE FALSE
## accel_belt_z          1.078767     1.5237998   FALSE FALSE
## magnet_belt_x         1.090141     1.6664968   FALSE FALSE
## magnet_belt_y         1.099688     1.5187035   FALSE FALSE
## magnet_belt_z         1.006369     2.3290184   FALSE FALSE
## roll_arm             52.338462    13.5256345   FALSE FALSE
## pitch_arm            87.256410    15.7323412   FALSE FALSE
## yaw_arm              33.029126    14.6570176   FALSE FALSE
## total_accel_arm       1.024526     0.3363572   FALSE FALSE
## gyros_arm_x           1.015504     3.2769341   FALSE FALSE
## gyros_arm_y           1.454369     1.9162165   FALSE FALSE
## gyros_arm_z           1.110687     1.2638875   FALSE FALSE
## accel_arm_x           1.017341     3.9598410   FALSE FALSE
## accel_arm_y           1.140187     2.7367241   FALSE FALSE
## accel_arm_z           1.128000     4.0362858   FALSE FALSE
## magnet_arm_x          1.000000     6.8239731   FALSE FALSE
## magnet_arm_y          1.056818     4.4439914   FALSE FALSE
## magnet_arm_z          1.036364     6.4468454   FALSE FALSE
## roll_dumbbell         1.022388    84.2065029   FALSE FALSE
## pitch_dumbbell        2.277372    81.7449801   FALSE FALSE
## yaw_dumbbell          1.132231    83.4828254   FALSE FALSE
## total_accel_dumbbell  1.072634     0.2191418   FALSE FALSE
## gyros_dumbbell_x      1.003268     1.2282132   FALSE FALSE
## gyros_dumbbell_y      1.264957     1.4167771   FALSE FALSE
## gyros_dumbbell_z      1.060100     1.0498420   FALSE FALSE
## accel_dumbbell_x      1.018018     2.1659362   FALSE FALSE
## accel_dumbbell_y      1.053061     2.3748853   FALSE FALSE
## accel_dumbbell_z      1.133333     2.0894914   FALSE FALSE
## magnet_dumbbell_x     1.098266     5.7486495   FALSE FALSE
## magnet_dumbbell_y     1.197740     4.3012945   FALSE FALSE
## magnet_dumbbell_z     1.020833     3.4451126   FALSE FALSE
## roll_forearm         11.589286    11.0895933   FALSE FALSE
## pitch_forearm        65.983051    14.8557741   FALSE FALSE
## yaw_forearm          15.322835    10.1467740   FALSE FALSE
## total_accel_forearm   1.128928     0.3567424   FALSE FALSE
## gyros_forearm_x       1.059273     1.5187035   FALSE FALSE
## gyros_forearm_y       1.036554     3.7763735   FALSE FALSE
## gyros_forearm_z       1.122917     1.5645704   FALSE FALSE
## accel_forearm_x       1.126437     4.0464784   FALSE FALSE
## accel_forearm_y       1.059406     5.1116094   FALSE FALSE
## accel_forearm_z       1.006250     2.9558659   FALSE FALSE
## magnet_forearm_x      1.012346     7.7667924   FALSE FALSE
## magnet_forearm_y      1.246914     9.5403119   FALSE FALSE
## magnet_forearm_z      1.000000     8.5771073   FALSE FALSE
## classe                1.469581     0.0254816   FALSE FALSE
```

## Examine different methods to check effectiveness

First, we predict with trees on the first training set (training1) and look at the tree plot.  

![](index_files/figure-html/tree-1.png) 

```
## Accuracy 
## 0.493533
```

```
##            
## treePredict   A   B   C   D   E
##           A 373 136 113 108  39
##           B   8  83   6  46  42
##           C  34  65 137  87  57
##           D   0   0   0   0   0
##           E   3   0   0   0 132
```

We see that this method only gave us a 49.3533016% overall accuracy.


```
##  Accuracy 
## 0.9673469
```

```
##          
## rfPredict   A   B   C   D   E
##         A 416  13   4   0   0
##         B   2 266   7   0   2
##         C   0   6 243   8   2
##         D   0   0   2 233   2
##         E   0   0   0   0 264
```
We use random forest on the second training set (training2) and see that this gives us a 96.7346939% overall accuracy. Clearly this is much better than the tree method.


```
##  Accuracy 
## 0.9455782
```

```
##             
## boostPredict   A   B   C   D   E
##            A 406   8   2   0   2
##            B   8 263  13   2   1
##            C   1  10 235  12   4
##            D   1   0   5 224   2
##            E   2   3   1   3 262
```
Boosting on training set 3 (training3) gives us a  94.5578231% overall accuracy.



```
##  Accuracy 
## 0.7025187
```

```
##           
## ldaPredict   A   B   C   D   E
##          A 343  41  27  16  13
##          B   8 178  31  15  43
##          C  37  43 166  22  22
##          D  29  10  29 178  25
##          E   1  12   3  10 167
```
Lastly, we try linear discrimant analysis on training set 4 (training4) and get a 70.251872% overall accuracy.

## Choose a model, make predictions, estimate error

Based on the results we achieved for the different models, we choose to use the random forest method to predict our results.  We will also use the boosting as a second choice to see how it differs on the test data from the random forest method.  


Based on the above, we estimate our out-of-sample error to be 1 - the accuracy of the random forest method or 0.0326531. 

Where do the two models produce different responses? What are the predictions for our test dataset from the two models?


```
##             rfPredict
## boostPredict   A   B   C   D   E
##            A 408   7   0   1   2
##            B  21 247  16   2   1
##            C   1  18 226  13   4
##            D   1   0  13 216   2
##            E   2   5   4   5 255
```

```
##  [1] E A A E A E D D A E A C B A E D E B E B
## Levels: A B C D E
```

```
##  [1] E B A E C B D D A E A B B A E B E B E B
## Levels: A B C D E
```
