# WaterLevelPrediction

During the First and the Second Word War, the water level at some parts of the river was not measured. Civil engineers need this information in order to perform some analyzes and plan the construction of dams, thus this information needs to be replaced. The data goes abck to 1983, until year 2020. Critical years are usually 1990 - 2000. In order to do that I would do few approaches:

- **First approach**: Linear regression using tensor flow. To predict the water level at one station I used data on water level at the station that was upstream and that had data and data on water level one day later at the next station downstream, and I also used the data which is the month in which I need to predict value. My learning rate here was: 0.000005 and my batch size was 10 and my number of epochs was 30. The mean squared value error that I usually got on the test data was about 0.45. 
