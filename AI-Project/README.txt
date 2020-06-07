The following doc describes the files in the project.

CSVs :

- APA, GL, nasdq, SP500, VIX, Treasury_Yield_10_Years : financial data of this symbols. taken from "Yahoo! finance".
- MULTPL-SHILLER_PE_RATIO_MONTH, MULTPL-SP500_DIV_MONTH, MULTPL-SP500_DIV_YIELD_MONTH, MULTPL-SP500_EARNINGS_YIELD_MONTH, 
  MULTPL-SP500_INFLADJ_MONTH, MULTPL-SP500_PE_RATIO_MONTH : financial dta of this symbolc. taken from YCharts.com
- combiner, combiner_final_lstm_to_predict, combiner_full, combiner_full_noLabel, combiner_full_noLabelnoCrisis,
  combiner_noLookback, combiner_noMonth, combiner_noMonth_noLookback : different permutations of the features, created by our csv generator.
- strategies : documentation of the financial strategies and its results.
  
PYs :
- main : 
	- CSV creator - create CSV's that will be used as an input to the algorithms.
	- Part #1 : test SVR, random forest, KNN with feature selection.
	- Part #2 : LSTM first experiments - choose features.
	- Part #3 : LSTM - evaluate Hyper Parameters.
	- Part #4 : predict according to the chosen LSTM configuration.
- lstm : contains the code for the LSTM algorithms.
- data_combiner_day : extract daily features from "Yahoo! finance" CSV's.
- data_combiner_month : extract monthly features from YCharts CSV's.
- data_extractor : extract crisis dates based on data history.
- feature_selection : functions of selecting the best k features out of a group of features.
- test : contains the code for the algorithms - SVR, random forest, KNN.

TXTs :
- featureselectionRes : results of experiments - SVR, random forest, KNN.
- param_lstm : LSTM's parameters tuning results.
- libs : python libraries under usage.