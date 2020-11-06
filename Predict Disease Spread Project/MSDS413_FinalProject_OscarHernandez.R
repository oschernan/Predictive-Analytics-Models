#MSDS 413 Final Project
#Oscar Hernandez


#Load necessary libraries
library(DataExplorer)
library(forecast)
library(zoo)
library(plyr)
library(ggplot2)
library(stargazer)
library(fpp2)
library(urca)
library(tseries)

#Load provided data files  
train.df.nolabel <- read.csv('dengue_features_train.csv')
train.df.labels <- read.csv('dengue_labels_train.csv')
submission.format <- read.csv('submission_format.csv')

#Combine to create train.df
train.df <- merge(features.df, train.df.labels, c('city', 'year', 'weekofyear'))
rm(train.df.labels)

#Load test data
test.df <- read.csv('dengue_features_test.csv')

#Review the dataframes
#TRAIN
summary(train.df.nolabel)
str(train.df.nolabel)

#TEST
summary(test.df)
str(test.df)

###################################DATA PREPARATION 
#View how many times 53 came up
table(train.df.nolabel$weekofyear) # 5 times 
#Seems that 53 was misclassified but probably won't have to change it 

#Light analysis; checks missing values on train and test data sets
plot_intro(train.df.nolabel)
plot_intro(test.df)

#Combine the both data sets 
temp.df <- rbind(train.df.nolabel, test.df)

#Impute missing values
#Take the last value to replace NaN
df<- na.locf(temp.df, fromLast = TRUE)  
##############
out.path <- 'C:\\Users\\herna_000\\Desktop\\MSDS 413\\';

file.name <- 'Table1.html';
stargazer(df, type=c('html'),out=paste(out.path,file.name,sep=''),
          title=c('Table 1: Summary Statistics for Default Variables'),
          align=TRUE, digits=2, digits.extra=2, initial.zero=TRUE, median=TRUE)
#############
#Add column for label
df$total_cases <- 0

#split data set into San Juan and Iquitos train set 
san_juan.train <- df[1:936,]
iquitos.train <- df[937:1456,]

#add label values for training sets
san_juan.train$total_cases <- train.df.labels[1:936,4]
iquitos.train$total_cases <- train.df.labels[937:1456, 4]

#split data into San Juan and Iquitos test set
sanjuan.test <- df[1457:1716, -25]
iquitos.test <- df[1717:1872, -25]

#Turn data sets into ts objects
#Weekly frequency 
sanjuan.train.ts <- ts(san_juan.train$total_cases, frequency = 52, start = c(1990,04,30))
iquitos.train.ts <- ts(iquitos.train$total_cases, frequency =52, start=c(2000,07,01))

#Features to be used for models
features <- c('station_avg_temp_c', 'station_diur_temp_rng_c', 'station_max_temp_c', 'station_min_temp_c', 'station_precip_mm')


###################################EDA
#Plot the data for train sets

autoplot(sanjuan.train.ts, main='San Juan: Dengue Fever Plot', ylab= 'Total Cases (Weekly)')
autoplot(iquitos.train.ts, main='Iquitos: Dengue Fever Plot', ylab= 'Total Cases (Weekly)')


#ACF and PACF Plot
ggAcf(sanjuan.train.ts)
ggpacf(sanjuan.train.ts)

ggacf(iquitos.train.ts)
ggpacf(iquitos.train.ts)

par(mfrow=c(2,1))
ggAcf(sanjuan.train.ts)
ggAcf(iquitos.train.ts)
par(mfrow=c(1,1))

#Conduct Unit Root Test
sanjuan.train.ts %>% ur.kpss() %>% summary()
iquitos.train.ts %>% ur.kpss() %>% summary()

#Plot lagplot of total_cases
gglagplot(sanjuan.train.ts)
gglagplot(iquitos.train.ts)

#compute Augmented Dickey-Fuller Test for stationarity
adf.test(sanjuan.train.ts)
adf.test(iquitos.train.ts)

#Seasonality plots
ggseasonplot(sanjuan.train.ts) + ggtitle('Seasonal Plot: San Juan')
ggseasonplot(iquitos.train.ts) + ggtitle('Seasonal Plot: Iquitos')


###################################Build Models 
#STL  Seasonal and Trend decomposition with Loess
#INCORRECTLY created variable that says ETS...model is STL 
ets.model.sj <- mstl(sanjuan.train.ts)
ets.model.iq <- mstl(iquitos.train.ts)

#Neural Network w/ Regressors (NOAA's GHCN daily climate data weather station measurements variables)
nn.model.sj <- nnetar(sanjuan.train.ts, size =10, repeats=4, xreg = san_juan.train[,features])
nn.model.iq <- nnetar(iquitos.train.ts, size =10, repeats=4, xreg = iquitos.train[,features])

#ARIMA w/ Regressors (NOAA's GHCN daily climate data weather station measurements variables)
arima.model.sj <- auto.arima(sanjuan.train.ts, xreg = san_juan.train[,features])
arima.model.iq <- auto.arima(iquitos.train.ts, xreg = iquitos.train[,features])


###################################Make Forecasts
# San Juan is 260 weeks and Iquitos is 156 weeks assuming 52 weeks in a year
#NEED TO SAVE EACH FORECAST INTO IT'S OWN VARIABLE 

ets.sj.fcast <- forecast(ets.model.sj, h=260)
ets.iq.fcast <- forecast(ets.model.iq, h=156)

nn.sj.fcast <- forecast(nn.model.sj,xreg=sanjuan.test[,features], h=260)
nn.iq.fcast <- forecast(nn.model.iq, xreg=iquitos.test[,features],h=156)

arima.sj.fcast <- forecast(arima.model.sj, xreg=sanjuan.test[,features], h=260)
arima.iq.fcast <- forecast(arima.model.iq, xreg=iquitos.test[,features], h=156)

autoplot(ets.sj.fcast,  main='San Juan: Forecasts from STL + ETS(A,Ad,N)', xlab = 'Time', ylab='Total Cases (Weekly)' )
autoplot(ets.iq.fcast, main='Iquitos: Forecasts from STL + ETS(A,N,N)', xlab = 'Time', ylab='Total Cases (Weekly)' )

autoplot(nn.sj.fcast, main='San Juan: Forecasts from NNAR(14,1,10)[52]', xlab = 'Time', ylab='Total Cases (Weekly)')
autoplot(nn.iq.fcast,main='Iquitos: Forecasts from NNAR(5,1,10)[52]',  xlab = 'Time', ylab='Total Cases (Weekly)')

autoplot(arima.sj.fcast, main='San Juan: Forecasts from Regression w/ ARIMA(1,1,1) errors',xlab = 'Time', ylab='Total Cases (Weekly)')
autoplot(arima.iq.fcast, main='Iquitos: Forecasts from Regression w/ ARIMA(1,0,4)(0,0,1)[52] errors',
         xlab = 'Time', ylab='Total Cases (Weekly)')

###################################Compare performance on MAE and Check Residuals 
#Calculate performance metrics
accuracy(ets.sj.fcast)
accuracy(ets.iq.fcast)

accuracy(nn.sj.fcast)
accuracy(nn.iq.fcast)

accuracy(arima.sj.fcast)
accuracy(arima.iq.fcast)

#Check residuals
checkresiduals(ets.sj.fcast)
checkresiduals(ets.iq.fcast)

checkresiduals(nn.sj.fcast)
checkresiduals(nn.iq.fcast)

checkresiduals(arima.sj.fcast)
checkresiduals(arima.iq.fcast)


######################################Output submission file using all models 
sj_submission <-  forecast(nn.model.sj,xreg=sanjuan.test[,features], h=260)
iq_submission <- forecast(nn.model.iq, xreg=iquitos.test[,features],h=156)

nn.sj.output <- data.frame(submission.format[1:260,-4], total_cases = round(sj_submission$mean))
nn.sj.output[,4] <- as.numeric(nn.sj.output[,4])
nn.iq.output <- data.frame(submission.format[261:416,-4], total_cases =round(iq_submission$mean))
nn.iq.output[,4] <- as.numeric(nn.iq.output[,4])

nn.total.output <- rbind(nn.sj.output,nn.iq.output)
write.csv(nn.total.output, file = 'nn_submission.csv', row.names = F)


#STL
stl.sj.output <- data.frame(submission.format[1:260,-4], total_cases = round(ets.sj.fcast$mean))
stl.sj.output[,4] <- as.numeric(stl.sj.output[,4])
stl.iq.output <- data.frame(submission.format[261:416,-4], total_cases =round(ets.iq.fcast$mean))
stl.iq.output[,4] <- as.numeric(stl.iq.output[,4])

stl.total.output <- rbind(stl.sj.output,stl.iq.output)
write.csv(stl.total.output, file = 'stl_submission.csv', row.names = F)

#ARIMA

arima.sj.output <- data.frame(submission.format[1:260,-4], total_cases = round(arima.sj.fcast$mean))
arima.sj.output[,4] <- as.numeric(arima.sj.output[,4])
arima.iq.output <- data.frame(submission.format[261:416,-4], total_cases =round(arima.iq.fcast$mean))
arima.iq.output[,4] <- as.numeric(arima.iq.output[,4])

arima.total.output <- rbind(arima.sj.output,arima.iq.output)
write.csv(arima.total.output, file = 'arima_submission.csv', row.names = F)




