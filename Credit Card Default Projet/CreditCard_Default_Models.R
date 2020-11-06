# Oscar Hernandez
# 04.1.2019
# read_credit_card_default_data.R


#####################################################################################
# Load Data & Initial Review
#####################################################################################
library(stargazer)
library(woeBinning)
library(plyr)
library(dplyr)
library(DataExplorer)
library(ggplot2)
library(rpart)
library(rattle)
library(rpart.plot)
library(RColorBrewer)
library(OneR)
library(caret)
library(randomForest)
library(e1071)
library(glm2)
library(gbm)
library(pROC)
library(MASS)

my.path <- 'C:\\Users\\herna_000\\Desktop\\MSDS 498\\MSDS 498 Project Data\\'
my.file <- paste(my.path,'credit_card_default.RData',sep='')

# Read the RData object using readRDS()
credit_card_default <- readRDS(my.file)

# Show dataframe structure
str(credit_card_default)

# Show descriptive statistics for the entire dataframe 
summary(credit_card_default)

# Show how many observations will be in the train, test and validate dataframes 
table(credit_card_default$data.group)

#####################################################################################
# Data Quality Check 
#####################################################################################

out.path <- 'C:\\Users\\herna_000\\Desktop\\MSDS 498\\';

file.name <- 'Table1.html';
stargazer(credit_card_default, type=c('html'),out=paste(out.path,file.name,sep=''),
          title=c('Table 1: Summary Statistics for Default Variables'),
          align=TRUE, digits=2, digits.extra=2, initial.zero=TRUE, median=TRUE)

#Change name of PAY_0 to PAY_1
colnames(credit_card_default)[colnames(credit_card_default)=='PAY_0'] <- 'PAY_1'


#Fix EDUCATION variable
#Turn values 0, 5, 6 into other 
file.name <- 'Table3.html';

edu.table <- as.data.frame(table(credit_card_default$EDUCATION))
edu.table
colnames(edu.table) <- c('0', '1','2', '3', '4', '5', '6',);

stargazer(edu.table, type=c('html'),out=paste(out.path,file.name,sep=''),
          title=c('Table 3: Frequency Table of EDUCATION'),
          align=TRUE, digits=2, digits.extra=2, initial.zero=TRUE,
          summary=FALSE) 

credit_card_default$EDUCATION[credit_card_default$EDUCATION==0] <- 4
credit_card_default$EDUCATION[credit_card_default$EDUCATION==5] <- 4
credit_card_default$EDUCATION[credit_card_default$EDUCATION==6] <- 4

table(credit_card_default$EDUCATION)

#Fix MARRIAGE variable
#Turn values 0 into other
credit_card_default$MARRIAGE[credit_card_default$MARRIAGE==0] <- 3

table(credit_card_default$MARRIAGE)

#Fix the PAY_X variables
credit_card_default$PAY_1[credit_card_default$PAY_1==0] <- 1
credit_card_default$PAY_1[credit_card_default$PAY_1==-2] <- -1

table(credit_card_default$PAY_1)

credit_card_default$PAY_2[credit_card_default$PAY_2==0] <- 1
credit_card_default$PAY_2[credit_card_default$PAY_2==-2] <- -1

table(credit_card_default$PAY_2)

credit_card_default$PAY_3[credit_card_default$PAY_3==0] <- 1
credit_card_default$PAY_3[credit_card_default$PAY_3==-2] <- -1

table(credit_card_default$PAY_3)

credit_card_default$PAY_4[credit_card_default$PAY_4==0] <- 1
credit_card_default$PAY_4[credit_card_default$PAY_4==-2] <- -1

table(credit_card_default$PAY_4)

credit_card_default$PAY_5[credit_card_default$PAY_5==0] <- 1
credit_card_default$PAY_5[credit_card_default$PAY_5==-2] <- -1

table(credit_card_default$PAY_5)

credit_card_default$PAY_6[credit_card_default$PAY_6==0] <- 1
credit_card_default$PAY_6[credit_card_default$PAY_6==-2] <- -1

table(credit_card_default$PAY_6)


# Use data.group variable to create train, test and validate dataframes
# Split the data prior to feature engineering to avoid data leakage 
train.df <- subset(credit_card_default, data.group==1)
test.df <- subset(credit_card_default, data.group==2)
validate.df <- subset(credit_card_default, data.group==3)


#####################################################################################
# Feature Engineering 
#####################################################################################

summary(train.df)
str(train.df)

#####################Discretize AGE variable
#Start with AGE
#Use WOE
age.tree <- woe.tree.binning(df=train.df,target.var=c('DEFAULT'),pred.var=c('AGE'))

# WOE plot for age bins;
woe.binning.plot(age.tree)
# Note that we got different bins;

# Score bins on data frame;
tree.df <- woe.binning.deploy(df=train.df,binning=age.tree)
head(tree.df)
table(tree.df$AGE.binned)

# See the WOE Binning Table
woe.binning.table(age.tree)

#Add the discretized version of AGE
train.df <- tree.df

#Change the names of the factors
train.df$Age_Bin <- revalue(train.df$AGE.binned, c("(-Inf,24]"="Age_21_24", "(24,25]"="Age_25", "(25,34]"="Age_26_34",
                                                   "(34,48]"="Age_35_48","(48, Inf]"="Age_49_79"))
train.df$Age_Bin <- droplevels(train.df$Age_Bin)

#Drop certain variables to create new dataframe
train.df <- subset(train.df, select = -c(u, data.group, AGE.binned, train, test, validate, ID))

#Change column name 
colnames(train.df)[colnames(train.df)=="Age_Bin"] <- "Age"

########################Create Average Bill Amount (Avg_Bill_Amt)
train.df <- within(train.df, Avg_Bill_Amt <- (BILL_AMT1 + BILL_AMT2 + BILL_AMT3 +BILL_AMT4 +BILL_AMT5 +BILL_AMT6)/6)

########################Create Average Payment Amount (Avg_Pmt_Amt)
train.df <- within(train.df, Avg_Pmt_Amt <- (PAY_AMT1 + PAY_AMT2 + PAY_AMT3 +PAY_AMT4 +PAY_AMT5 +PAY_AMT6)/6)

########################Create Payment Ratio (Pmt_Ratio_X)

train.df <- within(train.df, Pmt_Ratio_2 <- (PAY_AMT1/BILL_AMT2)*100)
train.df <- within(train.df, Pmt_Ratio_3 <- (PAY_AMT2/BILL_AMT3)*100)
train.df <- within(train.df, Pmt_Ratio_4 <- (PAY_AMT3/BILL_AMT4)*100)
train.df <- within(train.df, Pmt_Ratio_5 <- (PAY_AMT4/BILL_AMT5)*100)
train.df <- within(train.df, Pmt_Ratio_6 <- (PAY_AMT5/BILL_AMT6)*100)

summary(train.df$Pmt_Ratio_6) 
summary(train.df$Pmt_Ratio_5)        
summary(train.df$Pmt_Ratio_4)
summary(train.df$Pmt_Ratio_3)
summary(train.df$Pmt_Ratio_2)

summary(train.df)
str(train.df)

#Turn all 0 PMT and negative BILL to 101 for all Pmt_Ratio_X columns
train.df <- within(train.df, Pmt_Ratio_2[Pmt_Ratio_2 == 0 & PAY_AMT1 == 0 & BILL_AMT2 < 0] <- 101)
train.df <- within(train.df, Pmt_Ratio_3[Pmt_Ratio_3 == 0 & PAY_AMT2 == 0 & BILL_AMT3 < 0] <- 101)
train.df <- within(train.df, Pmt_Ratio_4[Pmt_Ratio_4 == 0 & PAY_AMT3 == 0 & BILL_AMT4 < 0] <- 101)
train.df <- within(train.df, Pmt_Ratio_5[Pmt_Ratio_5 == 0 & PAY_AMT4 == 0 & BILL_AMT5 < 0] <- 101)
train.df <- within(train.df, Pmt_Ratio_6[Pmt_Ratio_6 == 0 & PAY_AMT5 == 0 & BILL_AMT6 < 0] <- 101)

#Turn all NaN's into 100 
train.df[is.na(train.df)] <- 100

#Turn all negative numbers to 101 
train.df <- within(train.df, Pmt_Ratio_2[Pmt_Ratio_2 < 0] <- 101)
train.df <- within(train.df, Pmt_Ratio_3[Pmt_Ratio_3 < 0] <- 101)
train.df <- within(train.df, Pmt_Ratio_4[Pmt_Ratio_4 < 0] <- 101)
train.df <- within(train.df, Pmt_Ratio_5[Pmt_Ratio_5 < 0] <- 101)
train.df <- within(train.df, Pmt_Ratio_6[Pmt_Ratio_6 < 0] <- 101)

#Turn all Inf values to 101 
is.na(train.df)<-sapply(train.df, is.infinite)
train.df[is.na(train.df)]<-101

#Winsorize all values greater than 101 into 101
train.df <- within(train.df, Pmt_Ratio_2[Pmt_Ratio_2 > 101] <- 101)
train.df <- within(train.df, Pmt_Ratio_3[Pmt_Ratio_3 > 101] <- 101)
train.df <- within(train.df, Pmt_Ratio_4[Pmt_Ratio_4 > 101] <- 101)
train.df <- within(train.df, Pmt_Ratio_5[Pmt_Ratio_5 > 101] <- 101)
train.df <- within(train.df, Pmt_Ratio_6[Pmt_Ratio_6 > 101] <- 101)

################################### Average Payment Ratio (Avg_Pmt_Ratio)
train.df <- within(train.df, Avg_Pmt_Ratio <- (Pmt_Ratio_2 + Pmt_Ratio_3 +Pmt_Ratio_4 +Pmt_Ratio_5 +Pmt_Ratio_6 )/5)

################################### Utilization (Util)
train.df <- within(train.df, Util_1 <-  (BILL_AMT1/LIMIT_BAL)*100)
train.df <- within(train.df, Util_2 <-  (BILL_AMT2/LIMIT_BAL)*100)
train.df <- within(train.df, Util_3 <-  (BILL_AMT3/LIMIT_BAL)*100)
train.df <- within(train.df, Util_4 <-  (BILL_AMT4/LIMIT_BAL)*100)
train.df <- within(train.df, Util_5 <-  (BILL_AMT5/LIMIT_BAL)*100)
train.df <- within(train.df, Util_6 <-  (BILL_AMT6/LIMIT_BAL)*100)

summary(train.df$Util_1)
summary(train.df$Util_2)
summary(train.df$Util_3)
summary(train.df$Util_4)
summary(train.df$Util_5)
summary(train.df$Util_6)

#Convert negative values into 0
#Means that the customer was rewarded for not utilizing their balance
train.df <- within(train.df, Util_1[Util_1 < 0] <- 0)
train.df <- within(train.df, Util_2[Util_2 < 0] <- 0)
train.df <- within(train.df, Util_3[Util_3 < 0] <- 0)
train.df <- within(train.df, Util_4[Util_4 < 0] <- 0)
train.df <- within(train.df, Util_5[Util_5 < 0] <- 0)
train.df <- within(train.df, Util_6[Util_6 < 0] <- 0)

#Convert anything greater than 100 into 100
train.df <- within(train.df, Util_1[Util_1 > 100] <- 100)
train.df <- within(train.df, Util_2[Util_2 > 100] <- 100)
train.df <- within(train.df, Util_3[Util_3 > 100] <- 100)
train.df <- within(train.df, Util_4[Util_4 > 100] <- 100)
train.df <- within(train.df, Util_5[Util_5 > 100] <- 100)
train.df <- within(train.df, Util_6[Util_6 > 100] <- 100)

summary(train.df)
################################### Average Utilization (Avg_Util) 
train.df <- within(train.df, Avg_Util <- (Util_1 + Util_2 + Util_3 + Util_4 + Util_5 + Util_6)/6)

################################### Balance Growth Over 6 Months (Bal_Growth_6mo)
#(1=Yes, 0 = No)
train.df <- within(train.df, Bal_Growth_6mo <- BILL_AMT1 - BILL_AMT6)

train.df <- within(train.df, Bal_Growth_6mo[Bal_Growth_6mo > 0 & BILL_AMT1 > BILL_AMT6] <- 1)
train.df <- within(train.df, Bal_Growth_6mo[Bal_Growth_6mo == 0] <- 0)
train.df <- within(train.df, Bal_Growth_6mo[Bal_Growth_6mo < 0 & BILL_AMT1 == 0 & BILL_AMT6>0] <- 0)
train.df <- within(train.df, Bal_Growth_6mo[Bal_Growth_6mo < 0 & BILL_AMT1 <  0 & BILL_AMT6==0] <- 0)
train.df <- within(train.df, Bal_Growth_6mo[Bal_Growth_6mo < 0 & BILL_AMT1 < 0 & BILL_AMT6>0] <- 0)
train.df <- within(train.df, Bal_Growth_6mo[Bal_Growth_6mo < 0 & BILL_AMT1 < 0 & BILL_AMT6<0] <- 0)
train.df <- within(train.df, Bal_Growth_6mo[Bal_Growth_6mo < 0 & BILL_AMT1 < BILL_AMT6] <- 0)

summary(train.df$Bal_Growth_6mo)

table(train.df$Bal_Growth_6mo)

################################### Utilization Growth Over 6 Months (Util_Growth_6mo)
train.df <- within(train.df, Util_Growth_6mo <- Util_1 - Util_6)

summary(train.df$Util_Growth_6mo)

################################## Max Bill Amount (Max_Bill_Amt)
train.df[, "Max_Bill_Amt"] <- apply(train.df[, c('BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6' )], 1, max)


################################## Max Payment Amount (Max_Pmt_Amt)
train.df[, "Max_Pmt_Amt"] <- apply(train.df[, c('PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6' )], 1, max)

################################## Max Delinquency (Max_DLQ)
train.df[, "Max_DLQ"] <- apply(train.df[, c('PAY_1', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6' )], 1, max)

summary(train.df$Max_DLQ)

#####Additional Features####

################################### Payment Ratio Growth Over 6 Months (Pmt_Ratio_Growth_6mo)
train.df <- within(train.df, Pmt_Ratio_Growth_6mo <- Pmt_Ratio_2 - Pmt_Ratio_6)

summary(train.df$Util_Growth_6mo)

################################### Payment Status Change Over 6 Months (Pmt_Status_Change_6mo)
train.df <- within(train.df, Pmt_Status_Change_6mo <- PAY_1 - PAY_6)

summary(train.df$Pmt_Status_Change_6mo)

####################################


#Check to make sure all features have no errors 
#Might need to discretize continuous engineered features and old ones 
summary(train.df)
str(train.df)
#####################################################################################
# Exploratory Data Analsyis 
#####################################################################################

#Condense the training set to a smaller version that will be used to do more feature engineering
#This skinnier training set will include the raw new and old features
options(scipen=999)


#Create EDA dataframe of training set         
train_eda <- subset(train.df, select = c('LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 'Age', 'Avg_Bill_Amt',
                                         'Avg_Pmt_Amt', 'Pmt_Ratio_2', 'Pmt_Ratio_3', 'Pmt_Ratio_4', 'Pmt_Ratio_5', 
                                         'Pmt_Ratio_6', 'Avg_Pmt_Ratio', 'Util_1', 'Util_2', 'Util_3', 'Util_4', 'Util_5',
                                         'Util_6', 'Avg_Util', 'Bal_Growth_6mo', 'Util_Growth_6mo', 'Max_Bill_Amt', 'Max_Pmt_Amt', 
                                         'Max_DLQ', 'Pmt_Ratio_Growth_6mo', 'Pmt_Status_Change_6mo','DEFAULT'))

#Change DEFAULT to factor
#Used to be INT
train_eda$DEFAULT <- as.factor(train_eda$DEFAULT)

#Convert SEX, EDUCATION, Bal_Growth_6mo and MARRIAGE to categorical 
train_eda$SEX <- as.factor(train_eda$SEX)
train_eda$EDUCATION <- as.factor(train_eda$EDUCATION)
train_eda$MARRIAGE <- as.factor(train_eda$MARRIAGE)
train_eda$Bal_Growth_6mo <- as.factor(train_eda$Bal_Growth_6mo)

#Output statistical summary 
file.name <- 'Table4.html';
stargazer(train_eda, type=c('html'),out=paste(out.path,file.name,sep=''),
          title=c('Table XYZ: Summary Statistics for EDA Variables'),
          align=TRUE, digits=2, digits.extra=2, initial.zero=TRUE, median=TRUE)

str(train_eda)
summary(train_eda)

#Discretize LIMIT_BAL using WOE
limitbal.tree <- woe.tree.binning(df=train_eda,target.var=c('DEFAULT'),pred.var=c('LIMIT_BAL'))

# WOE plot for LIMIT_BAL bins;
woe.binning.plot(limitbal.tree)
# Note that we got different bins;

# Score bins on data frame;
limitbal.df <- woe.binning.deploy(df=train_eda,binning=limitbal.tree)
head(limitbal.df)
table(limitbal.df$LIMIT_BAL.binned)

# See the WOE Binning Table
woe.binning.table(limitbal.tree)

#Add the discretized version of LIMIT_BAL
train_eda <- limitbal.df

#Change the names of the factors
train_eda$LIMIT_BAL_Cat <- revalue(train_eda$LIMIT_BAL.binned, c("(-Inf,30000]"="Limit_Bal_10000_30000", 
                                                                 "(30000,140000]"="Limit_Bal_40000_140000", 
                                                                 "(140000, Inf]"="Limit_Bal_150000_1000000"))
                                                  
train_eda$LIMIT_BAL_Cat <- droplevels(train_eda$LIMIT_BAL_Cat)

#Drop LIMIT_BAL.binned
train_eda$LIMIT_BAL.binned <- NULL

summary(train_eda)
str(train_eda)

#Box plots to with DEFAULT and other predictor variables
ggplot(train_eda, aes(x=DEFAULT, y=Avg_Bill_Amt, fill=DEFAULT))+geom_boxplot(outlier.colour=NA) +
  scale_fill_brewer(palette = 'Blues', labels=c("No", "Yes")) + theme_classic()+coord_cartesian(ylim = c(0, 200000)) +
  labs(title="Figure 1", subtitle='Avg_Bill_Amt vs. DEFAULT (exc. outliers)') +  
  theme(plot.title = element_text(hjust = 0.5))+theme(plot.subtitle = element_text(hjust = 0.5))

ggplot(train_eda, aes(x=DEFAULT, y=Avg_Util, fill=DEFAULT))+geom_boxplot() +
  scale_fill_brewer(palette = 'Blues', labels=c("No", "Yes")) + theme_classic() +
  labs(title="Figure 2", subtitle='Avg_Util vs. DEFAULT') +  theme(plot.title = element_text(hjust = 0.5))+
  theme(plot.subtitle = element_text(hjust = 0.5))

ggplot(train_eda, aes(x=DEFAULT, y=Avg_Pmt_Ratio, fill=DEFAULT))+geom_boxplot() +
  scale_fill_brewer(palette = 'Blues', labels=c("No", "Yes")) + theme_classic() +
  labs(title="Figure 3", subtitle='Avg_Pmt_Ratio vs. DEFAULT') +  theme(plot.title = element_text(hjust = 0.5))+
  theme(plot.subtitle = element_text(hjust = 0.5))

ggplot(train_eda, aes(x=Age, y=Max_Bill_Amt, fill=DEFAULT))+geom_boxplot(outlier.colour=NA) + coord_cartesian(ylim = c(-3000, 220000))+
  scale_fill_brewer(palette = 'Blues', labels=c("No", "Yes")) + theme_classic()+
  labs(title="Figure 4", subtitle='Max_Bill_Amt vs. Age') +  theme(plot.title = element_text(hjust = 0.5))+
  theme(plot.subtitle = element_text(hjust = 0.5))

ggplot(train_eda, aes(as.factor(Max_DLQ))) + geom_bar(aes(fill=DEFAULT), width = 0.5) + 
  scale_fill_brewer(palette = 'Blues', labels=c("No", "Yes")) + theme_classic()+
  labs(title="Figure 6", subtitle='Max_DLQ') +  theme(plot.title = element_text(hjust = 0.5))+
  theme(plot.subtitle = element_text(hjust = 0.5)) + xlab("Max_DLQ") + ylab("Count") 

ggplot(train_eda, aes(Util_Growth_6mo)) + geom_histogram(aes(fill=DEFAULT), breaks=seq(-30, 30, by=5))+
  scale_fill_brewer(palette = 'Blues', labels=c("No", "Yes")) + theme_classic()+
  labs(title="Figure 5", subtitle='Util_Growth_6mo (Truncated)') +  theme(plot.title = element_text(hjust = 0.5))+
  theme(plot.subtitle = element_text(hjust = 0.5)) + ylab("Count") + scale_x_continuous(breaks=seq(-30, 30, by=5))

ggplot(train_eda, aes(as.factor(Pmt_Status_Change_6mo))) + geom_bar(aes(fill=DEFAULT), width = 0.5) + 
  scale_fill_brewer(palette = 'Blues', labels=c("No", "Yes")) + theme_classic()+
  labs(title="Figure 7", subtitle='Pmt_Status_Change_6mo') +  theme(plot.title = element_text(hjust = 0.5))+
  theme(plot.subtitle = element_text(hjust = 0.5)) + xlab("Pmt_Status_Change_6mo") + ylab("Count")


#######################Model Based EDA

summary(train_eda)
str(train_eda)


#Make big tree
form <- as.formula(DEFAULT~.)
tree.1 <- rpart(form, data=train_eda, method = "class")
prp(tree.1)
fancyRpartPlot(tree.1)


#Discretize variables using WOE
avg_bill_amt.tree <- woe.tree.binning(df=train_eda,target.var=c('DEFAULT'),pred.var=c('Avg_Bill_Amt'))
avg_pmt_amt.tree <- woe.tree.binning(df=train_eda,target.var=c('DEFAULT'),pred.var=c('Avg_Pmt_Amt'))
avg_pmt_ratio.tree <- woe.tree.binning(df=train_eda,target.var=c('DEFAULT'),pred.var=c('Avg_Pmt_Ratio'))
avg_util.tree <- woe.tree.binning(df=train_eda,target.var=c('DEFAULT'),pred.var=c('Avg_Util'))
util_growth_6mo.tree <- woe.tree.binning(df=train_eda,target.var=c('DEFAULT'),pred.var=c('Util_Growth_6mo'))
max_bill_amt.tree <- woe.tree.binning(df=train_eda,target.var=c('DEFAULT'),pred.var=c('Max_Bill_Amt'))
max_pmt_amt.tree <- woe.tree.binning(df=train_eda,target.var=c('DEFAULT'),pred.var=c('Max_Pmt_Amt'))
max_dlq.tree <- woe.tree.binning(df=train_eda,target.var=c('DEFAULT'),pred.var=c('Max_DLQ'))
pmt_ratio_growth_6mo.tree <- woe.tree.binning(df=train_eda,target.var=c('DEFAULT'),pred.var=c('Pmt_Ratio_Growth_6mo'))
pmt_status_change_6mo.tree <- woe.tree.binning(df=train_eda,target.var=c('DEFAULT'),pred.var=c('Pmt_Status_Change_6mo'))


# Score bins on data frame
avg_bill_amt.df <- woe.binning.deploy(df=train_eda,binning=avg_bill_amt.tree)
avg_pmt_amt.df <- woe.binning.deploy(df=train_eda,binning=avg_pmt_amt.tree)
avg_pmt_ratio.df <- woe.binning.deploy(df=train_eda,binning=avg_pmt_ratio.tree)
avg_util.df <- woe.binning.deploy(df=train_eda,binning=avg_util.tree)
util_growth_6mo.df <- woe.binning.deploy(df=train_eda,binning=util_growth_6mo.tree)
max_bill_amt.df <- woe.binning.deploy(df=train_eda,binning=max_bill_amt.tree)
max_pmt_amt.df <- woe.binning.deploy(df=train_eda,binning=max_pmt_amt.tree)
max_dlq.df <- woe.binning.deploy(df=train_eda,binning=max_dlq.tree)
pmt_ratio_growth_6mo.df <- woe.binning.deploy(df=train_eda,binning=pmt_ratio_growth_6mo.tree)
pmt_status_change_6mo.df <- woe.binning.deploy(df=train_eda,binning=pmt_status_change_6mo.tree)


#Create new training data frames
temp1 <- subset(avg_bill_amt.df, select = c(Avg_Bill_Amt.binned))
temp2 <- subset(avg_pmt_amt.df, select = c(Avg_Pmt_Amt.binned))
temp3 <- subset(avg_pmt_ratio.df, select = c(Avg_Pmt_Ratio.binned))
temp4 <- subset(avg_util.df, select = c(Avg_Util.binned))
temp5 <- subset(util_growth_6mo.df , select = c(Util_Growth_6mo.binned))
temp6 <- subset(max_bill_amt.df, select = c(Max_Bill_Amt.binned))
temp7 <- subset(max_pmt_amt.df, select = c(Max_Pmt_Amt.binned))
temp8 <- subset(max_dlq.df, select = c(Max_DLQ.binned))
temp9 <- subset(pmt_ratio_growth_6mo.df, select = c(Pmt_Ratio_Growth_6mo.binned))
temp10 <- subset(pmt_status_change_6mo.df, select = c(Pmt_Status_Change_6mo.binned))

train_final <- cbind(train_eda, temp1, temp2, temp3, temp4, temp5, temp6, temp7, temp8, temp9, temp10)

#Change column name 
str(train_final)
colnames(train_final)[colnames(train_final)=="Avg_Bill_Amt.binned"] <- "Avg_Bill_Amt_Cat"
colnames(train_final)[colnames(train_final)=="Avg_Pmt_Amt.binned"] <- "Avg_Pmt_Amt_Cat"
colnames(train_final)[colnames(train_final)=="Avg_Pmt_Ratio.binned"] <- "Avg_Pmt_Ratio_Cat"
colnames(train_final)[colnames(train_final)=="Avg_Util.binned"] <- "Avg_Util_Cat"
colnames(train_final)[colnames(train_final)=="Util_Growth_6mo.binned"] <- "Util_Growth_6mo_Cat"
colnames(train_final)[colnames(train_final)=="Max_Bill_Amt.binned"] <- "Max_Bill_Amt_Cat"
colnames(train_final)[colnames(train_final)=="Max_Pmt_Amt.binned"] <- "Max_Pmt_Amt_Cat"
colnames(train_final)[colnames(train_final)=="Max_DLQ.binned"] <- "Max_DLQ_Cat"
colnames(train_final)[colnames(train_final)=="Pmt_Ratio_Growth_6mo.binned"] <- "Pmt_Ratio_Growth_6mo_Cat"
colnames(train_final)[colnames(train_final)=="Pmt_Status_Change_6mo.binned"] <- "Pmt_Status_Change_6mo_Cat"


train_final$Avg_Bill_Amt_Cat <- droplevels(train_final$Avg_Bill_Amt_Cat)
train_final$Avg_Pmt_Amt_Cat <- droplevels(train_final$Avg_Pmt_Amt_Cat)
train_final$Avg_Pmt_Ratio_Cat <- droplevels(train_final$Avg_Pmt_Ratio_Cat)
train_final$Avg_Util_Cat <- droplevels(train_final$Avg_Util_Cat)
train_final$Util_Growth_6mo_Cat <- droplevels(train_final$Util_Growth_6mo_Cat)
train_final$Max_Bill_Amt_Cat <- droplevels(train_final$Max_Bill_Amt_Cat)
train_final$Max_Pmt_Amt_Cat <- droplevels(train_final$Max_Pmt_Amt_Cat)
train_final$Max_DLQ_Cat <- droplevels(train_final$Max_DLQ_Cat)
train_final$Pmt_Ratio_Growth_6mo_Cat <- droplevels(train_final$Pmt_Ratio_Growth_6mo_Cat)
train_final$Pmt_Status_Change_6mo_Cat <- droplevels(train_final$Pmt_Status_Change_6mo_Cat)

#Make second decision tree 
tree.2 <- rpart(form, data=train_final, method = "class")
prp(tree.2)
fancyRpartPlot(tree.2)

#Plot only the categorical variables 
str(train_final)
tree.3 <- rpart(DEFAULT ~ SEX + EDUCATION + MARRIAGE + Age + Bal_Growth_6mo + LIMIT_BAL_Cat + Avg_Bill_Amt_Cat + Avg_Pmt_Amt_Cat +
                  Avg_Pmt_Ratio_Cat + Avg_Util_Cat + Util_Growth_6mo_Cat + Max_Bill_Amt_Cat + Max_Pmt_Amt_Cat + Max_DLQ_Cat + 
                  Pmt_Ratio_Growth_6mo_Cat + Pmt_Status_Change_6mo_Cat, data=train_final, method = 'class')
fancyRpartPlot(tree.3)

#Plot categorical without MAX_DLQ
#Did not include in paper
tree.4 <- rpart(DEFAULT ~ SEX + EDUCATION + MARRIAGE + Age + Bal_Growth_6mo + LIMIT_BAL_Cat + Avg_Bill_Amt_Cat + Avg_Pmt_Amt_Cat +
                  Avg_Pmt_Ratio_Cat + Avg_Util_Cat + Util_Growth_6mo_Cat + Max_Bill_Amt_Cat + Max_Pmt_Amt_Cat + 
                  Pmt_Ratio_Growth_6mo_Cat + Pmt_Status_Change_6mo_Cat, data=train_final, method = 'class')
fancyRpartPlot(tree.4)

#OneR 
oner.model <- OneR(train_final, verbose = TRUE)
summary(oner.model)

##########################################Transform test data set 

#Create Age
# Score bins on data frame;
tree2.df <- woe.binning.deploy(df=test.df,binning=age.tree)
head(tree2.df)
table(tree2.df$AGE.binned)

#Add the discretized version of AGE
test.df <- tree2.df

table(test.df$AGE.binned)

#Change the names of the factors
test.df$Age_Bin <- revalue(test.df$AGE.binned, c("(-Inf,24]"="Age_21_24", "(24,25]"="Age_25", "(25,34]"="Age_26_34",
                                                   "(34,48]"="Age_35_48","(48, Inf]"="Age_49_79"))
test.df$Age_Bin <- droplevels(test.df$Age_Bin)

#Drop certain variables to create new dataframe
test.df <- subset(test.df, select = -c(u, data.group, AGE.binned, train, test, validate, ID))

#Change column name 
colnames(test.df)[colnames(test.df)=="Age_Bin"] <- "Age"
table(test.df$Age)


########################Create Average Bill Amount (Avg_Bill_Amt)
test.df <- within(test.df, Avg_Bill_Amt <- (BILL_AMT1 + BILL_AMT2 + BILL_AMT3 +BILL_AMT4 +BILL_AMT5 +BILL_AMT6)/6)


########################Create Average Payment Amount (Avg_Pmt_Amt)
test.df <- within(test.df, Avg_Pmt_Amt <- (PAY_AMT1 + PAY_AMT2 + PAY_AMT3 +PAY_AMT4 +PAY_AMT5 +PAY_AMT6)/6)
########################Create Payment Ratio (Pmt_Ratio_X)

test.df <- within(test.df, Pmt_Ratio_2 <- (PAY_AMT1/BILL_AMT2)*100)
test.df <- within(test.df, Pmt_Ratio_3 <- (PAY_AMT2/BILL_AMT3)*100)
test.df <- within(test.df, Pmt_Ratio_4 <- (PAY_AMT3/BILL_AMT4)*100)
test.df <- within(test.df, Pmt_Ratio_5 <- (PAY_AMT4/BILL_AMT5)*100)
test.df <- within(test.df, Pmt_Ratio_6 <- (PAY_AMT5/BILL_AMT6)*100)

#Turn all 0 PMT and negative BILL to 101 for all Pmt_Ratio_X columns
test.df <- within(test.df, Pmt_Ratio_2[Pmt_Ratio_2 == 0 & PAY_AMT1 == 0 & BILL_AMT2 < 0] <- 101)
test.df <- within(test.df, Pmt_Ratio_3[Pmt_Ratio_3 == 0 & PAY_AMT2 == 0 & BILL_AMT3 < 0] <- 101)
test.df <- within(test.df, Pmt_Ratio_4[Pmt_Ratio_4 == 0 & PAY_AMT3 == 0 & BILL_AMT4 < 0] <- 101)
test.df <- within(test.df, Pmt_Ratio_5[Pmt_Ratio_5 == 0 & PAY_AMT4 == 0 & BILL_AMT5 < 0] <- 101)
test.df <- within(test.df, Pmt_Ratio_6[Pmt_Ratio_6 == 0 & PAY_AMT5 == 0 & BILL_AMT6 < 0] <- 101)

#Turn all NaN's into 100 
test.df[is.na(test.df)] <- 100

#Turn all negative numbers to 101 
test.df <- within(test.df, Pmt_Ratio_2[Pmt_Ratio_2 < 0] <- 101)
test.df <- within(test.df, Pmt_Ratio_3[Pmt_Ratio_3 < 0] <- 101)
test.df <- within(test.df, Pmt_Ratio_4[Pmt_Ratio_4 < 0] <- 101)
test.df <- within(test.df, Pmt_Ratio_5[Pmt_Ratio_5 < 0] <- 101)
test.df <- within(test.df, Pmt_Ratio_6[Pmt_Ratio_6 < 0] <- 101)

#Turn all Inf values to 101 
is.na(test.df)<-sapply(test.df, is.infinite)
test.df[is.na(test.df)]<-101

#Winsorize all values greater than 101 into 101
test.df <- within(test.df, Pmt_Ratio_2[Pmt_Ratio_2 > 101] <- 101)
test.df <- within(test.df, Pmt_Ratio_3[Pmt_Ratio_3 > 101] <- 101)
test.df <- within(test.df, Pmt_Ratio_4[Pmt_Ratio_4 > 101] <- 101)
test.df <- within(test.df, Pmt_Ratio_5[Pmt_Ratio_5 > 101] <- 101)
test.df <- within(test.df, Pmt_Ratio_6[Pmt_Ratio_6 > 101] <- 101)

################################### Average Payment Ratio (Avg_Pmt_Ratio)
test.df <- within(test.df, Avg_Pmt_Ratio <- (Pmt_Ratio_2 + Pmt_Ratio_3 +Pmt_Ratio_4 +Pmt_Ratio_5 +Pmt_Ratio_6 )/5)

################################### Utilization (Util)
test.df <- within(test.df, Util_1 <-  (BILL_AMT1/LIMIT_BAL)*100)
test.df <- within(test.df, Util_2 <-  (BILL_AMT2/LIMIT_BAL)*100)
test.df <- within(test.df, Util_3 <-  (BILL_AMT3/LIMIT_BAL)*100)
test.df <- within(test.df, Util_4 <-  (BILL_AMT4/LIMIT_BAL)*100)
test.df <- within(test.df, Util_5 <-  (BILL_AMT5/LIMIT_BAL)*100)
test.df <- within(test.df, Util_6 <-  (BILL_AMT6/LIMIT_BAL)*100)

#Convert negative values into 0
#Means that the customer was rewarded for not utilizing their balance
test.df <- within(test.df, Util_1[Util_1 < 0] <- 0)
test.df <- within(test.df, Util_2[Util_2 < 0] <- 0)
test.df <- within(test.df, Util_3[Util_3 < 0] <- 0)
test.df <- within(test.df, Util_4[Util_4 < 0] <- 0)
test.df <- within(test.df, Util_5[Util_5 < 0] <- 0)
test.df <- within(test.df, Util_6[Util_6 < 0] <- 0)

#Convert anything greater than 100 into 100
test.df <- within(test.df, Util_1[Util_1 > 100] <- 100)
test.df <- within(test.df, Util_2[Util_2 > 100] <- 100)
test.df <- within(test.df, Util_3[Util_3 > 100] <- 100)
test.df <- within(test.df, Util_4[Util_4 > 100] <- 100)
test.df <- within(test.df, Util_5[Util_5 > 100] <- 100)
test.df <- within(test.df, Util_6[Util_6 > 100] <- 100)

################################### Average Utilization (Avg_Util) 
test.df <- within(test.df, Avg_Util <- (Util_1 + Util_2 + Util_3 + Util_4 + Util_5 + Util_6)/6)

################################### Balance Growth Over 6 Months (Bal_Growth_6mo)
#(1=Yes, 0 = No)
test.df <- within(test.df, Bal_Growth_6mo <- BILL_AMT1 - BILL_AMT6)

test.df <- within(test.df, Bal_Growth_6mo[Bal_Growth_6mo > 0 & BILL_AMT1 > BILL_AMT6] <- 1)
test.df <- within(test.df, Bal_Growth_6mo[Bal_Growth_6mo == 0] <- 0)
test.df <- within(test.df, Bal_Growth_6mo[Bal_Growth_6mo < 0 & BILL_AMT1 == 0 & BILL_AMT6>0] <- 0)
test.df <- within(test.df, Bal_Growth_6mo[Bal_Growth_6mo < 0 & BILL_AMT1 <  0 & BILL_AMT6==0] <- 0)
test.df <- within(test.df, Bal_Growth_6mo[Bal_Growth_6mo < 0 & BILL_AMT1 < 0 & BILL_AMT6>0] <- 0)
test.df <- within(test.df, Bal_Growth_6mo[Bal_Growth_6mo < 0 & BILL_AMT1 < 0 & BILL_AMT6<0] <- 0)
test.df <- within(test.df, Bal_Growth_6mo[Bal_Growth_6mo < 0 & BILL_AMT1 < BILL_AMT6] <- 0)

################################### Utilization Growth Over 6 Months (Util_Growth_6mo)
test.df <- within(test.df, Util_Growth_6mo <- Util_1 - Util_6)


################################## Max Bill Amount (Max_Bill_Amt)
test.df[, "Max_Bill_Amt"] <- apply(test.df[, c('BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6' )], 1, max)


################################## Max Payment Amount (Max_Pmt_Amt)
test.df[, "Max_Pmt_Amt"] <- apply(test.df[, c('PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6' )], 1, max)

################################## Max Delinquency (Max_DLQ)
test.df[, "Max_DLQ"] <- apply(test.df[, c('PAY_1', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6' )], 1, max)


################################### Payment Ratio Growth Over 6 Months (Pmt_Ratio_Growth_6mo)
test.df <- within(test.df, Pmt_Ratio_Growth_6mo <- Pmt_Ratio_2 - Pmt_Ratio_6)

summary(test.df$Util_Growth_6mo)

################################### Payment Status Change Over 6 Months (Pmt_Status_Change_6mo)
test.df <- within(test.df, Pmt_Status_Change_6mo <- PAY_1 - PAY_6)



#Create EDA dataframe of test set         
test_eda <- subset(test.df, select = c('LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 'Age', 'Avg_Bill_Amt',
                                       'Avg_Pmt_Amt', 'Pmt_Ratio_2', 'Pmt_Ratio_3', 'Pmt_Ratio_4', 'Pmt_Ratio_5', 
                                       'Pmt_Ratio_6', 'Avg_Pmt_Ratio', 'Util_1', 'Util_2', 'Util_3', 'Util_4', 'Util_5',
                                       'Util_6', 'Avg_Util', 'Bal_Growth_6mo', 'Util_Growth_6mo', 'Max_Bill_Amt', 'Max_Pmt_Amt', 
                                       'Max_DLQ', 'Pmt_Ratio_Growth_6mo', 'Pmt_Status_Change_6mo','DEFAULT'))

#Change DEFAULT to factor
#Used to be INT
test_eda$DEFAULT <- as.factor(test_eda$DEFAULT)

#Convert SEX, EDUCATION, Bal_Growth_6mo and MARRIAGE to categorical 
test_eda$SEX <- as.factor(test_eda$SEX)
test_eda$EDUCATION <- as.factor(test_eda$EDUCATION)
test_eda$MARRIAGE <- as.factor(test_eda$MARRIAGE)
test_eda$Bal_Growth_6mo <- as.factor(test_eda$Bal_Growth_6mo)



#Discretize LIMIT_BAL using WOE

# Score bins on data frame;
limitbal2.df <- woe.binning.deploy(df=test_eda,binning=limitbal.tree)
head(limitbal2.df)
table(limitbal2.df$LIMIT_BAL.binned)
#Add the discretized version of LIMIT_BAL
test_eda <- limitbal2.df
table(test_eda$LIMIT_BAL.binned)
#Change the names of the factors
test_eda$LIMIT_BAL_Cat <- revalue(test_eda$LIMIT_BAL.binned, c("(-Inf,30000]"="Limit_Bal_10000_30000", 
                                                               "(30000,140000]"="Limit_Bal_40000_140000", 
                                                               "(140000, Inf]"="Limit_Bal_150000_1000000"))

test_eda$LIMIT_BAL_Cat <- droplevels(test_eda$LIMIT_BAL_Cat)

#Drop LIMIT_BAL.binned
test_eda$LIMIT_BAL.binned <- NULL

#Discretize a lot of other features
# Score bins on data frame
avg_bill_amt2.df <- woe.binning.deploy(df=test_eda,binning=avg_bill_amt.tree)
avg_pmt_amt2.df <- woe.binning.deploy(df=test_eda,binning=avg_pmt_amt.tree)
avg_pmt_ratio2.df <- woe.binning.deploy(df=test_eda,binning=avg_pmt_ratio.tree)
avg_util2.df <- woe.binning.deploy(df=test_eda,binning=avg_util.tree)
util_growth_6mo2.df <- woe.binning.deploy(df=test_eda,binning=util_growth_6mo.tree)
max_bill_amt2.df <- woe.binning.deploy(df=test_eda,binning=max_bill_amt.tree)
max_pmt_amt2.df <- woe.binning.deploy(df=test_eda,binning=max_pmt_amt.tree)
max_dlq2.df <- woe.binning.deploy(df=test_eda,binning=max_dlq.tree)
pmt_ratio_growth_6mo2.df <- woe.binning.deploy(df=test_eda,binning=pmt_ratio_growth_6mo.tree)
pmt_status_change_6mo2.df <- woe.binning.deploy(df=test_eda,binning=pmt_status_change_6mo.tree)


#Create new test data frames
temp12 <- subset(avg_bill_amt2.df, select = c(Avg_Bill_Amt.binned))
temp22 <- subset(avg_pmt_amt2.df, select = c(Avg_Pmt_Amt.binned))
temp32 <- subset(avg_pmt_ratio2.df, select = c(Avg_Pmt_Ratio.binned))
temp42 <- subset(avg_util2.df, select = c(Avg_Util.binned))
temp52 <- subset(util_growth_6mo2.df , select = c(Util_Growth_6mo.binned))
temp62 <- subset(max_bill_amt2.df, select = c(Max_Bill_Amt.binned))
temp72 <- subset(max_pmt_amt2.df, select = c(Max_Pmt_Amt.binned))
temp82 <- subset(max_dlq2.df, select = c(Max_DLQ.binned))
temp92 <- subset(pmt_ratio_growth_6mo2.df, select = c(Pmt_Ratio_Growth_6mo.binned))
temp102 <- subset(pmt_status_change_6mo2.df, select = c(Pmt_Status_Change_6mo.binned))

test_final <- cbind(test_eda, temp12, temp22, temp32, temp42, temp52, temp62, temp72, temp82, temp92, temp102)

#Change column name 
colnames(test_final)[colnames(test_final)=="Avg_Bill_Amt.binned"] <- "Avg_Bill_Amt_Cat"
colnames(test_final)[colnames(test_final)=="Avg_Pmt_Amt.binned"] <- "Avg_Pmt_Amt_Cat"
colnames(test_final)[colnames(test_final)=="Avg_Pmt_Ratio.binned"] <- "Avg_Pmt_Ratio_Cat"
colnames(test_final)[colnames(test_final)=="Avg_Util.binned"] <- "Avg_Util_Cat"
colnames(test_final)[colnames(test_final)=="Util_Growth_6mo.binned"] <- "Util_Growth_6mo_Cat"
colnames(test_final)[colnames(test_final)=="Max_Bill_Amt.binned"] <- "Max_Bill_Amt_Cat"
colnames(test_final)[colnames(test_final)=="Max_Pmt_Amt.binned"] <- "Max_Pmt_Amt_Cat"
colnames(test_final)[colnames(test_final)=="Max_DLQ.binned"] <- "Max_DLQ_Cat"
colnames(test_final)[colnames(test_final)=="Pmt_Ratio_Growth_6mo.binned"] <- "Pmt_Ratio_Growth_6mo_Cat"
colnames(test_final)[colnames(test_final)=="Pmt_Status_Change_6mo.binned"] <- "Pmt_Status_Change_6mo_Cat"


test_final$Avg_Bill_Amt_Cat <- droplevels(test_final$Avg_Bill_Amt_Cat)
test_final$Avg_Pmt_Amt_Cat <- droplevels(test_final$Avg_Pmt_Amt_Cat)
test_final$Avg_Pmt_Ratio_Cat <- droplevels(test_final$Avg_Pmt_Ratio_Cat)
test_final$Avg_Util_Cat <- droplevels(test_final$Avg_Util_Cat)
test_final$Util_Growth_6mo_Cat <- droplevels(test_final$Util_Growth_6mo_Cat)
test_final$Max_Bill_Amt_Cat <- droplevels(test_final$Max_Bill_Amt_Cat)
test_final$Max_Pmt_Amt_Cat <- droplevels(test_final$Max_Pmt_Amt_Cat)
test_final$Max_DLQ_Cat <- droplevels(test_final$Max_DLQ_Cat)
test_final$Pmt_Ratio_Growth_6mo_Cat <- droplevels(test_final$Pmt_Ratio_Growth_6mo_Cat)
test_final$Pmt_Status_Change_6mo_Cat <- droplevels(test_final$Pmt_Status_Change_6mo_Cat)


##################################Build Models

#Random Forest
#Set cross-validation settings
trControl <- trainControl(method = "cv",
    number = 5,
    search = "random")

#Find best mtry
set.seed(11)
tuneGrid <- expand.grid(.mtry =c(2:38)) # total of 39 values
rf.mtry <- train(DEFAULT~., data=train_final, method='rf', metric='Accuracy', tuneGrid = tuneGrid, trControl=trControl, importance=TRUE)
print(rf.mtry)
best_mtry <- rf.mtry$bestTune$mtry  

#Find best maxnodes
store_maxnode <- list()
tuneGrid <- expand.grid(.mtry = best_mtry)
#Only test maxnodes between 
for (maxnodes in c(5: 15)) {
  set.seed(12)
  rf.maxnode <- train(DEFAULT~.,
                      data = train_final,
                      method = "rf",
                      metric = "Accuracy",
                      tuneGrid = tuneGrid,
                      trControl = trControl,
                      importance = TRUE,
                      maxnodes = maxnodes)
  current_iteration <- toString(maxnodes)
  store_maxnode[[current_iteration]] <- rf.maxnode
}
results_mtry <- resamples(store_maxnode)
summary(results_mtry)

#Find the best ntrees
store_maxtrees <- list()
for (ntree in c(400, 500, 600, 700, 1000, 2000)) {
  set.seed(13)
  rf.maxtrees <- train(DEFAULT~.,
                       data = train_final,
                       method = "rf",
                       metric = "Accuracy",
                       tuneGrid = tuneGrid,
                       trControl = trControl,
                       importance = TRUE,
                       maxnodes = 12,
                       ntree = ntree)
  key <- toString(ntree)
  store_maxtrees[[key]] <- rf.maxtrees
}
results_tree <- resamples(store_maxtrees)
summary(results_tree)



#Fit final RF MOdel 
#include savePredictions to create matrix table 
set.seed(14)
rf.final <- randomForest(DEFAULT~., data=train_final, mtry=12, maxnodes=12, importance=TRUE, ntree=400)
rf.final
  
#Create variable importance plot
#Training Set
feat_imp_df <- importance(rf.final) %>% data.frame() %>% mutate(feature = row.names(.))
topfeat.rf <- subset(feat_imp_df, select = c('MeanDecreaseGini', 'feature'))
top_n(topfeat.rf, n=15, MeanDecreaseGini) %>% ggplot(., aes(x = reorder(feature, MeanDecreaseGini), y=MeanDecreaseGini)) +
  geom_bar(stat='identity', fill='steelblue') +
  coord_flip() +
  theme_classic() +
  labs(
    x     = "Feature",
    y     = "Importance",
    title = "Figure 11: Random Forest Model Top 15 Features "
  )
  
#Create variable importance table
#Training Set
importance(rf.final)

#Make predictions using Test Set
rf.predictions <- predict(rf.final, test_final, type='response') 
#might change to probability and then use optimal cut off value 
#the second parameter will need to be the test_final set to generate predicted probablities for class; type would be 'prob' 

#Create confusion Matrix
#The rows are actuals and columns are predicted   
#Training Set
rf.train.table <- table(observed=rf.final$y, predicted=rf.final$predicted)  #confusion matrix
rf.train.normalize <- apply(rf.train.table, MARGIN=1, FUN=sum) 
rf.train.table/rf.train.normalize
rf.final$confusion   #confusion matrix from model 

#Test Set
rf.test.table <- table(observed=test_final$DEFAULT, predicted=rf.predictions)
rf.test.normalize <- apply(rf.test.table, MARGIN=1, FUN=sum)
rf.test.table/rf.test.normalize
confusionMatrix(rf.predictions, test_final$DEFAULT) #accuracy for test set

#########################Build Gradient Boosting Model 


gbm.grid <- expand.grid(n.trees=c(2000), interaction.depth=c(2), shrinkage=c(0.01), n.minobsinnode=c(1))
gbm.ctrl <- trainControl(method = 'cv', number=3)

set.seed(15)
gbm.final <- train(DEFAULT~., data=train_final, method="gbm", distribution = 'bernoulli', trControl=gbm.ctrl,
                   tuneGrid=gbm.grid, verbose=FALSE)

gbm.final

#Show table of variable influence 
summary(gbm.final)

#Make predictions using Test Set
gbm.predictions <- predict(gbm.final, test_final) 
#might change to probability and then use optimal cut off value 
#the second parameter will need to be the test_final set to generate predicted probablities for class; type would be 'prob' 

#Confusion Matrix
#Training Set
gbm.fittedvalues <-predict(gbm.final, train_final)
confusionMatrix(gbm.fittedvalues, train_final$DEFAULT)

gbm.train.table <- table(observed=train_final$DEFAULT, predicted=gbm.fittedvalues)  #confusion matrix
gbm.train.normalize <- apply(gbm.train.table, MARGIN=1, FUN=sum) 
gbm.train.table/gbm.train.normalize


#Test Set
confusionMatrix(gbm.predictions, test_final$DEFAULT)

gbm.test.table <- table(observed=test_final$DEFAULT, predicted=gbm.predictions)  #confusion matrix
gbm.test.normalize <- apply(gbm.test.table, MARGIN=1, FUN=sum) 
gbm.test.table/gbm.test.normalize
#############################Build Logistic Regression Model 

#Define stepwise algorithm
#Need to specifiy the upper model and lower models
# Define the upper model as the FULL model 
upper.glm <- glm(DEFAULT~Max_DLQ_Cat + Avg_Bill_Amt + Avg_Pmt_Amt + Avg_Pmt_Ratio + Pmt_Ratio_4 + Pmt_Status_Change_6mo_Cat +
                   Pmt_Ratio_3 + Util_Growth_6mo + Max_Bill_Amt + Max_Pmt_Amt + Max_DLQ + Pmt_Ratio_2 + Pmt_Status_Change_6mo,
                 data=train_final, family = binomial)

# Need a simple logistic regression model  to initialize stepwise selection 
simple.glm <- glm(DEFAULT ~ Max_DLQ_Cat,data=train_final, family = binomial) 

#Create glm.stepwise
glm.stepwise <- stepAIC(object=simple.glm,scope=list(upper=formula(upper.glm),lower=~1), direction=c('both'))
summary(glm.stepwise)

file.name <- 'Table6.html';
stargazer(glm.stepwise, type=c('html'),out=paste(out.path,file.name,sep=''),
          title=c('Table 18: Logistic Regression Model Summary'),
          align=TRUE, digits=2, digits.extra=2, initial.zero=TRUE) 

#Create optimal cutoff
glm.score <- glm.stepwise$fitted.values
  
roc.glm <- roc(response=train_final$DEFAULT, predictor=glm.score)
print(roc.glm)
plot(roc.glm)

# Compute AUC
auc.glm <- auc(roc.glm)
auc.glm

roc.glm.specs <- coords(roc=roc.glm,x=c('best'),
                    input=c('threshold','specificity','sensitivity'),
                    ret=c('threshold','specificity','sensitivity'),
                    as.list=TRUE
)

roc.glm.specs

#Output confusion matrix for logistic regrression model 
glm.ModelScores <- glm.stepwise$fitted.values;
glm.ModelScores <- as.data.frame(glm.ModelScores)
colnames(glm.ModelScores)[colnames(glm.ModelScores)=="glm.ModelScores"] <- "ModelScores"

glm.ModelScores$classes <- ifelse(glm.ModelScores$ModelScores>roc.glm.specs$threshold,1,0);

glm.train.table <- table(observed=train_final$DEFAULT, predicted=glm.ModelScores$classes)  #confusion matrix
glm.train.table
glm.train.normalize <- apply(glm.train.table, MARGIN=1, FUN=sum) 
glm.train.table/glm.train.normalize

#Make predictions from GLM model and confusion matrix 
glm.ModelPredict <- predict(glm.stepwise, test_final, type='response')
glm.ModelPredict <- as.data.frame(glm.ModelPredict)
colnames(glm.ModelPredict)[colnames(glm.ModelPredict)=="glm.ModelPredict"] <- "ModelPredict"

glm.ModelPredict$class <- ifelse(glm.ModelPredict$ModelPredict>roc.glm.specs$threshold,1,0)

glm.test.table <- table(observed=test_final$DEFAULT, predicted=glm.ModelPredict$class)  #confusion matrix
glm.test.table
glm.test.normalize <- apply(glm.test.table, MARGIN=1, FUN=sum) 
glm.test.table/glm.test.normalize


#Output ROC Curve for all models using optimal cutoff value 
#GLM results train
roc.glm <- roc(response=train_final$DEFAULT, predictor=glm.score)
print(roc.glm)
plot(roc.glm, main='ROC Curve: Logistic Regression (Training Set')

# Compute AUC
auc.glm <- auc(roc.glm)
auc.glm

#GLM results test
glm.predict <- predict(glm.stepwise, test_final, type='response')
roc.glm.test <- roc(response=test_final$DEFAULT, predictor=glm.predict) #these are probabilities 
print(roc.glm.test)
plot(roc.glm.test, main='ROC Curve: Logistic Regression (Test Set)')

# Compute AUC
auc.glm.test <- auc(roc.glm.test)
auc.glm.test

##############RF ROC Analysis
set.seed(21)
rf.final2 <- randomForest(DEFAULT~., cutoff=c(0.237, 0.763), data=train_final, mtry=12, maxnodes=12, importance=TRUE, ntree=400)
rf.final2

#RF results train
rf.prob.train <- predict(rf.final2, train_final, type='prob')
rf.prob.train <- rf.prob.train[,-1]
roc.rf.train <- roc(response=train_final$DEFAULT, predictor = rf.prob.train)
plot(roc.rf.train, main='ROC Curve: Random Forest (Training Set)')

auc.rf.train <- auc(roc.rf.train)
auc.rf.train

rf.prob.train <- as.data.frame(rf.prob.train)

rf.prob.train$classes <- ifelse(rf.prob.train$rf.prob.train>roc.glm.specs$threshold,1,0);

rf.train.table2 <- table(observed=train_final$DEFAULT, predicted=rf.prob.train$classes)  #confusion matrix
rf.train.table2
rf.train.normalize2 <- apply(rf.train.table2, MARGIN=1, FUN=sum) 
rf.train.table2/rf.train.normalize2


#RF results test 
rf.prob.test <- predict(rf.final2, test_final, type='prob')
rf.prob.test <- rf.prob.test[,-1]
roc.rf.test <- roc(response=test_final$DEFAULT, predictor = rf.prob.test)
plot(roc.rf.test, main='ROC Curve: Random Forest (Test Set)')

auc.rf.test <- auc(roc.rf.test)
auc.rf.test

rf.prob.test <- as.data.frame(rf.prob.test)

rf.prob.test$classes <- ifelse(rf.prob.test$rf.prob.test>roc.glm.specs$threshold,1,0);

rf.test.table2 <- table(observed=test_final$DEFAULT, predicted=rf.prob.test$classes)  #confusion matrix
rf.test.table2
rf.test.normalize2 <- apply(rf.test.table2, MARGIN=1, FUN=sum) 
rf.test.table2/rf.test.normalize2


##############GBM ROC Analysis
gbm.prob.train <- predict(gbm.final, train_final, type = 'prob')
gbm.prob.train <- gbm.prob.train[,-1]
roc.gbm.train <- roc(response=train_final$DEFAULT, predictor = gbm.prob.train)
plot(roc.gbm.train, main='ROC Curve: Gradient Boosting (Training Set)')

auc.gbm.train <- auc(roc.gbm.train)
auc.gbm.train

gbm.prob.train <- as.data.frame(gbm.prob.train)

gbm.prob.train$classes <- ifelse(gbm.prob.train$gbm.prob.train>roc.glm.specs$threshold,1,0);

gbm.train.table2 <- table(observed=train_final$DEFAULT, predicted=gbm.prob.train$classes)  #confusion matrix
gbm.train.table2
gbm.train.normalize2 <- apply(gbm.train.table2, MARGIN=1, FUN=sum) 
gbm.train.table2/rf.train.normalize2


#GBM results test 
gbm.prob.test <- predict(gbm.final, test_final, type='prob')
gbm.prob.test <- gbm.prob.test[,-1]
roc.gbm.test <- roc(response=test_final$DEFAULT, predictor = gbm.prob.test)
plot(roc.gbm.test, main='ROC Curve: Gradient Boosting (Test Set)')

auc.gbm.test <- auc(roc.gbm.test)
auc.gbm.test

gbm.prob.test <- as.data.frame(gbm.prob.test)

gbm.prob.test$classes <- ifelse(gbm.prob.test$gbm.prob.test>roc.glm.specs$threshold,1,0);
gbm.test.table2 <- table(observed=test_final$DEFAULT, predicted=gbm.prob.test$classes)  #confusion matrix
gbm.test.table2
gbm.test.normalize2 <- apply(gbm.test.table2, MARGIN=1, FUN=sum) 
gbm.test.table2/gbm.test.normalize2


#ROC Curve TRAIN
par(mfrow=c(1,3))
plot(roc.glm, main='ROC Curve: Logistic Regression')
plot(roc.rf.train, main='ROC Curve: Random Forest')
plot(roc.gbm.train, main='ROC Curve: Gradient Boosting')
par(mfrow=c(1,1))


#ROC Curve TEST
par(mfrow=c(1,3))
plot(roc.glm.test, main='ROC Curve: Logistic Regression')
plot(roc.rf.test, main='ROC Curve: Random Forest')
plot(roc.gbm.test, main='ROC Curve: Gradient Boosting')
par(mfrow=c(1,1))


