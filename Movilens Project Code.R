#### Read-Me ####
### This script loads in the 10M movielens database and splits it into an 
### edx and final_holdout_set sets. The edx dataset was used for exploratory
### data analysis and building a recommendation system and 
### final_holdout_set dataset was used to calculate the RMSE value to
### evaluate how close the predictions from the recommendation system 
### are to the data in final_holdout_se dataset. 

#######################################################################
## First step is to load required libraries, the 10M movilens database
## and create the edx and final_holdour_test sets by spliting the 10M
## movielens dataset.This is achieved with the code below.

##########################################################
# Create edx and final_holdout_test sets 
##########################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(knitr)) install.packages("knitr",repos = "http://cran.us.r-project.org")
if(!require(kableExtra)) install.packages("kableExtra", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(knitr)
library(kableExtra)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

options(timeout = 120)

dl <- "ml-10M100K.zip"
if(!file.exists(dl))
  download.file("https://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings_file <- "ml-10M100K/ratings.dat"
if(!file.exists(ratings_file))
  unzip(dl, ratings_file)

movies_file <- "ml-10M100K/movies.dat"
if(!file.exists(movies_file))
  unzip(dl, movies_file)

ratings <- as.data.frame(str_split(read_lines(ratings_file), fixed("::"), simplify = TRUE),
                         stringsAsFactors = FALSE)
colnames(ratings) <- c("userId", "movieId", "rating", "timestamp")
ratings <- ratings %>%
  mutate(userId = as.integer(userId),
         movieId = as.integer(movieId),
         rating = as.numeric(rating),
         timestamp = as.integer(timestamp))

movies <- as.data.frame(str_split(read_lines(movies_file), fixed("::"), simplify = TRUE),
                        stringsAsFactors = FALSE)
colnames(movies) <- c("movieId", "title", "genres")
movies <- movies %>%
  mutate(movieId = as.integer(movieId))

movielens <- left_join(ratings, movies, by = "movieId")

# Final hold-out test set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.6 or later
# set.seed(1) # if using R 3.5 or earlier
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in final hold-out test set are also in edx set
final_holdout_test <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from final hold-out test set back into edx set
removed <- anti_join(temp, final_holdout_test)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

##############################################################
## Exploratory data analysis. All data analysis was performed on the edx
##  dataset
##############################################################

## Set working directory to the Movilens-Project github repository
setwd("C:/Users/tig_m/Documents/Data Science/R Captsone/Movilens-Project")

## Analysis 1: Looking at the average number of ratings
Tab1 <- edx %>% group_by(rating) %>% summarize(avgnumber = mean(n()))

Fig1 <- edx %>% group_by(rating) %>% 
  summarize(avgnumber = mean(n())) %>%
  ggplot(aes(rating,avgnumber)) + geom_bar(stat = "identity") +
  xlab("Ratings") + ylab("Average number of ratings") +
  scale_y_continuous(breaks = seq(4000,4000000,200000))
Fig1

##  Analysis 2: Looking at the frequency of ratings
Tab2 <- edx %>% group_by(rating) %>% count() %>%
  rename(Rating = rating, count = n) %>%
  mutate(Frequency = count/nrow(edx)) %>%
  select(Rating,Frequency)

fig2 <- edx %>% group_by(rating) %>% count() %>%
  rename(Rating = rating, count = n) %>%
  mutate(Frequency = count/nrow(edx)) %>%
  ggplot(aes(Rating,Frequency)) + geom_bar(stat = "identity") +
  xlab("Ratings") + ylab("Frequency of ratings")
fig2

##  Analysis 3: The top 5 movie ratings
Tab3 <- edx %>% group_by(rating) %>%
  rename(Rating = rating) %>%
  summarize(Number = n()) %>%
  top_n(5) %>% arrange(desc(Number))
  
##############################################################
##  Model training. Model training was performed on the edx dataset. The edx dataset
##  was split into a train_set and test_set. Train_set contained 10% edx and the
##  test_set contained the remaining 90%
##############################################################

set.seed(1, sample.kind="Rounding") # if using R 3.6 or later
# set.seed(1) # if using R 3.5 or earlier
# Split edx dataset by creating a train_set which will be used to
# train the model and a test_set to calculate the RMSE value of the
# trained model
test_index <- createDataPartition(y = edx$rating, times = 1, p = 0.1, list = FALSE)
train_set <- edx[-test_index,]
temp <-  edx[test_index,]

# Make sure userId and movieId in test_set are also in train_set
test_set <- temp %>% 
  semi_join(train_set, by = "movieId") %>%
  semi_join(train_set, by = "userId")

# Add rows removed from test_set back into  train_set
removed <- anti_join(temp, test_set)
train_set <- rbind(train_set, removed)

rm(test_index, temp, removed)

### Now train the linear model ###
### Note: the codes used here and below for training linear regression come from the
### edx course on machine learning

# First define the RMSE and MSE
RMSE <- function(true_ratings,predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

# Start with Model 1 which is just the average
AverageRate = mean(train_set$rating, na.rm = TRUE)
naive_rmse <- RMSE(test_set$rating,AverageRate)

## Model 2: add in the movie effect
# Create a dataframe MovieAvg which groups on movieID which the variable we want
# since we are investigating movie effects on the linear model. Then calculate the
# movie effect valuem b_i.
MovieAvg <- train_set %>% group_by(movieId) %>%
  summarize(b_i = mean(rating - AverageRate))

# Create a dataframe MoviePred which will store the movie predictions with movie
# effects included.
MoviePred <- AverageRate + test_set %>% 
  left_join(MovieAvg, by='movieId') %>% .$b_i

# Report the RMSE and MSE values with movie effects included
MovieEffectRMSE <- RMSE(MoviePred,test_set$rating)

## Model 3: Include movie and user effects
# Create dataframe UserAvg by joining with MovieAvg. Then need to group on userID
# as this is the variable needed to determine user effects
UserAvg <- train_set %>% left_join(MovieAvg,by ='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating-AverageRate-b_i))

# Create dataframe UserEffectPred which will store movie predictions with
# user effects included.
UserEffectPred <- test_set %>% left_join(MovieAvg,by='movieId') %>%
  left_join(UserAvg,by='userId') %>% 
  mutate(preds = AverageRate + b_i + b_u)

# Report RMSE value with movie and user effects included
UserEffectRMSE <- RMSE(UserEffectPred$preds,test_set$rating)

# Now create a table to hold the RMSE and MSE values. This table 
# will be table 5 in the report.
PredComp <- data_frame(Model = c("Model 1","Model 2","Model 3"),
                       "Method" = c("Average Only Model","Movie Effect Model","User Effect Model"), 
                       RMSE= c(naive_rmse,MovieEffectRMSE,UserEffectRMSE))

## Here we regularize the movie and user effect combinations by running a
##  cross-validation on the train_set to select the best lambda tuning parameter
# Run cross-validation on train_set
lambdas <- seq(0,10,0.25)
rmses <- sapply(lambdas,function(l){
  mu <- mean(train_set$rating)
  b_i <- train_set %>% group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  b_u <- train_set %>% left_join(b_i,by='movieId') %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  predicted_ratings <-
    test_set %>%
    left_join(b_i,by='movieId') %>%
    left_join(b_u,by='userId') %>%
    mutate(pred = mu + b_i + b_u) %>%
    .$pred
  return(RMSE(predicted_ratings,test_set$rating))
})

# Now pick the best lambda
lambda_opt <- lambdas[which.min(rmses)]

# Now we apply the best lambda to regularize the movie and user effects
AverageRate <- mean(train_set$rating)

RegMovieEffect <- train_set %>% group_by(movieId) %>%
  summarize(b_i = sum(rating - AverageRate)/(n()+lambda_opt))

RegUserEffect <- train_set %>% left_join(RegMovieEffect,by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - AverageRate)/(n()+lambda_opt))

# Now create a dataframe called RegMoviePred which will store the movie predictions
RegMoviePred <- test_set %>%
  left_join(RegMovieEffect,by='movieId') %>%
  left_join(RegUserEffect,by='userId') %>%
  mutate(preds = AverageRate+b_i+b_u)

# Finally report RMSE value
RegRMSE <- RMSE(test_set$rating,RegMoviePred$preds)

# Now expand Table 5 to add in the RMSE values for the regularization penalty term
PredComp2 <- data_frame(Model = c("Model 1","Model 2","Model 3","Model 4"),
                        "Method" = c("Average Only Model","Movie Effect Model","User Effect Model","Regularization Model"), 
                        RMSE = c(naive_rmse,MovieEffectRMSE,UserEffectRMSE,RegRMSE))

### Next use the recosystem package to use matrix factorization on train_set ###
# Install the recosystem package
if(!require(recosystem)) install.packages("recosystem", repos = "http://cran.us.r-project.org")
library(recosystem)

set.seed(123,sample.kind="Rounding")

# From train_set and test_set only select userId, movieId, and rating as these are
# the columns needed to run recosystem

train_set_small <- train_set %>% select(userId,movieId,rating)
test_set_small <- test_set %>% select(userId,movieId,rating)

trainset <- data_memory(user_index = train_set_small$userId,
                        item_index = train_set_small$movieId,
                        rating = train_set_small$rating)

testset <- data_memory(user_index = test_set_small$userId,
                       item_index = test_set_small$movieId,
                       rating = test_set_small$rating)

# Now we need to create the model object, defined as r, by calling
# the Reco() function
r <- Reco()

# Then select ideal tuning parameters by running r$tune() on the
# trainset. Save the tuning parameters in object called opts. Note
# that this step takes time to execute. Recommendations for which parameters to tune
# and what values to use for the "nthread" and "niter" parameters comes from the
# following R blog on the recosystem package: 
# https://www.r-bloggers.com/2016/07/recosystem-recommender-system-using-parallel-matrix-factorization/. This blog gives a brief overview of how the recosystem package
# works.
opts <- r$tune(trainset,opts = list(dim = c(10,20,30),
                                    costp_l2 = c(0.01,0.1),
                                    costq_l2 = c(0.01,0.1),
                                    costp_l1 = 0,
                                    costq_l1 = 0,
                                    lrate = c(0.01,0.1),
                                    nthread = 4,
                                    niter = 10))

# Then train the model on trainset using the optimal tuning
# paramenters stored in opts
r$train(trainset,opts = c(opts$min,nthread = 4,niter = 20))

# Predict movies based on testset
MoviePredReco <- r$predict(testset,out_memory())

# Report RMSE value
RMSE_Reco <- RMSE(test_set$rating,MoviePredReco)

# Create new table reporting RMSE values from linear modeling and
# matrix factorization
PredComp3 <- data_frame(Model = c("Model 1","Model 2","Model 3","Model 4",
                                  "Matrix Factorization"),
                        Method = c("Average Only Model","Movie Effect Model",
                                   "User Effect Model","Regularization Model",
                                   "Recosystem"), 
                        RMSE = c(naive_rmse,MovieEffectRMSE,UserEffectRMSE,RegRMSE,
                                 RMSE_Reco))

## The results of the training indicates matrix factorization is the best machine
## learning approach to build a movie recommendation. So, we now apply the
## recosystem package on final_holdout_set which is our test set and use edx as
## the training set

set.seed(123,sample.kind="Rounding")

# Here edx is our training set and final_holdout_test is our test set. From these
# two datasets we select userId, movieId, and rating

edx_small <- edx %>% select(userId,movieId,rating)
final_holdout_test_small <- final_holdout_test %>% select(userId,movieId,rating)

edx_train <- data_memory(user_index = edx_small$userId,
                         item_index = edx_small$movieId,
                         rating = edx_small$rating)

FinalTest <- data_memory(user_index = final_holdout_test_small$userId,
                         item_index = final_holdout_test_small$movieId,
                         rating = final_holdout_test_small$rating)

# Create the model object, r
r <- Reco()

# Then select ideal tuning parameters by running r$tune() on edx_train. Save
# the tuning parameters in object called opts. Note this this step takes time to
# execute.
opts <- r$tune(edx_train,opts = list(dim = c(10,20,30),
                                     costp_l2 = c(0.01,0.1),
                                     costq_l2 = c(0.01,0.1),
                                     costp_l1 = 0,
                                     costq_l1 = 0,
                                     lrate = c(0.01,0.1),
                                     nthread = 4,
                                     niter = 10))

# Then train the model on trainset using the optimal tuning paramenters stored
# in opts
r$train(edx_train,opts = c(opts$min,nthread = 4,niter = 20))

# Predict movies based on FinalTest
FinalMoviePredReco <- r$predict(FinalTest,out_memory())

# Report RMSE value
FinalRMSE_Reco <- RMSE(final_holdout_test$rating,FinalMoviePredReco)
FinalRMSE_Reco