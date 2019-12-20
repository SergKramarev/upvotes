# Algorithm for analyzing data that was downloaded from Analitics Vidhya
# Full description of data and all conditions available on  https://datahack.analyticsvidhya.com/contest/enigma-codefest-machine-learning-1/

library(ggplot2)
library(dplyr)
library(keras)

upvote_train <- read.csv("train_NIR5Yl1.csv", stringsAsFactors = FALSE, header = TRUE)
upvotes_test <- read.csv("test_8i3B3FC.csv", stringsAsFactors = FALSE, header = TRUE)

# Author's score calculation. The idea about author's score is part of the 
# answered questions divided by number of views normalized by number of 
# questions issued by author

author_score <- upvote_train %>%
  group_by(Username) %>%
  summarise(score = sum(Answers/Views)/n())

upvote_train <- merge(upvote_train, author_score)

upvote_train$answer_ration <- upvote_train$Answers/upvote_train$Views
upvote_train$Upvotes_scaled <- scale(upvote_train$Upvotes)

# Balancing dataset

x <- balance_dataset(upvote_train, "Tag")

x_test <- x$Test_set
x_train <- x$Train_set

##############################################################################
# Creation of embeddings of Tag column

x_train$Upvotes_scaled <- scale(x_train$Upvotes)
x_test$Upvotes_scaled <- scale(x_test$Upvotes)

model <- keras_model_sequential()
input_dim <- length(levels(as.factor(x_train$Tag)))

model %>% 
  layer_embedding(input_dim = input_dim + 1, output_dim = 5, input_length = 1, name="embedding", trainable = TRUE) %>%
  layer_flatten()  %>%
  layer_dense(units = 40, activation = "relu") %>%
  layer_dense(units = 10, activation = "relu") %>%
  layer_dense(units=1, activation = "linear")

optimizer <- optimizer_rmsprop(lr = 0.0001)
model %>% 
  compile(loss = "mse", optimizer = optimizer, metric="mean_absolute_error")
model %>% 
  fit(x = as.matrix(as.integer(factor(x_train[, "Tag"]))), 
      y = as.matrix(x_train[, "Upvotes_scaled"]), 
      epochs = 50, 
      batch_size = 16,
      validation_data = list(as.matrix(as.integer(factor(x_test[, "Tag"]))),
                             as.matrix(x_test[, "Upvotes_scaled"])),
      verbose = 2
  )







layer <- get_layer(model, "embedding")
embeddings <- data.frame(layer$get_weights()[[1]])
embeddings$name <- c("none", levels(factor(upvote_train$Tag)))
for (i in 1:length(names(embeddings))){
  names(embeddings)[i] <- paste(names(embeddings)[i], "Tag_embe", sep = "_")
  
}
names(embeddings)[6] <- "Tag"

################################################################################
# Creation of Username Embedding
model <- keras_model_sequential()
input_dim <- length(levels(as.factor(upvote_train$Username)))

model %>% 
  layer_embedding(input_dim = input_dim + 1, output_dim = 50, input_length = 1, name="embedding1", trainable = TRUE) %>%
  layer_flatten()  %>%
  layer_dense(units = 40, activation = "relu") %>%
  layer_dense(units = 10, activation = "relu") %>%
  layer_dense(units=1, activation = "linear")

optimizer <- optimizer_rmsprop(lr = 0.001)
model %>% 
  compile(loss = "mse", optimizer = optimizer, metric="mean_absolute_error")
model %>% 
  fit(x = as.matrix(as.integer(factor(upvote_train[, "Username"]))), 
      y = as.matrix(upvote_train[, "Upvotes_scaled"]), 
      epochs = 50, 
      batch_size = 128,
      validation_data = list(as.matrix(as.integer(factor(x_test[, "Username"]))),
                             as.matrix(x_test[, "Upvotes_scaled"])),
      verbose = 2
      )


layer <- get_layer(model, "embedding1")
embeddings1 <- data.frame(layer$get_weights()[[1]])
embeddings1$name <- c("none", levels(factor(upvote_train$Username)))
for (i in 1:length(names(embeddings1))){
  names(embeddings1)[i] <- paste(names(embeddings1)[i], "Username_embe", sep = "_")
  
}
names(embeddings1)[51] <- "Username"

x_test <- merge(x_test, embeddings)
x_test <- merge(x_test, embeddings1)

x_train <- merge(x_train, embeddings)
x_train <- merge(x_train, embeddings1)


x_test <- select(x_test, -ID, -Upvotes, -Tag, -Username)
x_train <- select(x_train, -ID, -Upvotes, -Tag, -Username)

x_train$Reputation <- scale(x_train$Reputation)
x_train$Answers <- scale(x_train$Answers)
x_train$Views <- scale(x_train$Views)
x_train$answer_ration <- scale(x_train$answer_ration)

x_test$Reputation <- scale(x_test$Reputation)
x_test$Answers <- scale(x_test$Answers)
x_test$Views <- scale(x_test$Views)
x_test$answer_ration <- scale(x_test$answer_ration)

y_test <- select(x_test, Upvotes_scaled)
y_train <- select(x_train, Upvotes_scaled)

x_test <- select(x_test, -Upvotes_scaled)
x_train <- select(x_train, -Upvotes_scaled)


model <- keras_model_sequential()
model %>%
  layer_dense(units = 80, activation = "relu", input_shape = dim(x)[2]) %>%
  layer_dense(units = 40, activation = "relu") %>%
  layer_dense(units = 10, activation = "relu") %>%
  layer_dense(units = 1, activation = "linear")

optimizer <- optimizer_rmsprop(lr = 0.0001)

model %>% 
  compile(loss = "mse", 
          optimizer = optimizer, 
          metric = "mean_absolute_error")

call_backs <- list(
  callback_csv_logger("training_progress_log.csv"),
  callback_model_checkpoint("model_saved_log.hdf5", monitor = "loss")
#  callback_reduce_lr_on_plateau(monitor = "loss", factor = 0.1, min_delta = 0.01, min_lr = 0.00001, verbose = 1)
  
)

model %>% 
  fit(x = as.matrix(x_train), 
      y = as.matrix(y_train), 
      epochs = 500, 
      batch_size = 64,
      callbacks = call_backs,
      validation_data = list(as.matrix(x_test),
                             as.matrix(y_test)),
      verbose = 2)
 