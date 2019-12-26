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

author_score$score <- scale(author_score$score)
additional_attrib <- data.frame(Center = attributes(author_score$score)$`scaled:center`,
                                Scale = attributes(author_score$score)$`scaled:scale`,
                                Variable = "score",
                                stringsAsFactors = FALSE)
upvote_train <- merge(upvote_train, author_score)


upvote_train$answer_ration <- upvote_train$Answers/upvote_train$Views
upvote_train$Upvotes_scaled <- scale(upvote_train$Upvotes, scale = FALSE)
#upvote_train$Upvotes_log <- log(upvote_train$Upvotes + 1)


quant <- quantile(upvote_train$Reputation, probs = c(seq(from = 0, to = 0.975, by = 0.025), seq(0.98, 1, 0.002)) )
upvote_train$reput_group <- cut(upvote_train$Reputation, quant)
upvote_train[is.na(upvote_train$reput_group), "reput_group"] <- "(0,3]"

get_rep_slopes <- function(data = NULL, column = "reput_group"){
  coeff <- NULL
  groups <- levels(data[, column])
  for (group in groups){
    tmp <- data[data$reput_group == group, ]
    regression <- lm(Upvotes~Views - 1, tmp)$coefficients
    coeff <- rbind(coeff, regression)
  }
  coeff <- cbind(coeff, groups)
  rownames(coeff) <- NULL
  coeff <- data.frame(coeff, stringsAsFactors = FALSE)
  coeff$Views <- as.numeric(coeff$Views)
  names(coeff)[1] <- "slope"
  return(coeff)
  
}

coeffs <- get_rep_slopes(upvote_train, "reput_group")

# avg_view_in_group <- 0
# for (i in 1:length(quant)-1){
#   avg_view_in_group[i] <- (quant[i+1]+quant[i])/2 
# }
# 


# names(coeffs) <- c("Intercept", "Slope", "reput_group")
# coeffs$Intercept <- as.numeric(as.character(coeffs$Intercept))
# coeffs$Slope <- as.numeric(as.character(coeffs$Slope))
upvote_train <- merge(upvote_train, coeffs, by.x = "reput_group", by.y = "groups", all.x = TRUE)
upvote_train$upvotes_predicted_lm <- upvote_train$Views*upvote_train$slope

upvote_train$upvotes_predicted_lm <- scale(upvote_train$upvotes_predicted_lm)
additional_attrib[2, ] <- c(attributes(upvote_train$upvotes_predicted_lm)$`scaled:center`,
                            attributes(upvote_train$upvotes_predicted_lm)$`scaled:scale`,
                            "upvotes_predicted_lm")


tmp1 <- upvote_train %>% group_by(Tag) %>% summarise(mean = mean(Upvotes), sd = sd(Upvotes))
tmp1$x <- tmp1$mean/tmp1$sd
tmp1 <- select(tmp1, -mean, -sd)
upvote_train <- merge(upvote_train, tmp1)


x <- balance_dataset(upvote_train, "Upvotes_group")

# Balancing dataset

x <- balance_dataset(upvote_train, "Tag", balance = FALSE)

x_test <- x$Test_set
x_train <- x$Train_set

##############################################################################
# Creation of embeddings of Tag column

# Try to make Tag embeddings more meanable
tmp <- upvote_train %>%
  group_by(Tag, reput_group) %>%
  summarise(sum_upvotes = sum(Upvotes), 
            mean_upvotes = mean(Upvotes))

tmp$sum_upvotes <- scale(tmp$sum_upvotes)
tmp$mean_upvotes <- scale(tmp$mean_upvotes)

x <- balance_dataset(tmp, "Tag", balance = FALSE)
x_train <- x$Train_set
x_test <- x$Test_set


x_train$Upvotes_log <- log(x_train$Upvotes + 1)
x_test$Upvotes_log <- log(x_test$Upvotes + 1)

x_train$Upvotes_scaled <- scale(x_train$Upvotes)
x_test$Upvotes_scaled <- scale(x_test$Upvotes)

model <- keras_model_sequential()
input_dim <- length(levels(as.factor(x_train$Tag)))

model %>% 
  layer_embedding(input_dim = input_dim + 1, output_dim = 4, input_length = 1, name = "embedding", trainable = TRUE) %>%
  layer_flatten()  %>%
  layer_dense(units = 40, activation = "relu") %>%
  layer_dense(units = 10, activation = "relu") %>%
  layer_dense(units = 5, activation = "relu") %>%
  layer_dense(units = 1, activation = "linear", trainable = FALSE)

optimizer <- optimizer_rmsprop(lr = 0.001)
model %>% 
  compile(loss = "mse", optimizer = optimizer, metric="mean_absolute_error")
model %>% 
  fit(x = as.matrix(as.integer(factor(unlist(x_train[, "Tag"])))), 
      y = as.matrix(unlist(x_train[, "mean_upvotes"])), 
      epochs = 150, 
      batch_size = 128
      # validation_data = list(as.matrix(as.integer(factor(unlist(x_test[, "Tag"]))),
      #                          as.matrix(unlist(x_test[, "mean_upvotes"]))))
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

tmp <- upvote_train %>%
  group_by(Username) %>%
  summarise(sum_upvotes = sum(Upvotes), 
            mean_upvotes = mean(Upvotes))

tmp$sum_upvotes <- scale(tmp$sum_upvotes)
tmp$mean_upvotes <- scale(tmp$mean_upvotes)





model <- keras_model_sequential()
input_dim <- length(levels(as.factor(upvote_train$Username)))

model %>% 
  layer_embedding(input_dim = input_dim + 1, output_dim = 50, input_length = 1, name="embedding1", trainable = TRUE) %>%
  layer_flatten()  %>%
  layer_dense(units = 40, activation = "relu") %>%
  layer_dense(units = 10, activation = "relu") %>%
  layer_dense(units=1, activation = "linear", trainable = FALSE)

optimizer <- optimizer_rmsprop(lr = 0.001)
model %>% 
  compile(loss = "mse", optimizer = optimizer, metric="mean_absolute_error")
model %>% 
  fit(x = as.matrix(as.integer(factor(unlist(tmp[, "Username"])))), 
      y = as.matrix(unlist(tmp[, "mean_upvotes"])), 
      epochs = 50, 
      batch_size = 128
      # validation_data = list(as.matrix(as.integer(factor(x_test[, "Username"]))),
      #                        as.matrix(x_test[, "Upvotes_scaled"])),
      # verbose = 2
      )


layer <- get_layer(model, "embedding1")
embeddings1 <- data.frame(layer$get_weights()[[1]])
embeddings1$name <- c("none", levels(factor(upvote_train$Username)))
for (i in 1:length(names(embeddings1))){
  names(embeddings1)[i] <- paste(names(embeddings1)[i], "Username_embe", sep = "_")
  
}
names(embeddings1)[51] <- "Username"

################################################################################

x_test <- merge(x_test, embeddings)
x_test <- merge(x_test, embeddings1)

x_train <- merge(x_train, embeddings)
x_train <- merge(x_train, embeddings1)


x_test <- select(x_test, -ID, -Upvotes, -Tag, -Username, -reput_group, -Tag)
x_train <- select(x_train, -ID, -Upvotes, -Tag, -Username, -reput_group, -Tag)

x_train$Reputation <- scale(x_train$Reputation)
x_train$Answers <- scale(x_train$Answers)
x_train$Views <- scale(x_train$Views)
x_train$answer_ration <- scale(x_train$answer_ration)
x_train$slope <- scale(x_train$slope)

x_test$Reputation <- scale(x_test$Reputation)
x_test$Answers <- scale(x_test$Answers)
x_test$Views <- scale(x_test$Views)
x_test$answer_ration <- scale(x_test$answer_ration)
x_test$slope <- scale(x_test$slope)

y_test <- select(x_test, Upvotes_scaled)
y_train <- select(x_train, Upvotes_scaled)

x_test <- select(x_test, -Upvotes_scaled)
x_train <- select(x_train, -Upvotes_scaled)


model <- keras_model_sequential()
model %>%
  layer_dense(units = 80, activation = "relu", input_shape = dim(x_train)[2], kernel_regularizer = regularizer_l2()) %>%
  layer_dense(units = 40, activation = "relu", kernel_regularizer = regularizer_l2()) %>%
  layer_dense(units = 20, activation = "relu", kernel_regularizer = regularizer_l2()) %>%
  layer_dense(units = 10, activation = "relu", kernel_regularizer = regularizer_l2()) %>%
  layer_dense(units = 5, activation = "relu") %>%
  layer_dense(units = 1, activation = "linear")

optimizer <- optimizer_rmsprop(lr = 0.0001)

model %>% 
  compile(loss = "mean_absolute_error", 
          optimizer = optimizer, 
          metric = "mse")

call_backs <- list(
  callback_csv_logger("training_progress_log.csv"),
  callback_model_checkpoint("model_saved_log.hdf5", monitor = "loss")
#  callback_reduce_lr_on_plateau(monitor = "loss", factor = 0.1, min_delta = 0.01, min_lr = 0.00001, verbose = 1)
  
)

model %>% 
  fit(x = as.matrix(x_train), 
      y = as.matrix(y_train), 
      epochs = 50, 
      batch_size = 128,
      callbacks = call_backs,
      validation_data = list(as.matrix(x_test),
                             as.matrix(y_test))
      )
 