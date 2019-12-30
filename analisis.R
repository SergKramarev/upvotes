# Algorithm for analyzing data that was downloaded from Analitics Vidhya
# Full description of data and all conditions available on  
# https://datahack.analyticsvidhya.com/contest/enigma-codefest-machine-learning-1/

library(ggplot2)
library(dplyr)
library(keras)

upvote_train <- read.csv("train_NIR5Yl1.csv", 
                         stringsAsFactors = FALSE, header = TRUE)
upvotes_test <- read.csv("test_8i3B3FC.csv", 
                         stringsAsFactors = FALSE, header = TRUE)

tmp <- data.frame(n = 1:length(letters), let = letters)
tmp <- subset(tmp, let %in% upvote_train$Tag)
upvote_train <- merge(upvote_train, tmp, by.x = "Tag", by.y = "let")

upvote_train_subset_large <- subset(upvote_train, upvote_train$n %in% c(19, 16, 10, 8, 3, 1))

quant <- quantile(upvote_train_subset_large$Reputation, probs = c(seq(from = 0, to = 0.975, by = 0.025), seq(0.98, 1, 0.002)) )
upvote_train_subset_large$reput_group <- cut(upvote_train_subset_large$Reputation, quant)
upvote_train_subset_large[is.na(upvote_train_subset_large$reput_group), "reput_group"] <- "(0,3]"



avg_upvotes_by_username <- upvote_train_subset_large %>%
  group_by(Username) %>%
  summarise(avg_by_Username = mean(Upvotes))


avg_upvotes_by_reput <- upvote_train_subset_large %>%
  group_by(reput_group) %>%
  summarise(avg_by_reput = mean(Upvotes))


upvote_train_subset_large <- merge(upvote_train_subset_large, avg_upvotes_by_username)
upvote_train_subset_large <- merge(upvote_train_subset_large, avg_upvotes_by_reput)


y <- select(upvote_train_subset_large, Upvotes)
x <- select(upvote_train_subset_large, n, Reputation, Answers, Views, avg_by_Username, avg_by_reput)
x <- scale(x)

x <- keras_array(x)
y <- keras_array(y)


model <- keras_model_sequential()
model %>%
  layer_dense(units = 80, activation = "relu", input_shape = 6) %>%
  layer_dense(units = 40, activation = "relu") %>%
  layer_dense(units = 20, activation = "relu") %>%
  layer_dense(units = 10, activation = "relu") %>%
  layer_dense(units = 5, activation = "relu") %>%
  layer_dense(units = 1, activation = "relu")

optimizer <- optimizer_sgd(lr = 0.0001, momentum = 0.9)

model %>% 
  compile(loss = "msle", 
          optimizer = optimizer, 
          metric = "mean_absolute_error")


model %>% 
  fit(x = x, 
      y = as.matrix(y), 
      epochs = 500, 
      batch_size = 128,
      validation_split = 0.2
  )































###############################################################################

# First we need to create embeddings for Tag and Username for the first model
# for determination of Upvote group 

# 1. Tag embeddings

# 1.1 Dataset preparation, creating training and Validation set

tag_embed <- balance_dataset(upvote_train, "up_group")

x_train_tag <- tag_embed$Train_set
x_test_tag <- tag_embed$Train_set

#1.2 Creating levels and labels for training
levels_up <- levels(upvote_train$up_group) 

y_train <- array(0, dim= c(nrow(x_train_tag), length(levels_up)))

for (i in 1:nrow(y_train)){
  y_train[i,] <- as.integer(levels_up == x_train_tag[i, "up_group"])
}

y_test <- array(0, dim = c(nrow(x_test_tag), length(levels_up)))
for (i in 1:nrow(y_test)){
  y_test[i,] <- as.integer(levels_up == x_test_tag[i, "up_group"])
}


x_train_tag <- x_train_tag[, "Tag"]
x_test_tag <- x_test_tag[, "Tag"]

x_train_tag <- keras_array(as.integer(as.factor(x_train_tag)))
x_test_tag <- keras_array(as.integer(as.factor(x_test_tag)))

y_test <- keras_array(y_test)
y_train <- keras_array(y_train)


# 1.3 Creating and training model for Tag embeddings

input_dim <- length(levels(as.factor(upvote_train$Tag)))

model_tag_embed <- keras_model_sequential()

model_tag_embed %>%
  layer_embedding(input_dim = input_dim + 1, output_dim = 5, input_length = 1, name = "embedding_tag", trainable = TRUE) %>%
  layer_flatten()  %>%
  layer_dense(units = 80, activation = "relu") %>%
  layer_dense(units = 40, activation = "relu") %>%
  layer_dense(units = 20, activation = "relu") %>%
  layer_dense(units = 10, activation = "relu") %>%
  layer_dense(units = 5, activation = "relu") %>%
  layer_dense(units = length(levels_up), activation = "softmax")

optimizer <- optimizer_rmsprop(lr = 0.0001)

model_tag_embed %>% 
  compile(loss = "categorical_crossentropy", 
          optimizer = optimizer, 
          metric = "accuracy")

model_tag_embed %>% 
  fit(x = x_train_tag, 
      y = y_train, 
      epochs = 50, 
      batch_size = 128,
      validation_data = list(x_test_tag,
                             y_test)
  )

# Creating table with Tag embeddings

layer <- get_layer(model_tag_embed, "embedding_tag")
embeddings <- data.frame(layer$get_weights()[[1]])
embeddings$name <- c("none", levels(factor(upvote_train$Tag)))
for (i in 1:length(names(embeddings))){
  names(embeddings)[i] <- paste(names(embeddings)[i], "Tag_emb", sep = "_")
  
}
names(embeddings)[6] <- "Tag"



################################################################################




upvote_train <- merge(upvote_train, tmp, by.x = "Tag", by.y = "lev")
quant <- quantile(upvote_train$Reputation, probs = c(seq(from = 0, to = 0.975, by = 0.025), seq(0.98, 1, 0.001)) )
upvote_train$reput_group <- cut(upvote_train$Reputation, quant)
upvote_train[is.na(upvote_train$reput_group), "reput_group"] <- "(0,3]"



upvote_train_subset1 <- filter(upvote_train, n %in% c(19, 16, 10, 8, 3, 1))

upvote_train_subset1$Reputation <- scale(log(upvote_train_subset1$Reputation +1))
upvote_train_subset1$Answers <- scale(sqrt(upvote_train_subset1$Answers))
upvote_train_subset1$Views <- scale(log(upvote_train_subset1$Views + 1))
upvote_train_subset1$Upvotes <- scale(upvote_train_subset1$Upvotes)






get_rep_slopes <- function(data = NULL, column = "reput_group"){
  coeff <- list()
  groups <- levels(data[, column])
  for (group in groups){
    tmp <- data[data$reput_group == group, ]
    regression <- lm(Upvotes~Views, tmp)
    coeff[[group]] <- summary(regression)$coefficients
    
  }
  return(coeff)
  
}

x <- get_rep_slopes(upvote_train_subset1, "reput_group")

intercepts <- data.frame(intercept = sapply(x, "[[", 1))
intercepts$reput_group <- row.names(intercepts)
slopes <- data.frame(slope = sapply(x, "[[", 2))
slopes$reput_group <- row.names(slopes)

upvote_train_subset1 <- merge(upvote_train_subset1, intercepts)
upvote_train_subset1 <- merge(upvote_train_subset1, slopes)

upvote_train_subset1$Upvotes_pred <- max(-2, upvote_train_subset1$intercept + upvote_train_subset1$Views*upvote_train_subset1$slope)
upvote_train_subset1$n <- scale(upvote_train_subset1$n)

y <- select(upvote_train_subset1, Upvotes)

x <- select(upvote_train_subset1, -Username, -Tag, -ID, -reput_group, -Upvotes)

x$Reputation <- scale(log(x$Reputation + 1))
x$Answers <- scale(sqrt(x$Answers))


y <- as.matrix.data.frame(y)
x <- as.matrix.data.frame(x)

x <- keras_array(x)
y <- keras_array(y)


model <- keras_model_sequential()
model %>%
  layer_dense(units = 80, activation = "relu", input_shape = 7) %>%
  layer_dense(units = 40, activation = "relu") %>%
  layer_dense(units = 20, activation = "relu") %>%
  layer_dense(units = 10, activation = "relu") %>%
  layer_dense(units = 5, activation = "relu") %>%
  layer_dense(units = 1, activation = "sigmoid")

optimizer <- optimizer_rmsprop(lr = 0.0001)

model %>% 
  compile(loss = "msle", 
          optimizer = optimizer, 
          metric = "mse")


model %>% 
  fit(x = x, 
      y = y, 
      epochs = 500, 
      batch_size = 128,
      validation_split = 0.2
  )






























 ################################################################################

# Embeddings for Esernames that are in subset1

levels_user <- levels(upvote_train_subset1$Username) 

x_train_username <- as.character(upvote_train_subset1[, "Username"])

x_train_username <- keras_array(as.integer(as.factor(x_train_username)))

y_train <- upvote_train_subset1[, "Upvotes"]

y_train <- keras_array(y_train)


input_dim <- length(levels(as.factor(as.character(upvote_train_subset1$Username))))

model_tag_username <- keras_model_sequential()

model_tag_username %>%
  layer_embedding(input_dim = input_dim + 1, output_dim = 50, input_length = 1, name = "embedding_username", trainable = TRUE) %>%
  layer_flatten()  %>%
  layer_dense(units = 80, activation = "relu") %>%
  layer_dense(units = 40, activation = "relu") %>%
  layer_dense(units = 20, activation = "relu") %>%
  layer_dense(units = 10, activation = "relu") %>%
  layer_dense(units = 5, activation = "relu") %>%
  layer_dense(units = 1, activation = "linear")

optimizer <- optimizer_rmsprop(lr = 0.0001)

model_tag_username %>% 
  compile(loss = "mse", 
          optimizer = optimizer, 
          metric = "mean_absolute_error")

model_tag_username %>% 
  fit(x = x_train_username, 
      y = y_train, 
      epochs = 100, 
      batch_size = 128
  )

layer <- get_layer(model_tag_username, "embedding_username")
embeddings1 <- data.frame(layer$get_weights()[[1]])
embeddings1$name <- c("none", levels(factor(username_embed$Train_set$Username)))
for (i in 1:length(names(embeddings1))){
  names(embeddings1)[i] <- paste(names(embeddings1)[i], "Username_emb", sep = "_")
  
}
names(embeddings1)[51] <- "Username"







upvote_lebels <- select(upvote_train_subset1, Upvotes)
upvote_var <- select(upvote_train_subset1, -Tag, -ID, -Reputation, -Username, -ratio, -ratio_ans_vie, -Upvotes)

upvote_var <- as.matrix.data.frame(upvote_var)
upvote_lebels <- as.matrix.data.frame(upvote_lebels)

upvote_lebels <- keras_array(upvote_lebels)
upvote_var <- keras_array(upvote_var)

model1 <- keras_model_sequential()
model1 %>%
  layer_dense(units = 80, activation = "relu", 4) %>%
  layer_dense(units = 40, activation = "relu") %>%
  layer_dense(units = 20, activation = "relu") %>%
  layer_dense(units = 10, activation = "relu") %>%
  layer_dense(units = 5, activation = "relu") %>%
  layer_dense(units = 1, activation = "linear")

optimizer <- optimizer_rmsprop(lr = 0.0001)

model1 %>% 
  compile(loss = "mse", 
          optimizer = optimizer, 
          metric = "mean_absolute_error")



model1 %>% 
  fit(x = upvote_var, 
      y = upvote_lebels, 
      epochs = 500, 
      batch_size = 128
  )





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

quant1 <- quantile(upvote_train$Upvotes, probs = c(seq(from = 0, to = 0.6, by = 0.05), seq(from = 0.625, to = 0.975, by = 0.025), seq(0.98, 0.998, 0.002), 0.9985, 0.9999, 0.9995, 1) )
upvote_train$upvotes_group <- cut(upvote_train$Upvotes, quant1)
upvote_train[is.na(upvote_train$reput_group), "reput_group"] <- "(0,1]"


x <- balance_dataset(upvote_train, "upvotes_group")

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

y_te <- select(x_test, upvotes_group)
y_tr <- select(x_train, upvotes_group)

levels <- levels(upvote_train$upvotes_group)
y_train <- array(0, dim = c(nrow(y_tr), length(levels)))

for (i in 1:nrow(y_tr)){
  y_train[i,] <- as.integer(levels == y_tr[i, 1])
}

y_test <- array(0, dim = c(nrow(y_te), length(levels)))

for (i in 1:nrow(y_te)){
  y_test[i,] <- as.integer(levels == y_t[i, 1])
}







x_test <- select(x_test, -upvotes_group)
x_train <- select(x_train, -upvotes_group)


model <- keras_model_sequential()
model %>%
  layer_dense(units = 80, activation = "relu", input_shape = dim(x_train)[2], kernel_regularizer = regularizer_l2()) %>%
  layer_dense(units = 40, activation = "relu", kernel_regularizer = regularizer_l2()) %>%
  layer_dense(units = 20, activation = "relu", kernel_regularizer = regularizer_l2()) %>%
  layer_dense(units = 10, activation = "relu", kernel_regularizer = regularizer_l2()) %>%
  layer_dense(units = 5, activation = "relu") %>%
  layer_dense(units = length(levels), activation = "softmax")

optimizer <- optimizer_rmsprop(lr = 0.0001)

model %>% 
  compile(loss = "categorical_crossentropy", 
          optimizer = optimizer, 
          metric = "accuracy")

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
 