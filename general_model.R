# model for Upvotes prediction


library(keras)
library(tfruns)

FLAGS <- flags(
  flag_numeric("regularization_l2", 0.0001),
  flag_numeric("learning_rate", 0.0001),
  flag_numeric("regularization_l1", 0.0001)
)


# Inputs to model
tag_input <- layer_input(shape = 1, name = "tag")
reput_input <- layer_input(shape = 1, name = "reput")
user_input <- layer_input(shape = 1, name = "user")
other_vars_input <- layer_input(shape = 2, name = "other-vars")

# Embedding layers
tag_layer <- tag_input %>% 
            layer_embedding(input_dim = 4, 
                            output_dim = 2, 
                            input_length = 1, 
                            name = "tag_embedding", 
                            trainable = TRUE) %>%
            layer_flatten()

reput_layer <- reput_input %>% 
            layer_embedding(input_dim = 51,
                            output_dim = 25,
                            input_length = 1,
                            name = "reputation_embedding",
                            trainable = TRUE) %>%
            layer_flatten()

user_layer <- user_input %>%
            layer_embedding(input_dim = 48653,
                            output_dim = 50,
                            input_length = 1,
                            name = "user_embeddings",
                            trainable = TRUE, batch_size = 32) %>%
            layer_flatten()


# Concatenating model and adding more layers
combined_model <- layer_concatenate(c(tag_layer, reput_layer, user_layer, other_vars_input), name = "concatination_layer") %>%
  layer_dense(256, activation = "relu") %>%
  layer_dense(128, activation = "relu") %>%
  layer_dense(64, activation = "relu") %>%
  layer_dense(5, activation = "relu") %>%
  layer_dense(1, activation = "linear")
 

model <- keras_model(inputs = c(tag_input, reput_input, user_input, other_vars_input), outputs = combined_model)



inputs <- list(as.matrix(as.integer(as.factor(upvote_train$Tag))),
               as.matrix(as.integer(as.factor(upvote_train$reput_group))),
               as.matrix(as.integer(as.factor(upvote_train$Username))),
               as.matrix(upvote_train[, c("Answers", "Views")])
             )

opt <- optimizer_rmsprop(lr = 0.0001)

model %>% 
  compile(loss = "mse",
                  optimizer = opt,
                  metric = "mean_absolute_error")


model %>%
  fit(x = inputs,
      y = as.matrix(upvote_train$Upvotes),
      epochs = 100,
      batch_size = 128)



upvote_train <- arrange(as.data.frame(upvote_train), Upvotes)


