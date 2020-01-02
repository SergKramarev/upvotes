# model for Upvotes prediction


library(keras)
# Inputs to model
tag_input <- layer_input(shape = 1, name = "tag")
reput_input <- layer_input(shape = 1, name = "reput")
user_input <- layer_input(shape = 1, name = "user")
other_vars_input <- layer_input(shape = 2, name = "other-vars")

# Embedding layers
tag_layer <- tag_input %>% 
            layer_embedding(input_dim = 35, 
                            output_dim = 25, 
                            input_length = 1, 
                            name = "tag_embedding", 
                            trainable = TRUE) %>%
            layer_flatten()

reput_layer <- reput_input %>% 
            layer_embedding(input_dim = 25,
                            output_dim = 25,
                            input_length = 1,
                            name = "reputation_embedding",
                            trainable = TRUE) %>%
            layer_flatten()

user_layer <- user_input %>%
            layer_embedding(input_dim = 500,
                            output_dim = 50,
                            input_length = 1,
                            name = "user_embeddings") %>%
            layer_flatten()

# Concatenating model and adding more layers
combined_model <- layer_concatenate(c(tag_layer, reput_layer, user_layer,  other_vars_input), name = "concatination_layer") %>%
  layer_dense(80, activation = "relu") %>%
  layer_dense(40, activation = "relu") %>%
  layer_dense(20, activation = "relu") %>%
  layer_dense(10, activation = "relu") %>%
  layer_dense(5, activation = "relu") %>%
  layer_dense(1, activation = "linear")
 

model <- keras_model(inputs = c(tag_input, reput_input, user_input, other_vars_input), outputs = combined_model)




















max_words <- 20
nb_words <- 1000

text_one_hot <- layer_input(nb_words)
text_as_int <- layer_input(max_words)

vec_1 <- text_one_hot %>%
  layer_dense(100)

vec_2 <- text_as_int %>% layer_embedding(
  input_dim = nb_words, output_dim = 128, 
  input_length = max_words
) %>%
  layer_lstm(128)

out <- layer_concatenate(list(vec_1, vec_2))

model <- keras_model(list(input_1, input_2))
