# Script for dataset balancing. Performs undersampling of rows of most represented class
# 


balance_dataset <- function(dataset = NULL, balancing_column = NULL, deviation = 0.1, balance = TRUE){
  
  if (balance) {
  
  least_represented_class <- min(table(dataset[, balancing_column]))
  most_represented_class <- max(table(dataset[, balancing_column]))
  print(least_represented_class)
  print(most_represented_class)
 
   average <- (least_represented_class + most_represented_class)/2 
 print(average)
  
  selected_rows <- NULL
  Replace <- FALSE
  
  for (i in levels(factor(dataset[, balancing_column]))){
    
    n_rows <- nrow(dataset[which(dataset[,balancing_column] == i), ])
    
    if (n_rows < average*(deviation+1)){
      Replace <- TRUE
    }
    
    rows <- sample(which(dataset[, balancing_column] == i), ceiling(rnorm(1, average, average*deviation)), replace = Replace)
    
    selected_rows <- c(selected_rows, rows)
    
  }
  
  dataset <- dataset[selected_rows, ]
  }
  
  train_set <- sample(1:nrow(dataset), nrow(dataset)*0.7, replace = FALSE)
  
  test <- dataset[-train_set, ]
  train <- dataset[train_set, ]
  
  return(list(Train_set = train, Test_set = test))
}