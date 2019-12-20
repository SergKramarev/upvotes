# Upvotes prediction function

predict_upvotes <- function(model = model, new_data = NULL, train_dataset = NULL){
  # Test data preparation
  # Centering and scaling
  new_data$answer_ration <- new_data$Answers/new_data$Views
  columns <- c("Reputation", "Answers", "Views")
  attr <- data.frame(stringsAsFactors = FALSE, row.names = c("Center", "Scale", "Variable"))
  
  for (column in columns){
    attrib <- attributes(train_dataset[ , column])
    tmp <- data.frame(Center = attrib$`scaled:center`, Scale = attrib$`scaled:scale`, Variable = column)
    attr <- rbind(attr, tmp)
  }
  
  #return(attr)
  
  
  for (column in columns){
    new_data[, column] <- (new_data[, column] - attr[which(attr$Variable == column), "Center"])/attr[which(attr$Variable == column), "Scale"]
  }
  
  new_data <- merge(new_data, embeddings, all.x = TRUE)
  new_data <- merge(new_data, embeddings1, all.x = TRUE)
  
  new_data[is.na(new_data[ ,"X1_Username_embe"]), 13:62]  <- unlist(embeddings1[embeddings1$Username == "none", 1:50])
  ID <- dplyr::select(new_data, ID)
  new_data <- dplyr::select(new_data, -"Username", -"Tag", -"ID")
  
  print(head(new_data))
  print(sum(is.na(new_data)))
  print(dim(new_data))
  
  result <- predict(model, as.matrix(new_data))
  
  return(dplyr::arrange(data.frame(ID = ID, Upvotes = result), ID))
  
  
}


