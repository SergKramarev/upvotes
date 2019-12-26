# Upvotes prediction function

predict_upvotes <- function(model = model, new_data = NULL, train_dataset = NULL){
  # Test data preparation
  # Creating necessary factors 
  
  new_data <- merge(new_data, author_score, all.x = TRUE)
  new_data[is.na(new_data$score), "score"] <- 0
  new_data$score <- as.numeric(as.character(new_data$score))
  
  
  new_data$answer_ration <- new_data$Answers/new_data$Views
  new_data$reput_group <- cut(new_data$Reputation, quant)
  new_data[is.na(new_data$reput_group), "reput_group"] <- "(0,3]"
  new_data <- merge(new_data, coeffs, by.x = "reput_group", by.y = "groups", all.x = TRUE)
  new_data <- select(new_data, -reput_group)
  
  new_data$upvotes_predicted_lm <- new_data$Views*new_data$slope
  new_data <- merge(new_data, tmp1)
  new_data <- select(new_data, -Tag)
  print(head(new_data))
  # Centering and scaling
  
  columns <- c("Reputation", "Answers", "Views", "slope", "answer_ration")
  attr <- data.frame(stringsAsFactors = FALSE, row.names = c("Center", "Scale", "Variable"))
  
  for (column in columns){
    attrib <- attributes(train_dataset[ , column])
    tmp <- data.frame(Center = attrib$`scaled:center`, Scale = attrib$`scaled:scale`, Variable = column)
    attr <- rbind(attr, tmp)
  }
  
  attr <- rbind(attr, additional_attrib)
  attr$Variable <- as.character(attr$Variable)
  attr$Center <- as.numeric(attr$Center)
  attr$Scale <- as.numeric(attr$Scale)
  str(attr)
  #return(attr)
  columns <- c(columns, "score", "upvotes_predicted_lm")
  print(columns)
  
  for (column in columns){
    new_data[, column] <- (new_data[, column] - attr[which(attr$Variable == column), "Center"])/attr[which(attr$Variable == column), "Scale"]
  }
  
  #new_data <- merge(new_data, embeddings, all.x = TRUE)
  new_data <- merge(new_data, embeddings1, all.x = TRUE)
  print(names(new_data))
  new_data[is.na(new_data[ ,"X1_Username_embe"]), 11:60]  <- unlist(embeddings1[embeddings1$Username == "none", 1:50])
  ID <- dplyr::select(new_data, ID)
  new_data <- dplyr::select(new_data, -"Username", -"ID")
  
  print(str(new_data))
  print(sum(is.na(new_data)))
  print(dim(new_data))
  
  result <- predict(model, as.matrix(new_data))
  
  return(dplyr::arrange(data.frame(ID = ID, Upvotes = result), ID))
  
  
}


