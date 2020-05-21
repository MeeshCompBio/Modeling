# check if package is installed, if not run it
packages <- c("ISLR")
if (length(setdiff(packages, rownames(installed.packages()))) > 0) {
  install.packages(setdiff(packages, rownames(installed.packages())))  
}

library(ISLR)

# get the list of datasets
datasets <- ls("package:ISLR")

for (data in datasets){
  data_set <- eval(parse(text=paste("ISLR", data, sep = "::")))
  fname <- paste("Data/", data, ".csv", sep = "")
  # need to split this one in train test sets
  if (data == "Khan"){
    train <- data_set$xtrain
    train_fname <- paste("Data/", "train_", data, ".csv", sep = "")
    test <- data_set$xtest
    test_fname <- paste("Data/", "test_", data, ".csv", sep = "")
    write.csv(train, train_fname, row.names = FALSE)
    write.csv(test, test_fname, row.names = FALSE)
  }
  else {
    write.csv(data_set, fname, row.names = FALSE)
  }
  
}