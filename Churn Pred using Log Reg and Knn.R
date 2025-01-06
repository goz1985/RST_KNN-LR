#Using Bank Churn data to get back to speed on my research which looks into enhancing 
# feature selection for rough set theory for KNN, then interpreting results of KNN using Logistic regression
library(dplyr)
library(ggplot2)
library(caret)
library(tidyr)
library(RoughSets)
library(printr)
library(rpart)
library(car)
library(ROSE)

data <- read.csv("D:/Datasets/Bank_churn/Customer-Churn-Records.csv")
str(data)
# Summary statistics of the dataset
summary(data)
# show summary statistics of the variables 
summary(data[, !names(data) %in% c('RowNumber', 'CustomerId', 'Surname')])

# plot box plot
data[, names(data) %in% c('Age', 'Balance', 'CreditScore', 'EstimatedSalary')] %>%
  gather() %>%
  ggplot(aes(value)) +
  facet_wrap(~ key, scales = "free") +
  geom_boxplot() +
  theme(axis.text.x = element_text(size = 7, angle=90), axis.text.y = element_text(size = 7))


# drop non-useful variables
data = data[, !names(data) %in% c('RowNumber', 'CustomerId', 'Surname')]

# data encoding
data$Geography = factor(data$Geography, labels=c(0, 1, 2))
data$Gender = factor(data$Gender, labels=c(0, 1))
data$Card.Type <- as.factor(data$Card.Type)
str(data)

# Convert the target variable (Exited) into a factor
data$Exited <- as.factor(data$Exited)
# Rename levels of the target variable to valid R variable names
levels(data$Exited) <- c("No", "Yes")  # Assuming "0" = No, "1" = Yes

# 3. Normalize numeric variables (optional for machine learning purposes)
numeric_columns <- c("CreditScore", "Age", "Tenure", "Balance", 
                     "NumOfProducts", "EstimatedSalary", "Point.Earned")
preprocess_params <- preProcess(data[, numeric_columns], method = c("center", "scale"))
data[, numeric_columns] <- predict(preprocess_params, data[, numeric_columns])

# View a summary of the cleaned dataset
summary(data)


# Set up cross-validation
set.seed(123)  # For reproducibility
train_control <- trainControl(method = "cv",  # Cross-validation
                              number = 10,   # 10-fold
                              classProbs = TRUE, # To calculate probabilities
                              summaryFunction = twoClassSummary)  # Evaluation metrics


# Fit logistic regression model using cross-validation
logistic_model_cv <- train(Exited ~ CreditScore + Age + Tenure + Balance + NumOfProducts + 
                             HasCrCard + IsActiveMember + EstimatedSalary + Complain +
                             Satisfaction.Score + Geography + Gender + Card.Type,
                           data = data,
                           method = "glm",
                           family = "binomial",
                           trControl = train_control,
                           metric = "ROC")  # Use ROC AUC as the metric

summary(logistic_model_cv)

# Display cross-validation results
print(logistic_model_cv)

# Evaluate the model on cross-validation folds
cat("Cross-validated ROC AUC:", logistic_model_cv$results$ROC, "\n")



# Because I want to compare the performance of KNN and Logistic regression
# Split data into training and testing sets (80% train, 20% test)
set.seed(123)  # For reproducibility
trainIndex <- createDataPartition(data$Exited, p = 0.8, list = FALSE)
train_data <- data[trainIndex, ]
test_data <- data[-trainIndex, ]

# Set up repeated cross-validation
set.seed(123)
train_control <- trainControl(method = "repeatedcv",  # Repeated k-fold cross-validation
                              number = 10,          # 10 folds
                              repeats = 3,          # Repeat 3 times
                              classProbs = TRUE,    # Enable class probabilities
                              summaryFunction = twoClassSummary)  # Use ROC AUC as the metric


# Train the logistic regression model with cross-validation
logistic_model <- train(Exited ~ CreditScore + Age + Tenure + Balance + NumOfProducts + 
                          HasCrCard + IsActiveMember + EstimatedSalary + Complain +
                          Satisfaction.Score + Geography + Gender + Card.Type,
                        data = train_data,
                        method = "glm",
                        family = "binomial",
                        metric = "ROC",  # Optimize for ROC AUC
                        trControl = train_control)
# View the cross-validation results
print(logistic_model)

# Make predictions on the test dataset
test_data$Logistic_Predicted_Class <- predict(logistic_model, newdata = test_data)
test_data$Logistic_Predicted_Prob <- predict(logistic_model, newdata = test_data, type = "prob")[, "Yes"]

# Evaluate the logistic regression model
logistic_confusion_matrix <- confusionMatrix(test_data$Logistic_Predicted_Class, test_data$Exited)
print(logistic_confusion_matrix)


# Train the KNN model with cross-validation
set.seed(123)
knn_model <- train(Exited ~ CreditScore + Age + Tenure + Balance + NumOfProducts + 
                     HasCrCard + IsActiveMember + EstimatedSalary + Complain +
                     Satisfaction.Score + Geography + Gender + Card.Type,
                   data = train_data,
                   method = "knn",
                   metric = "ROC",  # Optimize for ROC AUC
                   trControl = train_control,
                   tuneLength = 10)  # Search across 10 values of k

# View the cross-validation results
print(knn_model)

# Make predictions on the test dataset
test_data$KNN_Predicted_Class <- predict(knn_model, newdata = test_data)
test_data$KNN_Predicted_Prob <- predict(knn_model, newdata = test_data, type = "prob")[, "Yes"]

# Evaluate the KNN model
knn_confusion_matrix <- confusionMatrix(test_data$KNN_Predicted_Class, test_data$Exited)
print(knn_confusion_matrix)



# Select a test point
test_point <- test_data[1, -which(names(test_data) == "Exited")]

# Compute distances from the test point to all training points

# Select numeric columns only
numeric_train_data <- train_data %>% select_if(is.numeric)
numeric_test_point <- test_point %>% select_if(is.numeric)

# Compute distances from the test point to all training points
distances <- apply(numeric_train_data, 1, 
                   function(x) sqrt(sum((x - numeric_test_point)^2)))

# Sort distances and select the k-nearest neighbors
k <- knn_model$bestTune$k
k_nearest_distances <- sort(distances)[1:k]

# Plot the distances of k-nearest neighbors
distance_df <- data.frame(Neighbor = 1:k, Distance = k_nearest_distances)

ggplot(distance_df, aes(x = Neighbor, y = Distance)) +
  geom_point() +
  geom_line() +
  labs(title = paste("Distances to", k, "Nearest Neighbors"),
       x = "Neighbor Index",
       y = "Distance") +
  theme_minimal()

