#################################################################
#    Sales Prediction & Customer Conversion Analytics Pipeline #
#################################################################

# ===============================================================
# SECTION 1: PREDICTION PROBLEM (SALES REGRESSION)
# ===============================================================

# 1. Data Loading and Initial Data Audit
myData <- read.csv("marketing_sales_data.csv")
str(myData)
sum(is.na(myData))

# 2. Reproducible Train-Validation Partitioning (70/30 Split)
set.seed(163)
training_index <- sample(x = nrow(myData),
                         size = 0.70 * nrow(myData),
                         replace = FALSE)

training_data   <- myData[training_index, ]
validation_data <- myData[-training_index, ]
dim(training_data)
dim(validation_data)

# 3. Full Model Estimation (All Predictors)
Model_1 <- lm(sales ~ spend + price + clicks + promo + emails + reviews,
              data = training_data)
flextable::as_flextable(Model_1)

# 4. Feature Selection & Trimmed Model Estimation
Model_2 <- lm(sales ~ spend + price + clicks + promo,
              data = training_data)
flextable::as_flextable(Model_2)

# 5. Out-of-Sample Validation & Metric Evaluation
source("hamR.R")
prediction_1 <- predict(Model_1, newdata = validation_data)
prediction_2 <- predict(Model_2, newdata = validation_data)

performance_1 <- calculate_metrics_prediction(actual = validation_data$sales,
                                              predicted = prediction_1)
performance_2 <- calculate_metrics_prediction(actual = validation_data$sales,
                                              predicted = prediction_2)

data.frame(Model_1 = performance_1,
           Model_2 = performance_2) |>
  round(4)


# ===============================================================
# SECTION 2: CLASSIFICATION PROBLEM (CONVERSION LOGISTIC MODEL)
# ===============================================================

# 1. Conversion Data Audit
myData <- read.csv("marketing_conversion_data_v2.csv")
str(myData)
sum(is.na(myData))

# 2. Reproducible Train-Validation Partitioning (70/30 Split)
set.seed(163)
train_index <- sample(x = nrow(myData),
                      size = 0.70 * nrow(myData),
                      replace = FALSE)

train_data      <- myData[train_index, ]
validation_data <- myData[-train_index, ]
dim(train_data)
dim(validation_data)

# 3. Full Logistic Regression Model Estimation
Model_1 <- glm(convert ~ spend + clicks + discount + age + email_len + tenure,
               family = binomial,
               data = train_data)
AIC(Model_1)
flextable::as_flextable(Model_1)

# 4. Significant Feature Model Optimization
Model_2 <- glm(convert ~ spend + clicks + discount,
               family = binomial,
               data = train_data)
AIC(Model_2)
flextable::as_flextable(Model_2)

# 5. Classifier Performance & ROC Validation
source("hamR.R")
prediction_prob_1 <- predict(Model_1, newdata = validation_data, type = "response")
prediction_prob_2 <- predict(Model_2, newdata = validation_data, type = "response")

# Model 1 Confusion Matrix & ROC Curve
confusion_matrix(actual = validation_data$convert,
                 predicted = prediction_prob_1)

roc_curve(actual = validation_data$convert,
          predicted = prediction_prob_1)

# Model 2 Confusion Matrix & ROC Curve
confusion_matrix(actual = validation_data$convert,
                 predicted = prediction_prob_2)

roc_curve(actual = validation_data$convert,
          predicted = prediction_prob_2)