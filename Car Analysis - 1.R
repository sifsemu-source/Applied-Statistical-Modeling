# Set up
library(car)
library(lmtest)
library(sandwich)
library(ggplot2)
options(scipen = 999)
set.seed(42)

# Section 1: Data Exploration
df <- read.csv("autoscout_car_sales.csv", stringsAsFactors = FALSE)
cat("Rows:", nrow(df), " Columns:", ncol(df), "\n")
str(df)
head(df)

# Section 2: Data Quality Control
colSums(is.na(df))  

summary(df[, c("price", "km", "age", "hp_kW",
               "Displacement_cc", "Weight_kg", "cons_comb")])

# frequencies of the key categorical variables
lapply(df[, c("Gearing_Type", "Fuel", "make_model", "body_type", "Type")], table)

# Section 3: Data Pre-processing and Cleaning
d <- df

# Hold the two transmission types
d <- subset(d, Gearing_Type %in% c("Manual", "Automatic"))

# Filter the two fuel types
d <- subset(d, Fuel %in% c("Benzine", "Diesel"))

# Remove rare car model less than 30 listings
keep_models <- names(which(table(d$make_model) >= 30))
d <- subset(d, make_model %in% keep_models)

# Trim 1% excess tails on price and mileage
p_lim  <- quantile(d$price, c(0.01, 0.99))
km_lim <- quantile(d$km,    c(0.01, 0.99))
d <- subset(d, price >= p_lim[1] & price <= p_lim[2] &
              km    >= km_lim[1] & km    <= km_lim[2])

# Feature engineering
d$Automatic <- ifelse(d$Gearing_Type == "Automatic", 1, 0)  
d$log_price <- log(d$price)                                
d$km_k      <- d$km / 1000                                 

# Change categorical variables to factors
for (col in c("Fuel", "make_model", "body_type")) d[[col]] <- as.factor(d[[col]])

cat("Rows after cleaning:", nrow(d), "\n")   

# SECTION 4: EXPLORATORY DATA ANALYSIS

# Summary for the numeric columns
summary(d[, c("price", "km", "age", "hp_kW")])
sapply(d[, c("price", "km", "age", "hp_kW")], sd)

# Checking price by transmission
aggregate(price ~ Gearing_Type, data = d,
          FUN = function(x) c(n = length(x), mean = mean(x),
                              median = median(x), sd = sd(x)))

# Visualization
par(mfrow = c(2, 2))
hist(d$price, breaks = 40, col = "blue",
     main = "Price distribution", xlab = "Price (EUR)")
hist(d$log_price, breaks = 40, col = "darkorange",
     main = "Log(price) distribution", xlab = "log(Price)")
boxplot(price ~ Gearing_Type, data = d, col = c("wheat", "lightgreen"),
        main = "Price by transmission type", ylab = "Price (EUR)")
plot(d$km_k, d$price, pch = 20, cex = 0.4, col = rgb(0, 0, 0, 0.15),
     main = "Price vs mileage", xlab = "Mileage (1000 km)", ylab = "Price (EUR)")
par(mfrow = c(1, 1))
# Correlations between the numerical values
cor_vars <- c("price", "km", "age", "hp_kW",
              "Displacement_cc", "Weight_kg")
round(cor(d[, cor_vars]), 2)

# Section 5: Application and Interpretation of Inferential Statistics
tt_result <- t.test(price ~ Gearing_Type, data = d)
tt_result

grp <- split(d$price, d$Gearing_Type)
na <- length(grp$Automatic); nm <- length(grp$Manual)
sp <- sqrt(((na - 1) * var(grp$Automatic) + (nm - 1) * var(grp$Manual)) / (na + nm - 2))
cohens_d <- (mean(grp$Automatic) - mean(grp$Manual)) / sp
cat("Cohen's d =", round(cohens_d, 3), "\n")

# Section 6: Sampling
set.seed(42)
n <- nrow(d)
train_idx <- sample(seq_len(n), size = floor(0.8 * n))
train <- d[train_idx, ]
test  <- d[-train_idx, ]
cat("Train:", nrow(train), " Test:", nrow(test), "\n")

model <- lm(log_price ~ Automatic + km_k + age + hp_kW + Fuel + body_type + make_model, data=train)

# Section 7: Model Validation
par(mfrow = c(2, 2)); plot(model); par(mfrow = c(1, 1))  

# Multi-collinearity check
vif(model)

# Normality of residuals
shapiro.test(sample(residuals(model), 5000))

# Breusch-Pagan test
bptest(model)

# Section 8: Calculate heteroscedasticity - Robust Inference

coeftest(model, vcov = vcovHC(model, type = "HC3"))

# Section 10: Prediction on the Hold-Out-Set
pred_price <- exp(predict(model, newdata = test))   
resid_eur  <- test$price - pred_price
rmse    <- sqrt(mean(resid_eur^2))
mae     <- mean(abs(resid_eur))
r2_test <- 1 - sum(resid_eur^2) / sum((test$price - mean(test$price))^2)
# Print metrics
rmse
mae
r2_test
