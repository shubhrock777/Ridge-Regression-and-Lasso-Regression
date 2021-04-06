
#######################Q1


startup_50 <- read.csv(file.choose())
View(startup_50)


# Reorder the variables
startup_50 <- startup_50[,c(5,1,2,3,4)]

class(startup_50)
attach(startup_50)

library(glmnet)

x <- model.matrix(Profit ~ ., data = startup_50)[ ,-1]
y <- startup_50$Profit

grid <- 10^seq(12, -2, length = 100)
grid

# Ridge Regression

model_ridge <- glmnet(x, y, alpha = 0, lambda = grid)
summary(model_ridge)

cv_fit <- cv.glmnet(x, y, alpha = 0, lambda = grid)
plot(cv_fit)
optimumlambda <- cv_fit$lambda.min
optimumlambda

y_a <- predict(model_ridge, s = optimumlambda, newx = x)
sse <- sum((y_a-y)^2)
sst <- sum((y - mean(y))^2)
rsquared <- 1-sse/sst
rsquared


predict(model_ridge, s = optimumlambda, type="coefficients", newx = x)

pred <- predict(model_ridge, s = optimumlambda, type="coefficients", newx = x)


#root mena squared error
error <- y - y_a

rmse <- sqrt(mean(error**2))
rmse





# Lasso Regression
model_lasso <- glmnet(x, y, alpha = 1, lambda = grid)
summary(model_lasso)

cv_fit_1 <- cv.glmnet(x, y, alpha = 1, lambda = grid)
plot(cv_fit_1)

optimumlambda_1 <- cv_fit_1$lambda.min
optimumlambda_1 

y_a <- predict(model_lasso, s = optimumlambda_1, newx = x)

sse <- sum((y_a-y)^2)

sst <- sum((y - mean(y))^2)
rsquared <- 1-sse/sst
rsquared

predict(model_lasso, s = optimumlambda, type="coefficients", newx = x)
pred <- predict(model_lasso, s = optimumlambda, type="coefficients", newx = x)

#root mena squared error
error <- y - y_a

rmse <- sqrt(mean(error**2))
rmse





