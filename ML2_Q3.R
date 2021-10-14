library(latex2exp)

q3 <- read.csv('train_dataQ3.csv')
q3_test <- read.csv('test_dataQ3.csv')

plot(q3$x, q3$y)

set.seed(1351308)
train_ind <- sample(1:dim(q3)[1], 200)
q3_train <- q3[train_ind,]

write.csv(q3_train,'./q3_train.csv', row.names = FALSE)

plot(q3_train$x, q3_train$y)
plot(q3_test$x, q3_test$y)

# compute kernel
# input x is column
kernel <- function(x, p) (x %*% t(x) + 1)^p

# kernel ridge regression
# leave one out cross validation
loo_cv <- function(lambda,p, x, y){
  Num_data_points <- length(y)
  test_mse <- rep(0,Num_data_points)
  K <- kernel(x,p)
  
  for (i in 1:Num_data_points){
    
    K_i <- K[-i,-i]
    
    params <- solve(K_i + lambda * diag(dim(K_i)[2]), y[-i])
    
    test_mse[i] <- (y[i] - sum(K[i,-i]*params))^2
    
  }
  return(mean(test_mse))
}


p_max<-7
kernel_powers <- seq(1,p_max)
lambda_vec <- 2^seq(-7,3,by=0.02)

av_mse <- matrix(0, nrow=length(kernel_powers), ncol=length(lambda_vec))

for (p in kernel_powers){
  av_mse[p,] <- sapply(lambda_vec, function(lambda) loo_cv(lambda, p, q3_train$x, q3_train$y))
}

lambda_vec <- 2^seq(-7,3,by=0.02)
length(lambda_vec)

plot(lambda_vec, av_mse[1,], log='x', ylim=c(0.01,0.06), type='l', xlab=TeX('$\\lambda$'), ylab='Average Validation Error', main= TeX('Average Validation Error vs $\\lambda$'), col=1)
for (p in 2:p_max){
  lines(lambda_vec, av_mse[p,], col=p)
}
legend('topright', legend=c(1:p_max), col=c(1:p_max), lty=rep(1,p_max), cex=0.8)
#legend('topright', legend=c( TeX('p=1'), TeX('p=2'), TeX('p=3'), TeX('p=4')), col=c(1:p_max), lty=rep(1,4), cex=0.8)


inds <- which(av_mse == min(av_mse), arr.ind = TRUE)
inds
lambda_vec[inds[2]]

plot(lambda_vec, av_mse[5,], ylim=c(0.017, 0.019), log='x', type='l', xlab=TeX('$\\lambda$'), ylab='Average Validation Error', main= TeX('Average Validation Error vs $\\lambda$'), col=5)
lines(lambda_vec, av_mse[6,], col=6)
legend('topright', legend=c('p=5','p=6'), col=c(5,6), lty=rep(1,2), cex=0.8)


#p <- 5
#lambda <- 0.007921558

p <- inds[1]
lambda <- lambda_vec[inds[2]]
K <- kernel(q3_train$x,p)

final_params <- solve(K + lambda * diag(dim(K)[2]), q3_train$y)

preds_train <- K%*% final_params

# predictions on test set
K_star <- (q3_test$x %*% t(q3_train$x) + 1)^p
preds_test <- K_star %*% final_params

#plot fit on train set
plot(q3_train$x, q3_train$y, xlab='x', ylab='y', main='Fit on train data')
lines(q3_train$x[order(q3_train$x)], preds_train[order(q3_train$x)], col='red')
#plot fit on test set
plot(q3_test$x, q3_test$y, xlab='x',ylim=c(-0.8,0), ylab='y', main='Fit on test data')
lines(q3_test$x[order(q3_test$x)], preds_test[order(q3_test$x)], col='red')


#test mse
test_mse <- mean((q3_test$y-preds_test)^2)
test_mae <- mean(abs(q3_test$y-preds_test))

test_mse
test_mae

mean((q3_train$y-preds_train)^2)















