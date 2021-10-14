library(ggplot2)
library(dplyr)
library(tidyr)
library(data.table)
library(GGally)
library(purrr)
library(cluster)
library("mice")
library(fossil)
library(kernlab)

set.seed(1351308)

cw_data <- read.csv('/Users/lorenzwolf/Desktop/MSc Statistics/Electives/Machine Learning/Coursework 2/CW2Data.csv')
head(cw_data)
summary(cw_data)



get_clust_assignment <- function(dist_method, linkage_method) {
  hclust(dist(X, method=dist_method), method=linkage_method)
}

# perform hierarchichal clustering using different arguments
hclust_args <- expand.grid(dist_method=c("euclidean", "manhattan"), linkage_method=c("complete", "single", "average", "ward.D2"))
hclust_list <- pmap(hclust_args, get_clust_assignment)

# plot the resulting dendrograms
par(mfrow=c(4,2))
for(i in 1:length(hclust_list)) {
  plot(hclust_list[[i]],
       main=sprintf("%s distance and %s linkage", hclust_args[i,1], hclust_args[i,2]),
       xlab="State")
  rect.hclust(hclust_list[[i]], k = 14, border = 2:n_clust) # add rectangle
}



# find best hierarchical clustering
set.seed(2)
X <- t(scale(features))
#X <- scale(X)

num_clusters <- seq(2,27)
E_dist <- dist(X, method='euclidean')
M_dist <- dist(X, method='manhattan')
silhouette_mat <- matrix(0, nrow=length(num_clusters), ncol=length(hclust_list))
for(j in 1:length(num_clusters)){
  for(i in 1:length(hclust_list)) {
    grp <- cutree(hclust_list[[i]], k = num_clusters[j])
    
    if (hclust_args[i,1]=="euclidean"){
      s_score <- summary(silhouette(grp, E_dist))$avg.width
    }
    else {
      s_score <- summary(silhouette(grp, M_dist))$avg.width
    }
    silhouette_mat[j,i] <- s_score
  }
}
which(silhouette_mat == max(silhouette_mat), arr.ind = TRUE)

plot(2:27, silhouette_mat[,8], xlab='Number of cluster', ylab='Average silhouette width', main='Silhouette score vs number of clusters')
lines(2:27, silhouette_mat[,8], lty=2)
lines(c(14,14), c(0,0.2), lty=2)

# regression imputation
imp <- mice(scale(features), method = "norm.nob", m = 1) # Impute data
X <- complete(imp)
silhouette_mat

A <- na.omit(X)
km <- kmeans(A, 2, nstart=10)
pca <- prcomp(A, rank. = 4)


df <- as.data.frame(cbind(pca$x, km$cluster))
colnames(df)[[5]] <- "cluster_label"
df$cluster_label <- as.factor(df$cluster_label)

ggpairs(df, columns=1:4, aes(colour=cluster_label), progress=FALSE)


# plot true labels
df_test <- as.data.frame(cbind(pca$x, cw_data$z))
colnames(df_test)[[5]] <- "cluster_label"
df_test$cluster_label <- as.factor(df_test$cluster_label)

ggplot(df_test, aes(x=PC1 , y=PC2, color=cluster_label)) +
  geom_point() + ggtitle('PC2 vs PC1 with true labels')

df_test <- as.data.frame(cbind(pca$x, km$cluster))
colnames(df_test)[[5]] <- "cluster_label"
df_test$cluster_label <- as.factor(df_test$cluster_label)

ggplot(df_test, aes(x=PC1 , y=PC2, color=cluster_label)) +
  geom_point()+ ggtitle('PC2 vs PC1 with kmeans labels')

# ran index
true_labels <- cw_data$z
ri <- rand.index(true_labels, km$cluster)

print(ri)

# adj rand index
adj.rand.index(true_labels+1, km$cluster)
# compare to random
# compare z to a random set
set.seed(1351308)
rand_random = replicate(400, rand.index(true_labels, sample(c(0,1), length(true_labels), replace=TRUE)))
hist(rand_random, breaks=50, main='Rand Index for random labels', xlab='Rand Index')
abline(v = ri, col='red', lty=2, lwd=3)

#kernel kmeans
kern_X <- as.matrix(X)
kernel_km <- kkmeans(kern_X, centers = 2, kernel = "rbfdot")
ri_kernel <- rand.index(true_labels, kernel_km)
ri_kernel

df_test <- as.data.frame(cbind(pca$x, -1*(kernel_km-3)))
colnames(df_test)[[5]] <- "cluster_label"
df_test$cluster_label <- as.factor(df_test$cluster_label)

ggplot(df_test, aes(x=PC1 , y=PC2, color=cluster_label)) +
  geom_point()+ ggtitle('PC2 vs PC1 with kernel kmeans labels')

adj.rand.index(true_labels+1, kernel_km)

