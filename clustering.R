setwd("D:/STUDIA/Sem_9/Data_mining/Project2")

set.seed(123)

library(cluster)
library(factoextra)
library(clusterSim)
library(ppclust)
library(fclust)
library(NbClust)
library("dbscan")
library(e1071)
library(clValid)
library(mclust)
library(xtable)

my_labels <- c("ID", "Diagnosis", 
               "Radius1", "Texture1", "Perimeter1", 
               "Area1", "Smoothness1", "Compactness1", 
               "Concavity1", "Concave_points1", "Symmetry1", "Fractal_dimension1",
               
               "Radius2", "Texture2", "Perimeter2", 
               "Area2", "Smoothness2", "Compactness2", 
               "Concavity2", "Concave_points2", "Symmetry2", "Fractal_dimension2",
               
               "Radius3", "Texture3", "Perimeter3", 
               "Area3", "Smoothness3", "Compactness3", 
               "Concavity3", "Concave_points3", "Symmetry3", "Fractal_dimension3")

data <- read.csv(file="wdbc.data", header = FALSE, col.names=my_labels, stringsAsFactors = TRUE)
attach(data)

data <- data[,-1]
data.features <- data[,-1]
data.labels <- data[,1]

#M <- 1, B <- 0
data.labels <- as.numeric(data.labels)-1

# library(caret)

# Check correlation
# corrFeat.id <- findCorrelation(cor(data.features))
# corrFeat.names <- findCorrelation(cor(data.features), names=T)

# delete correlation
# data.features <- data.features[, -c(corrFeat.id+1)]

library(NbClust)
NbClust.results.1 <- NbClust(data.features, distance="euclidean", min.nc=2, max.nc=10, method="complete", index="all")

NbClust.results.1$All.index
NbClust.results.1$Best.nc
NbClust.results.1$Best.partition

source("fviz_nbclust_fixed.R") # fixes the bug in factoextra::fviz_nbclust(...)
png("optimal_clusters.png", width= 2048, height = 1536, res = 300)
factoextra::fviz_nbclust(NbClust.results.1) + theme_minimal() + ggtitle("Optimal number of clusters")
dev.off()

library(mclust)

d_clust2 <- Mclust(as.matrix(data.features), G=1:10)
m.best2 <- dim(d_clust2$z)[2]

cat("model-based optimal number of clusters:", m.best2, "\n")
## model-based optimal number of clusters: (removed correlated: 6) (all: 7)
png("BIC_clusters.png", width= 2048, height = 1536, res = 300)
plot(d_clust2$BIC)
dev.off()
# my_labels_1 <- c("Diagnosis", "Radius1", "Texture1", "Perimeter1", 
#                  "Area1", "Smoothness1", "Compactness1", 
#                  "Concavity1", "Concave_points1", "Symmetry1", "Fractal_dimension1")
# 
# my_labels_2 <- c("Diagnosis", "Radius2", "Texture2", "Perimeter2", 
#                  "Area2", "Smoothness2", "Compactness2", 
#                  "Concavity2", "Concave_points2", "Symmetry2", "Fractal_dimension2")
# 
# my_labels_3 <- c("Diagnosis", "Radius3", "Texture3", "Perimeter3", 
#                  "Area3", "Smoothness3", "Compactness3", 
#                  "Concavity3", "Concave_points3", "Symmetry3", "Fractal_dimension3")
# 
# data1 <- data[my_labels_1]
# data2 <- data[my_labels_2]
# data3 <- data[my_labels_3]


# OPTIMAL NUMBER OF GROUPS

# Elbow method
png("elbow_method_hierar.png", width= 2048, height = 1536, res = 300)
fviz_nbclust(data.features, FUNcluster = hcut, method = "wss") +
  geom_vline(xintercept = 2, linetype = 2)+
  labs(subtitle = "Elbow method")
dev.off()

# Silhouette method
png("silhouette_method_hierar.png", width= 2048, height = 1536, res = 300)
fviz_nbclust(data.features, FUNcluster = hcut, method = "silhouette")+
  labs(subtitle = "Silhouette method")
dev.off()

# Gap statistic
# nboot = 50 to keep the function speedy. 
# recommended value: nboot= 500 for your analysis.
# Use verbose = FALSE to hide computing progression.
png("gap_statistic_hierrar.png", width= 2048, height = 1536, res = 300)
fviz_nbclust(data.features, FUNcluster = hcut, nstart = 25,  method = "gap_stat", nboot = 50)+
  labs(subtitle = "Gap statistic method")
dev.off()

# fviz_nbclust(data.features, FUNcluster = cluster::pam, method = "silhouette") # PAM
# fviz_nbclust(data.features, FUNcluster = hcut, method = "silhouette") # hierarchical clustering


####################################

# KMEANS

####################################

kmeans.1 <- kmeans(data.features, centers = 2, iter.max = 10, nstart = 1)

plot(data.features$Radius1, data.features$Radius2, col = kmeans.1$cluster + 1, pch=data.labels)
points(kmeans.1$centers[,c("Radius1","Radius2")],pch=16,cex=1.5,col=4:5)

fviz_cluster(kmeans.1, data.features)

fviz_cluster(kmeans.1, data.features, ellipse.type="euclid") #nie xd

library(ggplot2)
library(kableExtra)
library(dplyr)
as.data.frame(data.features) %>% mutate(Cluster = kmeans.1$cluster) %>% group_by(Cluster) %>% summarise_all("mean") %>% kable() %>% kable_styling()


data.features_df <- as.data.frame(data.features)
data.features_df$cluster <- kmeans.1$cluster
data.features_df$cluster <- as.character(kmeans.1$cluster)

ggplot(data.features_df, aes(x = cluster, y = Radius1)) + 
  geom_boxplot(aes(fill = cluster))

library(GGally)
# Parallel coordiante plots allow us to put each feature on seperate column and lines connecting each column
ggparcoord(data = data.features_df, columns = 1:3, groupColumn = 6, alphaLines = 0.4, title = "Parallel Coordinate Plot", scale = "globalminmax", showPoints = TRUE) + theme(legend.position = "bottom")

# changing "nstart"

k2.1 <- kmeans(data.features, centers = 2, nstart = 1)
k2.2 <- kmeans(data.features, centers = 2, nstart = 2)
k2.5 <- kmeans(data.features, centers = 2, nstart = 5)
k2.10 <- kmeans(data.features, centers = 2, nstart = 10)

# par(mfrow = c(2, 2))
# png("1.png", width= 2048, height = 1536, res = 300)
# plot(data.features, col = k2.1$cluster + 1, main ="nstart = 1")
# points(k2.1$centers, cex = 5, lwd = 2)
# 
# plot(data.features, col = k2.2$cluster + 1, main = "nstart = 2")
# points(k2.2$centers, cex = 5, lwd = 2)
# 
# plot(data.features, col = k2.5$cluster + 1, main = "nstart = 5")
# points(k2.5$centers, cex = 5, lwd = 2)
# 
# plot(data.features, col = k2.10$cluster + 1, main = "nstart = 10")
# points(k2.10$centers, cex = 5, lwd = 2)
# dev.off()
# par(mfrow = c(1, 1))

# ALL THE SAME
par(mfrow = c(2, 2))
png("kmeans_nstarts.png", width= 2048, height = 1536, res = 300)
fviz_cluster(k2.1, data.features)
fviz_cluster(k2.2, data.features)
fviz_cluster(k2.5, data.features)
fviz_cluster(k2.10, data.features)
dev.off()
par(mfrow = c(1, 1))

# changing "centers"

k2 <- kmeans(data.features, centers = 2, nstart = 1)
k3 <- kmeans(data.features, centers = 3, nstart = 1)
k4 <- kmeans(data.features, centers = 4, nstart = 1)
k7 <- kmeans(data.features, centers = 7, nstart = 1)

# par(mfrow = c(2, 2))
# 
# plot(data.features, col = k2$cluster + 1, main ="k = 2")
# points(k2$centers, cex = 5, lwd = 2)
# 
# plot(data.features, col = k3$cluster + 1, main = "k = 3")
# points(k3$centers, cex = 5, lwd = 2)
# 
# plot(data.features, col = k4$cluster + 1, main = "k = 4")
# points(k4$centers, cex = 5, lwd = 2)
# 
# plot(data.features, col = k5$cluster + 1, main = "k = 5")
# points(k5$centers, cex = 5, lwd = 2)
# 
# par(mfrow = c(1, 1))

png("kmeans_clusters_7.png", width= 2048, height = 1536, res = 300)
fviz_cluster(k7, data.features)
dev.off()
fviz_cluster(k3, data.features)
fviz_cluster(k4, data.features)
fviz_cluster(k7, data.features)

# multiple runs for the same parameters
# 
# k11 <- kmeans(data.features, centers = 2, nstart = 1)
# k12 <- kmeans(data.features, centers = 2, nstart = 1)
# k13 <- kmeans(data.features, centers = 2, nstart = 1)
# k14 <- kmeans(data.features, centers = 2, nstart = 1)

# par(mfrow = c(2, 2))
# 
# plot(data.features, col = k11$cluster + 1)
# points(k11$centers, cex = 5, lwd = 2)
# 
# plot(data.features, col = k12$cluster + 1)
# points(k12$centers, cex = 5, lwd = 2)
# 
# plot(data.features, col = k13$cluster + 1)
# points(k13$centers, cex = 5, lwd = 2)
# 
# plot(data.features, col = k14$cluster + 1)
# points(k14$centers, cex = 5, lwd = 2)
# 
# par(mfrow = c(1, 1))


## VALIDATION ##

Within.kmeans   <- c()
Between.kmeans  <- c()
Total.kmeans    <- c()

K.range <- 1:10

for (k in K.range)
{
  print(k)
  kmeans.k  <- kmeans(data.features, centers=k, iter.max=10, nstart=10)
  Within.kmeans  <- c(Within.kmeans, kmeans.k$tot.withinss)	# total within-cluster sum of squares =  sum(withinss))
  Between.kmeans <- c(Between.kmeans, kmeans.k$betweenss) # between-cluster sum of squares
  Total.kmeans   <- c(Total.kmeans,kmeans.k$totss) 	# total sum of squares.
  # remark: Total == Within + Between
}

y.range.kmeans <- range(c(Within.kmeans, Between.kmeans, Total.kmeans))

png("kmeans_dispersion.png", width= 2048, height = 1536, res = 300)
plot(K.range,  Between.kmeans, col="red", type="b", lwd=2, xlab="K", ylim=y.range.kmeans, ylab="B/W")
lines(K.range,  Within.kmeans, col="blue", lwd=2, type="b")
lines(K.range,  Total.kmeans, col="black", lwd=2, type="b")
legend(x='right', legend=c("Total SS (total dispersion)", "Between SS (between-cluster dispersion)","Within SS (within-cluster dispersion)"), lwd=2, col=c("black","blue","red"), bg="azure2", cex=0.7)
grid()
title("Comparison of the within-cluster and between-cluster dispersion")
dev.off()

# Visualize k-means clustering

sil.kmeans.2 <- silhouette(k2$cluster, dist(data.features))
sil.kmeans.3 <- silhouette(k3$cluster, dist(data.features))
sil.kmeans.4 <- silhouette(k4$cluster, dist(data.features))
sil.kmeans.7 <- silhouette(k7$cluster, dist(data.features))

png("kmeans_silh_2.png", width= 2048, height = 1536, res = 300)
fviz_silhouette(sil.kmeans.2, xlab="K-means")
dev.off()

png("kmeans_silh_3.png", width= 2048, height = 1536, res = 300)
fviz_silhouette(sil.kmeans.3, xlab="K-means")
dev.off()

png("kmeans_silh_4.png", width= 2048, height = 1536, res = 300)
fviz_silhouette(sil.kmeans.4, xlab="K-means")
dev.off()

png("kmeans_silh_7.png", width= 2048, height = 1536, res = 300)
fviz_silhouette(sil.kmeans.7, xlab="K-means")
dev.off()

# Cluster labels vs. actual class labels

(tab.kmeans.2 <- table(k2$cluster, data.labels))
(tab.kmeans.3 <- table(k3$cluster, data.labels))
(tab.kmeans.4 <- table(k4$cluster, data.labels))
(tab.kmeans.7 <- table(k7$cluster, data.labels))

# partition agreement
matchClasses(tab.kmeans.2)
compareMatchedClasses(k2$cluster, data.labels)$diag
matchClasses(tab.kmeans.3)
compareMatchedClasses(k3$cluster, data.labels)$diag
matchClasses(tab.kmeans.4)
compareMatchedClasses(k4$cluster, data.labels)$diag
matchClasses(tab.kmeans.7)
compareMatchedClasses(k7$cluster, data.labels)$diag

# Internal validation

internal.validation.kmeans.2 <- clValid(data.features, nClust=2, clMethods="kmeans", validation="internal")
internal.validation.kmeans.3 <- clValid(data.features, nClust=3, clMethods="kmeans", validation="internal")
internal.validation.kmeans.4 <- clValid(data.features, nClust=4, clMethods="kmeans", validation="internal")
internal.validation.kmeans.7 <- clValid(data.features, nClust=7, clMethods="kmeans", validation="internal")

summary(internal.validation)
optimalScores(internal.validation)

par(mfrow = c(2, 2))
plot(internal.validation, legend = FALSE, lwd=2)
plot.new()
legend("center", clusterMethods(internal.validation), col=1:9, lty=1:9, pch=paste(1:9))


### Stability indices (APN, AD, ADM)

stability.validation.kmeans <- clValid(data.features, nClust=K.range, clMethods="kmeans", validation="stability")
summary(stability.validation.kmeans)
optimalScores(stability.validation.kmeans)

## STATISTISC OF CLUSTERS ##

funcs <- c("mean", "sd", "median", "mad")
my_labels_new <- my_labels[c(-1,-2)]

desc.k2 <- cluster.Description(data.features, k2$cluster)

df.k7.1 <- desc.k7[1,,-5]
colnames(df.k7.1) <- funcs
rownames(df.k7.1) <- my_labels_new[]
df.k7.1
data.frame.k7.1 <- data.frame(df.k7.1)

df.k7.2 <- desc.k7[2,,-5]
colnames(df.k7.2) <- funcs
rownames(df.k7.2) <- my_labels_new[]
df.k7.2
data.frame.k7.2 <- data.frame(df.k7.2)

df.k7.3 <- desc.k7[3,,-5]
colnames(df.k7.3) <- funcs
rownames(df.k7.3) <- my_labels_new[]
df.k7.3
data.frame.k7.3 <- data.frame(df.k7.3)

df.k7.4 <- desc.k7[4,,-5]
colnames(df.k7.4) <- funcs
rownames(df.k7.4) <- my_labels_new[]
df.k7.4
data.frame.k7.4 <- data.frame(df.k7.4)

df.k7.5 <- desc.k7[5,,-5]
colnames(df.k7.5) <- funcs
rownames(df.k7.5) <- my_labels_new[]
df.k7.5
data.frame.k7.5 <- data.frame(df.k7.5)

df.k7.6 <- desc.k7[6,,-5]
colnames(df.k7.6) <- funcs
rownames(df.k7.6) <- my_labels_new[]
df.k7.6
data.frame.k7.6 <- data.frame(df.k7.6)

df.k7.7 <- desc.k7[2,,-5]
colnames(df.k7.7) <- funcs
rownames(df.k7.7) <- my_labels_new[]
df.k7.7
data.frame.k7.7 <- data.frame(df.k7.7)


# par(mfrow = c(2,2))
# plot(stability.validation.kmeans, measure=c("APN","AD","ADM"), legend=FALSE, lwd=2)
# plot.new()
# legend("center", clusterMethods(stability.validation.kmeans), col=1:9, lty=1:9, pch=paste(1:9))
# par(mfrow = c(1,1))

####################################

# DISSIMILARITY MATRIX

####################################


data.DissimilarityMatrix <- daisy(data.features)

# Conversion to matrix
data.DissimilarityMatrix.mat <- as.matrix(data.DissimilarityMatrix)

# without ordering
fviz_dist(data.DissimilarityMatrix, order = FALSE)
# after ordering
fviz_dist(data.DissimilarityMatrix, order = TRUE)


####################################

# PAM

####################################

# data.pam3 <- pam(x=data.DissimilarityMatrix.mat, diss=TRUE, k=2)
# 
# plot(data.pam3) # default visualization (note: plot() works differently for quantitative and mixed data types)
# 
# 
# by(Cars93$Type, Cars93.pam3$clustering, FUN=table)
# by(Cars93$Price, Cars93.pam3$clustering, FUN=summary)
# boxplot(Cars93$Price~Cars93.pam3$clustering)


pam1 <- pam(data.features, k=2, nstart = 1)

plot(pam1)

fviz_cluster(pam1)

desc.pam <- cluster.Description(data.features, pam.1$clustering)
names <- c("SL", "SW", "PL", "PW")
funcs <- c("mean", "sd", "median", "mad")

df.1 <- desc[1,,-5]
colnames(df.1) <- funcs
rownames(df.1) <- names
df.1

df.2 <- desc[2,,-5]
colnames(df.2) <- funcs
rownames(df.2) <- names
df.2

plot(data[,3], data[,2], col=pam1$clustering, pch=c(0,1,2))

fviz_cluster(pam1, data.features)

# changing nstart
# THE SAME

pam.1 <- pam(data.features, k=2, nstart = 1)
pam.2 <- pam(data.features, k=2, nstart = 2)
pam.5 <- pam(data.features, k=2, nstart = 5)
pam.10 <- pam(data.features, k=2, nstart = 10)

png("kmeans_nstarts.png", width= 2048, height = 1536, res = 300)
plot(pam.5)
dev.off()

# png("pam_nstarts_all.png", width= 2048, height = 1536, res = 300)
# par(mfrow = c(2, 2))
# plot(data.features, col = pam.1$clustering + 1, main ="nstart = 1")
# plot(data.features, col = pam.2$clustering + 1, main ="nstart = 2")
# plot(data.features, col = pam.5$clustering + 1, main ="nstart = 5")
# plot(data.features, col = pam.10$clustering + 1, main ="nstart = 10")
# par(mfrow = c(1, 1))
# dev.off()

png("pam_nstarts.png", width= 2048, height = 1536, res = 300)
fviz_cluster(pam.1, data.features)
dev.off()

#changing k

pam.1.k2 <- pam(data.features, k=2, nstart = 1)
pam.1.k3 <- pam(data.features, k=3, nstart = 1)
pam.1.k4 <- pam(data.features, k=4, nstart = 1)
pam.1.k7 <- pam(data.features, k=7, nstart = 1)

plot(pam.1.k3)

png("pam_clusters_7.png", width= 2048, height = 1536, res = 300)
fviz_cluster(pam.1.k7, data.features)
dev.off()

par(mfrow = c(2, 2))
plot(data.features, col = pam.1.k2$clustering + 1, main ="k = 2")
plot(data.features, col = pam.1.k3$clustering + 1, main ="k = 3")
plot(data.features, col = pam.1.k4$clustering + 1, main ="k = 4")
plot(data.features, col = pam.1.k5$clustering + 1, main ="k = 5")
par(mfrow = c(1, 1))

# multiple starts

# pam.1.k2.1 <- pam(data.features, k=2, nstart = 1)
# pam.1.k3.2 <- pam(data.features, k=2, nstart = 1)
# pam.1.k4.3 <- pam(data.features, k=2, nstart = 1)
# pam.1.k5.4 <- pam(data.features, k=2, nstart = 1)
# 
# par(mfrow = c(2, 2))
# plot(data.features, col = pam.1.k2.1$clustering + 1, main ="k = 2")
# plot(data.features, col = pam.1.k3.2$clustering + 1, main ="k = 2")
# plot(data.features, col = pam.1.k4.3$clustering + 1, main ="k = 2")
# plot(data.features, col = pam.1.k5.4$clustering + 1, main ="k = 2")
# par(mfrow = c(1, 1))

## VALIDATION ##


# Visualize k-means clustering

sil.pam.2 <- silhouette(pam.1.k2$clustering, dist(data.features))
sil.pam.3 <- silhouette(pam.1.k3$clustering, dist(data.features))
sil.pam.4 <- silhouette(pam.1.k4$clustering, dist(data.features))
sil.pam.7 <- silhouette(pam.1.k7$clustering, dist(data.features))

png("pam_silh_2.png", width= 2048, height = 1536, res = 300)
fviz_silhouette(sil.pam.2, xlab="PAM")
dev.off()

png("pam_silh_3.png", width= 2048, height = 1536, res = 300)
fviz_silhouette(sil.pam.3, xlab="PAM")
dev.off()

png("pam_silh_4.png", width= 2048, height = 1536, res = 300)
fviz_silhouette(sil.pam.4, xlab="PAM")
dev.off()

png("pam_silh_7.png", width= 2048, height = 1536, res = 300)
fviz_silhouette(sil.pam.7, xlab="PAM")
dev.off()

# Cluster labels vs. actual class labels

(tab.pam.2 <- table(pam.1.k2$clustering, data.labels))
(tab.pam.3 <- table(pam.1.k3$clustering, data.labels))
(tab.pam.4 <- table(pam.1.k4$clustering, data.labels))
(tab.pam.7 <- table(pam.1.k7$clustering, data.labels))

# partition agreement
matchClasses(tab.pam.2)
compareMatchedClasses(pam.1.k2$clustering, data.labels)$diag
matchClasses(tab.pam.3)
compareMatchedClasses(pam.1.k3$clustering, data.labels)$diag
matchClasses(tab.pam.4)
compareMatchedClasses(pam.1.k4$clustering, data.labels)$diag
matchClasses(tab.pam.7)
compareMatchedClasses(pam.1.k7$clustering, data.labels)$diag

# Internal validation

internal.validation.pam.2 <- clValid(data.features, nClust=2, clMethods="pam", validation="internal")
internal.validation.pam.3 <- clValid(data.features, nClust=3, clMethods="pam", validation="internal")
internal.validation.pam.4 <- clValid(data.features, nClust=4, clMethods="pam", validation="internal")
internal.validation.pam.7 <- clValid(data.features, nClust=7, clMethods="pam", validation="internal")

summary(internal.validation.pam.2)
optimalScores(internal.validation.pam)

par(mfrow = c(2, 2))
plot(internal.validation.pam.2, legend = FALSE, lwd=2)
plot.new()
legend("center", clusterMethods(internal.validation.pam), col=1:9, lty=1:9, pch=paste(1:9))
par(mfrow = c(1, 1))

### Stability indices (APN, AD, ADM)

stability.validation.pam <- clValid(data.features, nClust=K.range, clMethods="pam", validation="stability")
summary(stability.validation.pam)
optimalScores(stability.validation.pam)


## STATISTISC OF CLUSTERS ##

funcs <- c("mean", "sd", "median", "mad")
my_labels_new <- my_labels[c(-1,-2)]

desc.pam.2 <- cluster.Description(data.features, pam.1.k2$clustering)

df.pam.2.1 <- desc.pam.2[1,,-5]
colnames(df.pam.2.1) <- funcs
rownames(df.pam.2.1) <- my_labels_new[]
df.pam.2.1
data.frame.pam.2.1 <- data.frame(df.pam.2.1)

df.pam.2.2 <- desc.pam.2[2,,-5]
colnames(df.pam.2.2) <- funcs
rownames(df.pam.2.2) <- my_labels_new[]
df.pam.2.2
data.frame.pam.2.2 <- data.frame(df.pam.2.2)

df.pam.3.3 <- desc.pam.3[3,,-5]
colnames(df.pam.3.3) <- funcs
rownames(df.pam.3.3) <- my_labels_new[]
df.pam.3.3
data.frame.pam.3.3 <- data.frame(df.pam.3.3)

df.pam.4.4 <- desc.pam.4[4,,-5]
colnames(df.pam.4.4) <- funcs
rownames(df.pam.4.4) <- my_labels_new[]
df.pam.4.4
data.frame.pam.4.4 <- data.frame(df.pam.4.4)

df.pam.7.5 <- desc.pam.7[5,,-5]
colnames(df.pam.7.5) <- funcs
rownames(df.pam.7.5) <- my_labels_new[]
df.pam.7.5
data.frame.pam.7.5 <- data.frame(df.pam.7.5)

df.pam.7.6 <- desc.pam.7[6,,-5]
colnames(df.pam.7.6) <- funcs
rownames(df.pam.7.6) <- my_labels_new[]
df.pam.7.6
data.frame.pam.7.6 <- data.frame(df.pam.7.6)

df.pam.7.7 <- desc.pam.7[2,,-5]
colnames(df.pam.7.7) <- funcs
rownames(df.pam.7.7) <- my_labels_new[]
df.pam.7.7
data.frame.pam.7.7 <- data.frame(df.pam.7.7)

####################################

# FUZZY C-MEANS

####################################

# centers -> 2, 3, 4, 7
# m -> 2
fcm2 <- fcm(data.features, centers = 2, m = 2, nstart = 1)
fcm3 <- fcm(data.features, centers = 3, m = 2, nstart = 1)
fcm4 <- fcm(data.features, centers = 4, m = 2, nstart = 1)
fcm7 <- fcm(data.features, centers = 7, m = 2, nstart = 1)

fcm2.15 <- fcm(data.features, centers = 2, m = 1.5, nstart = 1)
fcm2.25 <- fcm(data.features, centers = 2, m = 2.5, nstart = 1)

# fcm5 <- fcm(data.features, centers = 2, m = 2,
#             nstart = 5, fixmemb = TRUE)

# png("fcm_clusters.png", width= 2048, height = 1536, res = 300)
# plotcluster(fcm1, cp=1, trans=TRUE)
# dev.off()

fcm2.25 <- ppclust2(fcm2.25, "kmeans")

png("fcm1_clusters_2_m_25.png", width= 2048, height = 1536, res = 300)
fviz_cluster(fcm2.25, data.features, ellipse.type="convex")
dev.off()


# data.features.2 <- iris[,-1]
# iris.features.2 <- iris.features.2[,-1]
# iris.features.2 <- iris.features.2[,-3]
# fcm1.2 <- fcm(iris.features.2, centers = 3, m = 2, nstart = 1)
# 
# d <- cbind(iris.features.2,fcm1.2$u,fcm1.2$cluster)
# d$region <- factor(1:nrow(d))
# 
# library("scatterpie")
# 
# png("iris_fcm1_3.png", width= 2048, height = 1536, res = 300)
# ggplot()  +
#   geom_scatterpie(data = d, alpha = 0.6, aes(x = Petal.Length, y = Petal.Width, group=region), cols = colnames(fcm1.2$u), legend_name = "Membership") + coord_equal()
# dev.off()

# VALIDATION

fcm2.val <- ppclust2(fcm2, "fclust")
fcm3.val <- ppclust2(fcm3, "fclust")
fcm4.val <- ppclust2(fcm4, "fclust")
fcm7.val <- ppclust2(fcm7, "fclust")

fcm2.25.val <- ppclust2(fcm2.25, "fclust")
fcm2.15.val <- ppclust2(fcm2.15, "fclust")

sil.fcm2 <- SIL.F(fcm2.val$Xca, fcm2.val$U)
sil.fcm3 <- SIL.F(fcm3.val$Xca, fcm3.val$U)
sil.fcm4 <- SIL.F(fcm4.val$Xca, fcm4.val$U)
sil.fcm7 <- SIL.F(fcm7.val$Xca, fcm7.val$U)

sil.fcm2.25 <- SIL.F(fcm2.25.val$Xca, fcm2.25.val$U)
sil.fcm2.15 <- SIL.F(fcm2.15.val$Xca, fcm2.15.val$U)

idxpe.fcm2 <- PE(fcm2.val$U)
idxpe.fcm3 <- PE(fcm3.val$U)
idxpe.fcm4 <- PE(fcm4.val$U)
idxpe.fcm7 <- PE(fcm7.val$U)

idxpe.fcm2.25 <- PE(fcm2.25.val$U)
idxpe.fcm2.15 <- PE(fcm2.15.val$U)

idxpc.fcm2 <- PC(fcm2.val$U)
idxpc.fcm3 <- PC(fcm3.val$U)
idxpc.fcm4 <- PC(fcm4.val$U)
idxpc.fcm7 <- PC(fcm7.val$U)

idxpc.fcm2.25 <- PC(fcm2.25.val$U)
idxpc.fcm2.15 <- PC(fcm2.15.val$U)

idxmpc.fcm2 <- MPC(fcm2.val$U)
idxmpc.fcm3 <- MPC(fcm3.val$U)
idxmpc.fcm4 <- MPC(fcm4.val$U)
idxmpc.fcm7 <- MPC(fcm7.val$U)

idxmpc.fcm2.25 <- MPC(fcm2.25.val$U)
idxmpc.fcm2.15 <- MPC(fcm2.15.val$U)

# Cluster labels vs. actual class labels

(tab.fcm.2 <- table(fcm2$cluster, data.labels))
(tab.fcm.3 <- table(fcm3$cluster, data.labels))
(tab.fcm.4 <- table(fcm4$cluster, data.labels))
(tab.fcm.7 <- table(fcm7$cluster, data.labels))

(tab.fcm2.25 <- table(fcm2.25$cluster, data.labels))
(tab.fcm2.15 <- table(fcm2.15$cluster, data.labels))

desc.fcm2 <- cluster.Description(data.features, fcm2$cluster)

df.fcm2.1 <- desc.fcm2[1,,-5]
colnames(df.fcm2.1) <- funcs
rownames(df.fcm2.1) <- my_labels_new[]
df.fcm2.1
#data.frame.fcm2.1 <- data.frame(df.k7.1)

df.fcm2.2 <- desc.fcm2[2,,-5]
colnames(df.fcm2.2) <- funcs
rownames(df.fcm2.2) <- my_labels_new[]
df.k7.2
#data.frame.k7.2 <- data.frame(df.k7.2)

matchClasses(tab.fcm.2)

####################################

# DBSCAN <- DIDN'T WORK

####################################

db.5.4 <- dbscan(data.features, eps = 0.5, minPts = 7)
db.4.4 <- dbscan(data.features, eps = 0.4, minPts = 7)
db.3.4 <- dbscan(data.features, eps = 0.3, minPts = 7)

db.5.4 <- dbscan(data.features, eps = 0.5, minPts = 4)
db.5.4 <- dbscan(data.features, eps = 0.4, minPts = 4)
db.5.4 <- dbscan(data.features, eps = 0.3, minPts = 4)

# png("db1.png", width= 2048, height = 1536, res = 300)
# pairs(data.features, col = db1$cluster + 1L)
# dev.off()

table(db1$cluster, data.labels) 

plot(db1, data.features, main = "DBScan") 
# 
# kNNdistplot(data.features, k = 5)
# abline(h=.5, col = "red", lty=2)


png("db1_2.png", width= 2048, height = 1536, res = 300)
fviz_cluster(db2, data.features, ellipse.type="convex")
dev.off()


noise1 <- db1$cluster==0
clusters1 <- db1$cluster[!noise1]
d1 <- dist(iris.features[!noise1,1:2])
clusterColours1 <- brewer.pal(9,"Set1")

sil1 <- silhouette(clusters1, d1)

png("silh_db1.png", width= 2048, height = 1536, res = 300)
plot(sil1, border=NA, col=sort(clusters1), main="")
dev.off()


####################################

#AGNES

####################################

# Linkage methods:
# method = 'average'  (average distance)
# method = 'complete' (farthest neighbor)
# method = 'single'   (nearest neigbor)
# for more linkage methods see:  ?agnes

agnes.avg      <- agnes(x=data.DissimilarityMatrix.mat, diss=TRUE, method="average")
agnes.single   <- agnes(x=data.DissimilarityMatrix.mat, diss=TRUE, method="single")
agnes.complete <- agnes(x=data.DissimilarityMatrix.mat, diss=TRUE, method="complete")

agnes.avg      <- agnes(x=data.features, method="average")
agnes.single   <- agnes(x=data.features, method="single")
agnes.complete <- agnes(x=data.features, method="complete")

png("agnes_avg_basic.png", width= 2048, height = 1536, res = 300)
plot(agnes.avg,which.plots=2,main="AGNES: average linkage")
dev.off()

png("agnes_single_basic.png", width= 2048, height = 1536, res = 300)
plot(agnes.single,which.plots=2,main="AGNES: single linkage")
dev.off()

png("agnes_complete_basic.png", width= 2048, height = 1536, res = 300)
plot(agnes.complete,which.plots=2, main="AGNES: complete linkage")
dev.off()

plot(agnes.avg, which.plots=2, cex=0.5)

(agnes.avg.k2 <- cutree(agnes.avg, k=2))  # 2 clusters
(agnes.avg.k3 <- cutree(agnes.avg, k=3))  # 3 clusters
(agnes.avg.k4 <- cutree(agnes.avg, k=4))  # 4 clusters
(agnes.avg.k7 <- cutree(agnes.avg, k=7))

(agnes.single.k2 <- cutree(agnes.single, k=2))  # 2 clusters
(agnes.single.k3 <- cutree(agnes.single, k=3))  # 3 clusters
(agnes.single.k4 <- cutree(agnes.single, k=4))  # 4 clusters
(agnes.single.k7 <- cutree(agnes.single, k=7))

(agnes.complete.k2 <- cutree(agnes.complete, k=2))  # 2 clusters
(agnes.complete.k3 <- cutree(agnes.complete, k=3))  # 3 clusters
(agnes.complete.k4 <- cutree(agnes.complete, k=4))  # 4 clusters
(agnes.complete.k7 <- cutree(agnes.complete, k=7))

table(agnes.avg.k2)
table(agnes.avg.k3)
table(agnes.avg.k4)
table(agnes.avg.k7)

table(agnes.single.k2)
table(agnes.single.k3)
table(agnes.single.k4)
table(agnes.single.k7)

table(agnes.complete.k2)
table(agnes.complete.k3)
table(agnes.complete.k4)
table(agnes.complete.k7)

### Silhouette indices for hierarchical clustering

sil.agnes.k2 <- silhouette(x=agnes.avg.k2, dist=DissimilarityMatrix.mat)

# the average silhouette value for each cluster can be obtained as follows:
# (avg.sil.agnes.k2 <- summary(sil.agnes.k2)$clus.avg.widths)

# the average silhouette value for the whole partition can be obtained by taking an average over all K cluster silhouette averages.
# mean(avg.sil.agnes.k2)

# Alternatively one can compute an average over all n object silhouette indices
# mean(sil.agnes.k2[,3])

# standard dendrogram
fviz_dend(agnes.avg, cex=0.3, main="Dendrogram - average linkage")
# horizontal layout
fviz_dend(agnes.avg, horiz=TRUE, cex=0.3, main="Dendrogram - average linkage")

# dendrogram + partition into K clusters
png("agnes_single_k2.png", width= 2048, height = 1536, res = 300)
fviz_dend(agnes.single, k=2, cex=0.2)
dev.off()

png("agnes_single_k3.png", width= 2048, height = 1536, res = 300)
fviz_dend(agnes.single, k=3, cex=0.2)
dev.off()

png("agnes_single_k4.png", width= 2048, height = 1536, res = 300)
fviz_dend(agnes.single, k=4, cex=0.2)
dev.off()

png("agnes_single_k7.png", width= 2048, height = 1536, res = 300)
fviz_dend(agnes.single, k=7, cex=0.2)
dev.off()

# circular dendrogram
fviz_dend(agnes.avg, type="circular", cex=0.4, k=6,  main="Dendrogram - average linkage")

# 2D scatter plot using PCA (only for quantitative data)
fviz_dend(agnes.avg, k=3)
fviz_cluster(list(data=data.features, cluster=agnes.avg))


# colors = real classes
labels.colors <- data.labels + 2

# AVERAGE LINKAGE

# we order the colors according to the order of the objects in the dendrogram
colors.objects.avg <- labels.colors[agnes.avg$order]

# dendrogram + colors (actual classes)
dendrogram.agnes.avg <- as.dendrogram(agnes.avg)
fviz_dend(dendrogram.agnes.avg, cex=0.3, label_cols=colors.objects.avg, main="Colors = actual classes")

# Comparison: partition into 3 clusters vs. actual classes
png("agnes_avg_k2_actual.png", width= 2048, height = 1536, res = 300)
fviz_dend(dendrogram.agnes.avg, cex=0.3, k=2,  label_cols=colors.objects.avg, k_colors=c("grey","orange"),main="Partition into 2 clusters vs. actual class labels", rect = T, lower_rect=-0.5)
dev.off()

png("agnes_avg_k3_actual.png", width= 2048, height = 1536, res = 300)
fviz_dend(dendrogram.agnes.avg, cex=0.3, k=3,  label_cols=colors.objects.avg, k_colors=c("grey","orange", "blue"),main="Partition into 3 clusters vs. actual class labels", rect = T, lower_rect=-0.5)
dev.off()

png("agnes_avg_k4_actual.png", width= 2048, height = 1536, res = 300)
fviz_dend(dendrogram.agnes.avg, cex=0.3, k=4,  label_cols=colors.objects.avg, k_colors=c("grey","orange", "blue", "darkorchid"),main="Partition into 4 clusters vs. actual class labels", rect = T, lower_rect=-0.5)
dev.off()

png("agnes_avg_k7_actual.png", width= 2048, height = 1536, res = 300)
fviz_dend(dendrogram.agnes.avg, cex=0.3, k=7,  label_cols=colors.objects.avg, k_colors=c("grey","orange", "blue", "darkorchid", "cyan", "darkolivegreen", "darksalmon"),main="Partition into 7 clusters vs. actual class labels", rect = T, lower_rect=-0.5)
dev.off()

# SINGLE LINKAGE

# we order the colors according to the order of the objects in the dendrogram
colors.objects.single <- labels.colors[agnes.single$order]

# dendrogram + colors (actual classes)
dendrogram.agnes.single <- as.dendrogram(agnes.single)
fviz_dend(dendrogram.agnes.single, cex=0.3, label_cols=colors.objects.single, main="Colors = actual classes")

# Comparison: partition into 3 clusters vs. actual classes
png("agnes_single_k2_actual.png", width= 2048, height = 1536, res = 300)
fviz_dend(dendrogram.agnes.single, cex=0.3, k=2,  label_cols=colors.objects.single, k_colors=c("grey","orange"),main="Partition into 2 clusters vs. actual class labels", rect = T, lower_rect=-0.5)
dev.off()

png("agnes_single_k3_actual.png", width= 2048, height = 1536, res = 300)
fviz_dend(dendrogram.agnes.single, cex=0.3, k=3,  label_cols=colors.objects.single, k_colors=c("grey","orange", "blue"),main="Partition into 3 clusters vs. actual class labels", rect = T, lower_rect=-0.5)
dev.off()

png("agnes_single_k4_actual.png", width= 2048, height = 1536, res = 300)
fviz_dend(dendrogram.agnes.single, cex=0.3, k=4,  label_cols=colors.objects.single, k_colors=c("grey","orange", "blue", "darkorchid"),main="Partition into 4 clusters vs. actual class labels", rect = T, lower_rect=-0.5)
dev.off()

png("agnes_single_k7_actual.png", width= 2048, height = 1536, res = 300)
fviz_dend(dendrogram.agnes.single, cex=0.3, k=7,  label_cols=colors.objects.single, k_colors=c("grey","orange", "blue", "darkorchid", "cyan", "darkolivegreen", "darksalmon"), main="Partition into 7 clusters vs. actual class labels", rect = T, lower_rect=-0.5)
dev.off()

# COMPLETE LINKAGE

# we order the colors according to the order of the objects in the dendrogram
colors.objects.complete <- labels.colors[agnes.complete$order]

# dendrogram + colors (actual classes)
dendrogram.agnes.complete <- as.dendrogram(agnes.complete)
fviz_dend(dendrogram.agnes.complete, cex=0.3, label_cols=colors.objects.complete, main="Colors = actual classes")

# Comparison: partition into 3 clusters vs. actual classes
png("agnes_complete_k2_actual.png", width= 2048, height = 1536, res = 300)
fviz_dend(dendrogram.agnes.complete, cex=0.3, k=2,  label_cols=colors.objects.complete, k_colors=c("grey","orange"),main="Partition into 2 clusters vs. actual class labels", rect = T, lower_rect=-0.5)
dev.off()

png("agnes_complete_k3_actual.png", width= 2048, height = 1536, res = 300)
fviz_dend(dendrogram.agnes.complete, cex=0.3, k=3,  label_cols=colors.objects.complete, k_colors=c("grey","orange", "blue"),main="Partition into 3 clusters vs. actual class labels", rect = T, lower_rect=-0.5)
dev.off()

png("agnes_complete_k4_actual.png", width= 2048, height = 1536, res = 300)
fviz_dend(dendrogram.agnes.complete, cex=0.3, k=4,  label_cols=colors.objects.complete, k_colors=c("grey","orange", "blue", "darkorchid"),main="Partition into 4 clusters vs. actual class labels", rect = T, lower_rect=-0.5)
dev.off()

png("agnes_complete_k7_actual.png", width= 2048, height = 1536, res = 300)
fviz_dend(dendrogram.agnes.complete, cex=0.3, k=7,  label_cols=colors.objects.complete, k_colors=c("grey","orange", "blue", "darkorchid", "cyan", "darkolivegreen", "darksalmon"), main="Partition into 7 clusters vs. actual class labels", rect = T, lower_rect=-0.5)
dev.off()

## VALIDATION ##

# AVERAGE LINKAGE

sil.agnes.avg.2 <- silhouette(agnes.avg.k2, dist(data.features))
sil.agnes.avg.3 <- silhouette(agnes.avg.k3, dist(data.features))
sil.agnes.avg.4 <- silhouette(agnes.avg.k4, dist(data.features))
sil.agnes.avg.7 <- silhouette(agnes.avg.k7, dist(data.features))

png("agnes_avg_silh_2.png", width= 2048, height = 1536, res = 300)
fviz_silhouette(sil.agnes.avg.2, xlab="AGNES (avg)")
dev.off()

png("agnes_avg_silh_3.png", width= 2048, height = 1536, res = 300)
fviz_silhouette(sil.agnes.avg.3, xlab="AGNES (avg)")
dev.off()

png("agnes_avg_silh_4.png", width= 2048, height = 1536, res = 300)
fviz_silhouette(sil.agnes.avg.4, xlab="AGNES (avg)")
dev.off()

png("agnes_avg_silh_7.png", width= 2048, height = 1536, res = 300)
fviz_silhouette(sil.agnes.avg.7, xlab="AGNES (avg)")
dev.off()

# Cluster labels vs. actual class labels

tab.agnes.avg.2 <- table(agnes.avg.k2, data.labels)
tab.agnes.avg.3 <- table(agnes.avg.k3, data.labels)
tab.agnes.avg.4 <- table(agnes.avg.k4, data.labels)
tab.agnes.avg.7 <- table(agnes.avg.k7, data.labels)

# partition agreement
matchClasses(tab.agnes.avg.2)
compareMatchedClasses(agnes.avg.k2, data.labels)$diag
matchClasses(tab.agnes.avg.3)
compareMatchedClasses(agnes.avg.k3, data.labels)$diag
matchClasses(tab.agnes.avg.4)
compareMatchedClasses(agnes.avg.k4, data.labels)$diag
matchClasses(tab.agnes.avg.7)
compareMatchedClasses(agnes.avg.k7, data.labels)$diag

# Internal validation

internal.validation.agnes.avg.2 <- clValid(data.features, nClust=2, clMethods="agnes", method="average", validation="internal")
internal.validation.agnes.avg.3 <- clValid(data.features, nClust=3, clMethods="agnes", method="average", validation="internal")
internal.validation.agnes.avg.4 <- clValid(data.features, nClust=4, clMethods="agnes", method="average", validation="internal")
internal.validation.agnes.avg.7 <- clValid(data.features, nClust=7, clMethods="agnes", method="average", validation="internal")


summary(internal.validation.agnes.avg.2)
optimalScores(internal.validation.agnes.avg)

### Stability indices (APN, AD, ADM)

stability.validation.agnes.avg <- clValid(data.features, nClust=K.range, clMethods="agnes",method = "average", validation="stability")
summary(stability.validation.agnes.avg)
optimalScores(stability.validation.agnes.avg)


# COMPLETE LINKAGE

sil.agnes.complete.2 <- silhouette(agnes.complete.k2, dist(data.features))
sil.agnes.complete.3 <- silhouette(agnes.complete.k3, dist(data.features))
sil.agnes.complete.4 <- silhouette(agnes.complete.k4, dist(data.features))
sil.agnes.complete.7 <- silhouette(agnes.complete.k7, dist(data.features))

png("agnes_complete_silh_2.png", width= 2048, height = 1536, res = 300)
fviz_silhouette(sil.agnes.complete.2, xlab="AGNES (complete)")
dev.off()

png("agnes_complete_silh_3.png", width= 2048, height = 1536, res = 300)
fviz_silhouette(sil.agnes.complete.3, xlab="AGNES (complete)")
dev.off()

png("agnes_complete_silh_4.png", width= 2048, height = 1536, res = 300)
fviz_silhouette(sil.agnes.complete.4, xlab="AGNES (complete)")
dev.off()

png("agnes_complete_silh_7.png", width= 2048, height = 1536, res = 300)
fviz_silhouette(sil.agnes.complete.7, xlab="AGNES (complete)")
dev.off()

# Cluster labels vs. actual class labels

tab.agnes.complete.2 <- table(agnes.complete.k2, data.labels)
tab.agnes.complete.3 <- table(agnes.complete.k3, data.labels)
tab.agnes.complete.4 <- table(agnes.complete.k4, data.labels)
tab.agnes.complete.7 <- table(agnes.complete.k7, data.labels)

# partition agreement
matchClasses(tab.agnes.complete.2)
compareMatchedClasses(agnes.complete.k2, data.labels)$diag
matchClasses(tab.agnes.complete.3)
compareMatchedClasses(agnes.complete.k3, data.labels)$diag
matchClasses(tab.agnes.complete.4)
compareMatchedClasses(agnes.complete.k4, data.labels)$diag
matchClasses(tab.agnes.complete.7)
compareMatchedClasses(agnes.complete.k7, data.labels)$diag

# Internal validation

internal.validation.agnes.complete.2 <- clValid(data.features, nClust=2, clMethods="agnes", method="complete", validation="internal")
internal.validation.agnes.complete.3 <- clValid(data.features, nClust=3, clMethods="agnes", method="complete", validation="internal")
internal.validation.agnes.complete.4 <- clValid(data.features, nClust=4, clMethods="agnes", method="complete", validation="internal")
internal.validation.agnes.complete.7 <- clValid(data.features, nClust=7, clMethods="agnes", method="complete", validation="internal")

summary(internal.validation.agnes.complete.2)
optimalScores(internal.validation.agnes.complete)


### Stability indices (APN, AD, ADM)

stability.validation.agnes.complete <- clValid(data.features, nClust=K.range, clMethods="agnes",method = "complete", validation="stability")
summary(stability.validation.agnes.complete)
optimalScores(stability.validation.agnes.complete)

# SINGLE LINKAGE

sil.agnes.single.2 <- silhouette(agnes.single.k2, dist(data.features))
sil.agnes.single.3 <- silhouette(agnes.single.k3, dist(data.features))
sil.agnes.single.4 <- silhouette(agnes.single.k4, dist(data.features))
sil.agnes.single.7 <- silhouette(agnes.single.k7, dist(data.features))

png("agnes_single_silh_2.png", width= 2048, height = 1536, res = 300)
fviz_silhouette(sil.agnes.single.2, xlab="AGNES (single)")
dev.off()

png("agnes_single_silh_3.png", width= 2048, height = 1536, res = 300)
fviz_silhouette(sil.agnes.single.3, xlab="AGNES (single)")
dev.off()

png("agnes_single_silh_4.png", width= 2048, height = 1536, res = 300)
fviz_silhouette(sil.agnes.single.4, xlab="AGNES (single)")
dev.off()

png("agnes_single_silh_7.png", width= 2048, height = 1536, res = 300)
fviz_silhouette(sil.agnes.single.7, xlab="AGNES (single)")
dev.off()

# Cluster labels vs. actual class labels

tab.agnes.single.2 <- table(agnes.single.k2, data.labels)
tab.agnes.single.3 <- table(agnes.single.k3, data.labels)
tab.agnes.single.4 <- table(agnes.single.k4, data.labels)
tab.agnes.single.7 <- table(agnes.single.k7, data.labels)

# partition agreement
matchClasses(tab.agnes.single.2)
compareMatchedClasses(agnes.single.k2, data.labels)$diag
matchClasses(tab.agnes.single.3)
compareMatchedClasses(agnes.single.k3, data.labels)$diag
matchClasses(tab.agnes.single.4)
compareMatchedClasses(agnes.single.k4, data.labels)$diag
matchClasses(tab.agnes.single.7)
compareMatchedClasses(agnes.single.k7, data.labels)$diag

# Internal validation

internal.validation.agnes.single.2 <- clValid(data.features, nClust=2, clMethods="agnes", method="single", validation="internal")
internal.validation.agnes.single.3 <- clValid(data.features, nClust=3, clMethods="agnes", method="single", validation="internal")
internal.validation.agnes.single.4 <- clValid(data.features, nClust=4, clMethods="agnes", method="single", validation="internal")
internal.validation.agnes.single.7 <- clValid(data.features, nClust=7, clMethods="agnes", method="single", validation="internal")

summary(internal.validation.agnes.single.2)
optimalScores(internal.validation.agnes.single)


### Stability indices (APN, AD, ADM)

stability.validation.agnes.single <- clValid(data.features, nClust=K.range, clMethods="agnes", method = "single", validation="stability")
summary(stability.validation.agnes.single)
optimalScores(stability.validation.agnes.single)


desc.agnes.single2 <- cluster.Description(data.features, agnes.single.k2)

df.agnes.single.1 <- desc.agnes.single2[1,,-5]
colnames(df.agnes.single.1) <- funcs
rownames(df.agnes.single.1) <- my_labels_new[]
df.agnes.single.1
#data.frame.fcm2.1 <- data.frame(df.k7.1)

df.agnes.single.2 <- desc.agnes.single2[2,,-5]
colnames(df.agnes.single.2) <- funcs
rownames(df.agnes.single.2) <- my_labels_new[]
df.agnes.single.2


desc.agnes.complete2 <- cluster.Description(data.features, agnes.complete.k2)

df.agnes.complete.1 <- desc.agnes.complete2[1,,-5]
colnames(df.agnes.complete.1) <- funcs
rownames(df.agnes.complete.1) <- my_labels_new[]
df.agnes.complete.1
#data.frame.fcm2.1 <- data.frame(df.k7.1)

df.agnes.complete.2 <- desc.agnes.complete2[2,,-5]
colnames(df.agnes.complete.2) <- funcs
rownames(df.agnes.complete.2) <- my_labels_new[]
df.agnes.complete.2

########################################

# DIANA

#######################################

diana <- diana(x=DissimilarityMatrix.mat, diss=TRUE)

diana <- diana(data.features)

png("diana_basic.png", width= 2048, height = 1536, res = 300)
plot(diana,which.plot=2, main="DIANA", cex=0.2)
dev.off()

png("diana_k2.png", width= 2048, height = 1536, res = 300)
fviz_dend(diana, k=2, cex=0.2)
dev.off()

png("diana_k3.png", width= 2048, height = 1536, res = 300)
fviz_dend(diana, k=3, cex=0.2)
dev.off()

png("diana_k4.png", width= 2048, height = 1536, res = 300)
fviz_dend(diana, k=4, cex=0.2)
dev.off()

png("diana_k7.png", width= 2048, height = 1536, res = 300)
fviz_dend(diana, k=7, cex=0.2)
dev.off()

# We cut off the dendrogram so as to get exactly K = 3 clusters
(diana.k2 <- cutree(diana, k=2))
(diana.k3 <- cutree(diana, k=3))
(diana.k4 <- cutree(diana, k=4))
(diana.k7 <- cutree(diana, k=7))

table(diana.k2)
table(diana.k3)
table(diana.k4)
table(diana.k7)


# we order the colors according to the order of the objects in the dendrogram
colors.objects.diana <- labels.colors[diana$order]

# dendrogram + colors (actual classes)
dendrogram.diana <- as.dendrogram(diana)
fviz_dend(dendrogram.diana, cex=0.3, label_cols=colors.objects.diana, main="Colors = actual classes")

# Comparison: partition into 3 clusters vs. actual classes
png("diana_k2_actual.png", width= 2048, height = 1536, res = 300)
fviz_dend(dendrogram.diana, cex=0.3, k=2,  label_cols=colors.objects.diana, k_colors=c("grey","orange"),main="Partition into 2 clusters vs. actual class labels", rect = T, lower_rect=-0.5)
dev.off()

png("diana_k3_actual.png", width= 2048, height = 1536, res = 300)
fviz_dend(dendrogram.diana, cex=0.3, k=3,  label_cols=colors.objects.diana, k_colors=c("grey","orange", "blue"),main="Partition into 3 clusters vs. actual class labels", rect = T, lower_rect=-0.5)
dev.off()

png("diana_k4_actual.png", width= 2048, height = 1536, res = 300)
fviz_dend(dendrogram.diana, cex=0.3, k=4,  label_cols=colors.objects.diana, k_colors=c("grey","orange", "blue", "darkorchid"),main="Partition into 4 clusters vs. actual class labels", rect = T, lower_rect=-0.5)
dev.off()

png("diana_k7_actual.png", width= 2048, height = 1536, res = 300)
fviz_dend(dendrogram.diana, cex=0.3, k=7,  label_cols=colors.objects.diana, k_colors=c("grey","orange", "blue", "darkorchid", "cyan", "darkolivegreen", "darksalmon"), main="Partition into 7 clusters vs. actual class labels", rect = T, lower_rect=-0.5)
dev.off()

## VALIDATION ##

sil.diana.2 <- silhouette(diana.k2, dist(data.features))
sil.diana.3 <- silhouette(diana.k3, dist(data.features))
sil.diana.4 <- silhouette(diana.k4, dist(data.features))
sil.diana.7 <- silhouette(diana.k7, dist(data.features))

png("diana_silh_2.png", width= 2048, height = 1536, res = 300)
fviz_silhouette(sil.diana.2, xlab="Diana")
dev.off()

png("diana_silh_3.png", width= 2048, height = 1536, res = 300)
fviz_silhouette(sil.diana.3, xlab="Diana")
dev.off()

png("diana_silh_4.png", width= 2048, height = 1536, res = 300)
fviz_silhouette(sil.diana.4, xlab="Diana")
dev.off()

png("diana_silh_7.png", width= 2048, height = 1536, res = 300)
fviz_silhouette(sil.diana.7, xlab="Diana")
dev.off()

# Cluster labels vs. actual class labels

tab.diana.2 <- table(diana.k2, data.labels)
tab.diana.3 <- table(diana.k3, data.labels)
tab.diana.4 <- table(diana.k4, data.labels)
tab.diana.7 <- table(diana.k7, data.labels)

# partition agreement
matchClasses(tab.diana.2)
compareMatchedClasses(diana.k2, data.labels)$diag
matchClasses(tab.diana.3)
compareMatchedClasses(diana.k3, data.labels)$diag
matchClasses(tab.diana.4)
compareMatchedClasses(diana.k4, data.labels)$diag
matchClasses(tab.diana.7)
compareMatchedClasses(diana.k7, data.labels)$diag

# Internal validation

internal.validation.diana.2 <- clValid(data.features, nClust=2, clMethods="diana", validation="internal")
internal.validation.diana.3 <- clValid(data.features, nClust=3, clMethods="diana", validation="internal")
internal.validation.diana.4 <- clValid(data.features, nClust=4, clMethods="diana", validation="internal")
internal.validation.diana.7 <- clValid(data.features, nClust=7, clMethods="diana", validation="internal")

summary(internal.validation.diana.2)
optimalScores(internal.validation.diana)

### Stability indices (APN, AD, ADM)

stability.validation.diana <- clValid(data.features, nClust=K.range, clMethods="diana", validation="stability")
summary(stability.validation.diana)
optimalScores(stability.validation.diana)

## STATISTISC OF CLUSTERS ##

funcs <- c("mean", "sd", "median", "mad")
my_labels_new <- my_labels[c(-1,-2)]

desc.diana.2 <- cluster.Description(data.features, diana.k2)

df.diana.2.1 <- desc.diana.2[1,,-5]
colnames(df.diana.2.1) <- funcs
rownames(df.diana.2.1) <- my_labels_new[]
df.diana.2.1

data.frame.pam.2.1 <- data.frame(df.pam.2.1)

df.diana.2.2 <- desc.diana.2[2,,-5]
colnames(df.diana.2.2) <- funcs
rownames(df.diana.2.2) <- my_labels_new[]
df.diana.2.2
data.frame.pam.2.2 <- data.frame(df.pam.2.2)

df.pam.3.3 <- desc.pam.3[3,,-5]
colnames(df.pam.3.3) <- funcs
rownames(df.pam.3.3) <- my_labels_new[]
df.pam.3.3
data.frame.pam.3.3 <- data.frame(df.pam.3.3)

df.pam.4.4 <- desc.pam.4[4,,-5]
colnames(df.pam.4.4) <- funcs
rownames(df.pam.4.4) <- my_labels_new[]
df.pam.4.4
data.frame.pam.4.4 <- data.frame(df.pam.4.4)

df.pam.7.5 <- desc.pam.7[5,,-5]
colnames(df.pam.7.5) <- funcs
rownames(df.pam.7.5) <- my_labels_new[]
df.pam.7.5
data.frame.pam.7.5 <- data.frame(df.pam.7.5)

df.pam.7.6 <- desc.pam.7[6,,-5]
colnames(df.pam.7.6) <- funcs
rownames(df.pam.7.6) <- my_labels_new[]
df.pam.7.6
data.frame.pam.7.6 <- data.frame(df.pam.7.6)

df.pam.7.7 <- desc.pam.7[2,,-5]
colnames(df.pam.7.7) <- funcs
rownames(df.pam.7.7) <- my_labels_new[]
df.pam.7.7
data.frame.pam.7.7 <- data.frame(df.pam.7.7)



#######################################

# VALIDATION

#######################################

# Dispersion within-clusters and  dispersion between-clusters for a different number of clusters (K)
Within   <- c()
Between  <- c()
Total    <- c()

K.range <- 1:10

for (k in K.range)
{
  print(k)
  kmeans.k  <- kmeans(data.features, centers=k, iter.max=10, nstart=10)
  Within  <- c(Within, kmeans.k$tot.withinss)	# total within-cluster sum of squares =  sum(withinss))
  Between <- c(Between, kmeans.k$betweenss) # between-cluster sum of squares
  Total   <- c(Total,kmeans.k$totss) 	# total sum of squares.
  # remark: Total == Within + Between
}

y.range <- range(c(Within, Between, Total))

plot(K.range,  Within, col="red", type="b", lwd=2, xlab="K", ylim=y.range, ylab="B/W")
lines(K.range,  Between, col="blue", lwd=2, type="b")
lines(K.range,  Total, col="black", lwd=2, type="b")
legend(x='right', legend=c("Total SS (total dispersion)", "Between SS (between-cluster dispersion)","Within SS (within-cluster dispersion)"), lwd=2, col=c("black","blue","red"), bg="azure2", cex=0.7)
grid()
title("Comparison of the within-cluster and between-cluster dispersion")

# Visualize k-means clustering
km.res <- kmeans(iris.features, 3, nstart=10)
sil.kmeans <- silhouette(km.res$cluster, dist(iris.features))
fviz_silhouette(sil.kmeans, xlab="K-means")

# Visualize AGNES (hierarchical clustering)
agnes.res <- agnes(iris.features, method="complete")
agnes.partition <- cutree(agnes.res, k=3)
sil.agnes <- silhouette(agnes.partition, dist(iris.features))
fviz_silhouette(sil.agnes, xlab="AGNES")

# Cluster labels vs. actual class labels

kmeans.k3 <- kmeans(iris.features, centers=3)
labels.kmeans <- kmeans.k3$cluster
labels.real <- iris$Species

(tab <- table(labels.kmeans, labels.real))

# partition agreement
matchClasses(tab)
compareMatchedClasses(labels.kmeans, labels.real)$diag

# We force each cluster to be assigned to a different class
matchClasses(tab, method="exact")
compareMatchedClasses(labels.kmeans, labels.real, method="exact")$diag


library(clValid)
library(mclust)

# clustering algorithms
methods <- c("hierarchical","kmeans", "diana", "fanny", "pam", "clara","model")

# range for number of clusters
K.range <- 2:6

internal.validation <- clValid(data.features, nClust=K.range, clMethods=methods, validation="internal")

summary(internal.validation)
optimalScores(internal.validation)

par(mfrow = c(2, 2))
plot(internal.validation, legend = FALSE, lwd=2)
plot.new()
legend("center", clusterMethods(internal.validation), col=1:9, lty=1:9, pch=paste(1:9))

###########################################################################################
### Stability indices (APN, AD, ADM)

stability.validation <- clValid(data.features, nClust=K.range, clMethods=methods, validation="stability")
summary(stability.validation)
optimalScores(stability.validation)

par(mfrow = c(2,2))
plot(stability.validation, measure=c("APN","AD","ADM"), legend=FALSE, lwd=2)
plot.new()
legend("center", clusterMethods(stability.validation), col=1:9, lty=1:9, pch=paste(1:9))
par(mfrow = c(1,1))
