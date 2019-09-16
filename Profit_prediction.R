# libraries --------------

install.packages("corrplot")
install.packages("Hmisc")
install.packages("PerformanceAnalytics")
install.packages("ggiraphExtra")
install.packages("BBmisc")

pacman::p_load(readr, caret, ggplot2, reshape2, DataExplorer, MASS, tidyr, mlbench)

# Reading data -------------
existing_data <- read.csv("/Drive/Ubiqum/Course_2_Data_Analytics_II/Task_3/existingproductattributes2017.csv")
new_data <- read.csv("/Drive/Ubiqum/Course_2_Data_Analytics_II/Task_3/newproductattributes2017.csv")
summary(existing_data)
str(existing_data)
names(existing_data)
is.na(existing_data)
sum(is.na(existing_data))
set.seed(12345)


# remove features and change type -------------
existing_data1 <- existing_data[-c(2,12,17)]
summary(existing_data1)
new_data1 <- new_data[-c(2,12,17)]
summary(new_data)


# dummify the data

existing_data2 <- dummyVars(" ~ .", data = existing_data1)
readyData <- data.frame(predict(existing_data2, newdata = existing_data1))
str(readyData)
summary(readyData)

new_data2 <- dummyVars(" ~ .", data = new_data1)
new_data3 <- data.frame(predict(new_data2, newdata = new_data1))
str(new_data3)
summary(new_data3)

# correlation matrix --------

corrData <- cor(readyData)
corrData
corrplot(corrData, type = "upper", order = "hclust", 
         tl.col = "black", tl.srt = 45)
res2 <- rcorr(as.matrix(readyData))

corrplot(res2$r, type="upper", order="hclust", 
         p.mat = res2$P, sig.level = 0.01, insig = "blank")

create_report(readyData)
create_report(readyData1)

# remove features that correlated
readyData1 <- readyData[-c(14,16,18)]

res3 <- rcorr(as.matrix(readyData1))

corrplot(res3$r, type="upper", order="hclust", 
         p.mat = res3$P, sig.level = 0.01, insig = "blank")

new_data4 <- new_data3[-c(14,16,18)]



# data exploration ---------

chart.Correlation(readyData1, histogram=TRUE, pch=19)

par(mfrow=c(1,1))
for (i in 1:length(readyData1)) {
  boxplot(readyData1[,i], main=names(readyData1[i]), type="l")
  
}

### removing outliers
outliers <- boxplot(readyData1$Volume, plot=FALSE)$out
print(outliers)
readyData1[which(readyData1$Volume %in% outliers),]

readyData2 <- readyData1[-which(readyData1$Volume %in% outliers),]

for (i in 1:length(readyData2)) {
  boxplot(readyData2[,i], main=names(readyData2[i]), type="l")
  
}

# Split the data in 75%/25% --------
index <- createDataPartition(readyData2$Volume,
                             p = 0.75, 
                             list = F)
TrainSet <- readyData2[index,]
TestSet <- readyData2[-index,]
nrow(TrainSet)
nrow(TestSet)

# Parallel working -------------
library(doParallel)
cl <- makePSOCKcluster(6)
registerDoParallel(cl)
stopCluster(cl)

# Regression Models ---------
# Linear model 
lmFit <- lm(Volume~ ., TrainSet)
summary(lmFit)
lmPredictions <- predict(lmFit, TestSet)
lmPredictions
postResample(lmPredictions, TestSet$Volume)


ctrl <- trainControl(method = "repeatedcv", repeats = 1)

# SVM
svmFit <- train(Volume ~ ., 
                data = TrainSet, 
                method = "svmLinear2",
                trControl = ctrl, 
                preProcess = c("center", "scale"),
                tuneLength = 20)
svmFit
svmPredictions <- predict(svmFit, newdata = TestSet)
summary(svmPredictions)
postResample(svmPredictions, TestSet$Volume)

# Random Forest

rfFit <- train(Volume ~ ., 
               data = TrainSet, 
               method = "rf", 
               trControl = ctrl, 
               preProcess = c("center", "scale"),
               tuneLength = 20)
rfFit
rfPredictions <- predict(rfFit, newdata = TestSet)
summary(rfPredictions)
postResample(rfPredictions, TestSet$Volume)

# kNN
knnFit <- train(Volume ~ ., 
                data = TrainSet, 
                method = "knn", 
                trControl = ctrl, 
                preProcess = c("center", "scale"),
                tuneLength = 20)
knnFit
knnPredictions <- predict(knnFit, newdata = TestSet)
summary(knnPredictions)
postResample(knnPredictions, TestSet$Volume)

# compare models -------------
resamps <- resamples(list(SVM = svmFit, rf = rfFit, KNN = knnFit))
summary(resamps)
xyplot(resamps, what = "BlandAltman")


# predicting brand in the incomplete data ------------
rf_newdata <- predict(rfFit, newdata = new_data4)
rf_newdata
summary(rf_newdata)

output <- new_data 
output$predictions <- rf_newdata
write.csv(output, file="C2.T3output.csv", row.names = TRUE)

# plotting --------------
plot(rfFit)
plot(rfPredictions, type="l")
points(TestSet$Volume)     
 

results <- read.csv("/Drive/Ubiqum/Course_2_Data_Analytics_II/Task_3/results.csv")
summary(results)
str(results)

plot(results$Profit~results$ProductType)

plot1 <- ggplot(results, aes(x=ProductType, y=Profit, color= ProductType)) +
   geom_point(size =4)+
  xlab("Product Type") + ylab("Profit") +
  ggtitle("Predicted profit by product type") +
  labs(fill="Product Type")

plot2 <- plot1 + plot.settings

plot(existing_data$Volume~existing_data$x4StarReviews)

# plotting custumer reviews ----------------

data_for_melt <- data.frame(existing_data$Volume,existing_data$x5StarReviews,
                            existing_data$x4StarReviews,existing_data$x3StarReviews,
                            existing_data$x2StarReviews,existing_data$x1StarReviews, existing_data$ProductType)

names(data_for_melt) <- c("Volume", "x5", "x4", "x3", "x2", "x1", "ProductType")

### removing outliers
outliers.melt <- boxplot(data_for_melt$Volume, plot=FALSE)$out
print(outliers.melt)
data_for_melt[which(data_for_melt$Volume %in% outliers),]

data_for_melt1 <- data_for_melt[-which(data_for_melt$Volume %in% outliers),]

melt.data <- melt(data_for_melt1, id = c("Volume", "ProductType"))

melt.data <- normalize(melt.data, method = "scale")

plot3 <- ggplot(melt.data, aes(x=Volume, y=value, color=variable)) +
  geom_point() +
  geom_smooth(method=lm)+
  xlab("Volume") + ylab("Number of reviews") +
  ggtitle("Relationship between Volume and Costumer reviews") +
  labs(color="Costumer Reviews")

plot4 <- plot3 + plot.settings

plot5 <- ggplot(subset(melt.data, ProductType %in% c("Laptop","Netbook","PC","Smartphone")), aes(x=Volume, y=value, color=variable)) +
  geom_point() +
  geom_smooth(method=lm)+
  xlab("Volume") + ylab("Number of reviews") +
  ggtitle("Relationship between Volume and Costumer reviews") +
  labs(color="Costumer Reviews") + facet_wrap(vars(ProductType), scales = "free")

plot6 <- plot5 + plot.settings

# plotting positive and negative reviews ------------

data_for_meltP <- data.frame(existing_data$Volume,
                             existing_data$PositiveServiceReview,
                             existing_data$NegativeServiceReview,
                             existing_data$ProductType)

names(data_for_meltP) <- c("Volume", "Positive", "Negative","ProductType")

### removing outliers
outliers.meltP <- boxplot(data_for_meltP$Volume, plot=FALSE)$out
print(outliers.meltP)

data_for_meltP1 <- data_for_meltP[-which(data_for_meltP$Volume %in% outliers.meltP),]

melt.dataP <- melt(data_for_meltP1, id = c("Volume", "ProductType"))

melt.dataP <- normalize(melt.dataP, method = "scale")

plotP3 <- ggplot(melt.dataP, aes(x=Volume, y=value, color=variable)) +
  geom_point() +
  geom_smooth(method=lm)+
  xlab("Volume") + ylab("Type of reviews") +
  ggtitle("Relationship between Volume and Type of reviews") +
  labs(color="Type of Reviews")

plotP4 <- plotP3 + plot.settings

plotP5 <- ggplot(subset(melt.dataP, ProductType %in% c("Laptop","Netbook","PC","Smartphone")), aes(x=Volume, y=value, color=variable)) +
  geom_point() +
  geom_smooth(method=lm)+
  xlab("Volume") + ylab("Type of reviews") +
  ggtitle("Relationship between Volume and Type of reviews") +
  labs(color="Type of Reviews") + facet_wrap(vars(ProductType), scales = "free")

plotP6 <- plotP5 + plot.settings

# General plot settings ----------------
plot.settings <- theme(
  axis.line.x =       element_line(colour = "black", size = 1),                                                       # Settings x-axis line
  axis.line.y =       element_line(colour = "black", size = 1),                                                       # Settings y-axis line 
  axis.text.x =       element_text(colour = "black", size = 16, lineheight = 0.9, vjust = 1, face = "bold"),        # Font x-axis 
  axis.text.y =       element_text(colour = "black", size = 16, lineheight = 0.9, hjust = 1),                         # Font y-axis
  axis.ticks =        element_line(colour = "black", size = 0.3),                                                     # Color/thickness axis ticks
  axis.title.x =      element_text(size = 20, vjust = 1, face = "bold", margin = margin(10,1,1,1)),                   # Font x-axis title
  axis.title.y =      element_text(size = 20, angle = 90, vjust = 1, face = "bold", margin = margin(1,10,1,1)),       # Font y-axis title
  
  legend.background = element_rect(colour=NA),                                                                        # Background color legend
  legend.key =        element_blank(),                                                                                # Background color legend key
  legend.key.size =   unit(1.2, "lines"),                                                                             # Size legend key
  legend.text =       element_text(size = 18),                                                                        # Font legend text
  legend.title =      element_text(size = 20, face = "bold", hjust = 0),                                              # Font legend title  
  legend.position =   "right",                                                                                        # Legend position
  
  panel.background =  element_blank(),                                                                                # Background color graph
  panel.border =      element_blank(),                                                                                # Border around graph (use element_rect())
  panel.grid.major =  element_blank(),                                                                                # Major gridlines (use element_line())
  panel.grid.minor =  element_blank(),                                                                                # Minor gridlines (use element_line())
  panel.margin =      unit(1, "lines"),                                                                               # Panel margins
  
  strip.background =  element_rect(fill = "grey80", colour = "grey50"),                                               # Background colour strip 
  strip.text.x =      element_text(size = 20),                                                                        # Font strip text x-axis
  strip.text.y =      element_text(size = 20, angle = -90),                                                           # Font strip text y-axis
  
  plot.background =   element_rect(colour = NA),                                                                      # Background color of entire plot
  plot.title =        element_text(size = 20, face = "bold", hjust = 0.5),                                                                        # Font plot title 
  plot.margin =       unit(c(1, 1, 1, 1), "lines")                                                                    # Plot margins
)
