# Predicting the Occurrence of Cancer

## Step 1 - Collecting the Data

# http://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29


# Breast cancer data includes 569 observations of cancer biopsies,
# each with 32 characteristics (variables). A characteristic is a number of
# identification (ID), another is the cancer diagnosis, and 30 are laboratory measurements
# numeric. The diagnosis is coded as "M" to indicate malignant or "B" for
# indicate benign.
dados <- read.csv("bc_data.csv", stringsAsFactors = FALSE)
str(dados)
head(dados)


## Step 2 - Exploring the Data

# Deleting the ID column
# Regardless of the machine learning method, it should always be excluded
# ID variables. Otherwise, this can lead to erroneous results because the ID
# can be used to "predict" each example only. Therefore, a model
# that includes an identifier may suffer from overfitting, and it will be very difficult to use it for
# generalize other data.
dados <- dados[-1]
str(dados)
any(is.na(dados))

# Many classifiers require variables to be of the Factor type
table(dados$diagnosis)
dados$diagnosis <- factor(dados$diagnosis, levels = c("B", "M"), labels = c("Benigno", "Maligno"))
str(dados$diagnosis)

# Checking the aspect ratio
round(prop.table(table(dados$diagnosis)) * 100, digits = 1) 

# Cetral Trend Measures
# We detected here a scale problem between the data, which then needs to be normalized
# The distance calculation made by kNN is dependent on the scale measures in the input data.
summary(dados[c("radius_mean", "area_mean", "smoothness_mean")])

# Creating a normalization function
normalizar <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}

# Testing the normalization function - the results should be identical
normalizar(c(1, 2, 3, 4, 5))
normalizar(c(10, 20, 30, 40, 50))

# Normalizing the data
dados_norm <- as.data.frame(lapply(dados[2:31], normalizar))

# Confirming that normalization worked
summary(dados[c("radius_mean", "area_mean", "smoothness_mean")])
summary(dados_norm[c("radius_mean", "area_mean", "smoothness_mean")])


## Step 3: Training the model

# Loading the library package
# install.packages("class")
library(class)
?knn

# Creating training data and test data
dados_treino <- dados_norm[1:469, ]
dados_teste <- dados_norm[470:569, ]

# Creating labels for training and test data
dados_treino_labels <- dados[1:469, 1]
dados_teste_labels <- dados[470:569, 1]
length(dados_treino_labels)
length(dados_teste_labels)

# Creating the model
modelo <- knn(train = dados_treino, 
              test = dados_teste,
              cl = dados_treino_labels, 
              k = 21)

# The knn () function returns a factor object with the predictions for each example in the test dataset
class(modelo)


## Step 4: Evaluating and Interpreting the Model

# Loading gmodels
install.packages("gmodels")
library(gmodels)

# Creating a crosstab of predicted data vs. current data
# We will use a sample with 100 observations: length (data_labels_test)
CrossTable(x = dados_teste_labels, y = modelo, prop.chisq = FALSE)

# Interpreting the Results
# The crosstab shows 4 possible values, which represent the false / true positive and negative
# The first column lists the original labels in the observed data
# The two columns of the model (Benign and Malignant) of the model, show the forecast results
# We have:
# Scenario 1: Benign Cell (label) x Benign (Model) - 61 cases - true negative
# Scenario 2: Benign Cell (label) x Malignant (Model) - 00 cases - false positive
# Scenario 3: Malignant Cell (label) x Benign (Model) - 02 cases - false negative (the model was wrong)
# Scenario 4: Malignant cell (label) x Malignant (Model) - 37 cases - true positive

# Reading the Confusion Matrix (Perspective of having the disease or not):

# True Negative = our model predicted that the person did NOT have the disease and the data showed that the person did NOT actually have the disease
# False Positive = our model predicted that the person had the disease and the data showed that NO, the person had the disease
# False Negative = our model predicted that the person did NOT have the disease and the data showed that YES, the person had the disease
# True Positive = our model predicted that the person had the disease and the data showed that YES, the person had the disease

# False Positive - Type I Error
# False Negative - Type II Error

# Model hit rate: 98% (hit 98 out of 100)

# Consult the definition of confusion matrix in case of doubt !!!


## Step 5: Optimizing the model's performance

# Using the scale () function to standardize the z-score
?scale()
dados_z <- as.data.frame(scale(dados[-1]))

# Confirming successful transformation
summary(dados_z$area_mean)

# Creating new training and test datasets
dados_treino <- dados_z[1:469, ]
dados_teste <- dados_z[470:569, ]

dados_treino_labels <- dados[ 1: 469, 1] 
dados_teste_labels <- dados[ 470: 569, 1]

# Reclassifying
modelo_v2 <- knn(train = dados_treino, 
                 test = dados_teste,
                 cl = dados_treino_labels, 
                 k = 21)

# Creating a crosstab of predicted data vs. current data
CrossTable(x = dados_teste_labels, y = modelo_v2, prop.chisq = FALSE)

# Testing different values for k
# Creating training data and test data
dados_treino <- dados_norm[1:469, ]
dados_teste <- dados_norm[470:569, ]

# Creating labels for training and test data
dados_treino_labels <- dados[1:469, 1]
dados_teste_labels <- dados[470:569, 1]

# Different values for k
modelo_v3 <- knn(train = dados_treino, 
                 test = dados_teste, 
                 cl = dados_treino_labels, 
                 k = 1)
CrossTable(x = dados_teste_labels, y = modelo_v3, prop.chisq = FALSE)

modelo_v4 <- knn(train = dados_treino, 
                 test = dados_teste, 
                 cl = dados_treino_labels, 
                 k = 5)
CrossTable(x = dados_teste_labels, y = modelo_v4, prop.chisq = FALSE)

modelo_v5 <- knn(train = dados_treino, 
                 test = dados_teste, 
                 cl = dados_treino_labels, 
                 k = 11)
CrossTable(x = dados_teste_labels, y = modelo_v5, prop.chisq=FALSE)

modelo_v6 <- knn(train = dados_treino, 
                 test = dados_teste, 
                 cl = dados_treino_labels, 
                 k = 15)
CrossTable(x = dados_teste_labels, y = modelo_v6, prop.chisq = FALSE)

modelo_v7 <- knn(train = dados_treino, 
                 test = dados_teste, 
                 cl = dados_treino_labels, 
                 k = 27)
CrossTable(x = dados_teste_labels, y = modelo_v7, prop.chisq = FALSE)

modelo_v2 <- knn(train = dados_treino, 
                 test = dados_teste,
                 cl = dados_treino_labels, 
                 k = 21)
CrossTable(x = dados_teste_labels, y = modelo_v2, prop.chisq = FALSE)


## Calculating the error rate
prev = NULL
taxa_erro = NULL

suppressWarnings(
  for(i in 1:20){
    set.seed(101)
    prev = knn(train = dados_treino, test = dados_teste, cl = dados_treino_labels, k = i)
    taxa_erro[i] = mean(dados$diagnosis != prev)
  })

# Getting k values and error rates
library(ggplot2)
k.values <- 1:20
df_erro <- data.frame(taxa_erro, k.values)
df_erro

# As we increase k, we decrease the model's error rate
ggplot(df_erro, aes(x = k.values, y = taxa_erro)) + geom_point()+ geom_line(lty = "dotted", color = 'red')






