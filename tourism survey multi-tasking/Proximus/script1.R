library(xlsx) 
library(readxl)
library("cba")
df <- read_xlsx("data_binary.xlsx", col_names = FALSE)
data <- as.matrix(df)
data <- apply(data,2,as.logical)
data <- t(data)

pr1 <- proximus(data, max.radius=8, debug=TRUE)
summary(pr1)

pr2<- proximus(data, max.radius=3, debug=TRUE)
summary(pr2)

# 4-22

# Proximus for for data excluding 1 or 8, no transpose, no merge
df <- read_xlsx("data_binary.xlsx", sheet = 'Kill18', col_names = FALSE)
data <- as.matrix(df)
# no transpose, thus to categorize people
pr3 <- proximus(data, max.radius=8, debug=TRUE)
summary(pr3)

# Proximus for data excluding 1 or 8, no merge
df <- read_xlsx("data_binary.xlsx", sheet = 'Kill18', col_names = FALSE)
data <- as.matrix(df)
data <- t(data)  # transpose, so that each column is a 'Document'
pr3 <- proximus(data, max.radius=8, debug=TRUE)
summary(pr3)
print('4 purposes are clustered into the 1st category, while the remaining are seen unique.')

# Proximus for data excluding 1 or 8, with merge
df <- read_xlsx("data_binary.xlsx", sheet = 'Kill18Mer', col_names = FALSE)
data <- as.matrix(df)
data <- t(data)  # transpose, so that each column is a 'Document'
pr4 <- proximus(data, max.radius=8, debug=TRUE)
summary(pr4)
print('3 purposes are clustered into the 1st category, which are: ')
print(pr4[["a"]][[1]][["x"]])

# 4-23

# Proximus for data excluding 1 or 8, no merge
df <- read_xlsx("data_binary.xlsx", sheet = 'Zeros18', col_names = FALSE)
data <- as.matrix(df)
data <- t(data)  # transpose, so that each column is a 'Document'
pr3 <- proximus(data, max.radius=8, debug=TRUE)
summary(pr3)

# Proximus for for data excluding 1 or 8, no transpose, no merge
df <- read_xlsx("data_binary.xlsx", sheet = 'Zeros18', col_names = FALSE)
data <- as.matrix(df)
# no transpose, thus to categorize people
pr3 <- proximus(data, max.radius=8, debug=TRUE)
summary(pr3)

# Proximus for data excluding 1 or 8, with merge
df <- read_xlsx("data_binary.xlsx", sheet = 'Zeros16', col_names = FALSE)
data <- as.matrix(df)
data <- t(data)  # transpose, so that each column is a 'Document'
pr4 <- proximus(data, max.radius=8, debug=TRUE)
summary(pr4)

# Proximus for Des_choice
df <- read_xlsx("data_binary.xlsx", sheet = 'Des_choice', col_names = FALSE)
data <- as.matrix(df)
# data <- t(data)  # transpose, so that each column is a 'Document'
data <- apply(data,2,as.logical)
pr <- proximus(data, max.radius=8, debug=TRUE)
summary(pr)
