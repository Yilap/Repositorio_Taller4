################################################################
# Problem Set 4: Script
# Authors: Yilmer Palacios, Betina Cortés, Lida Jimena Cárdenas,
# Nelson Fabián López
################################################################

# Loading Libraries ----
rm(list = ls()) 

# install.packages("pacman")
library("pacman")

p_load("tm",
       "stringi",
       "proxy",
       "tidyverse",
       "sf",
       "naniar",
       "tidymodels",
       "readxl",
       "psych",
       "ranger",
       "glmnet",
       "naniar",
       "tidyverse",
       "caret",
       "glmnet",
       "ggplot2",
       "ggraph",
       "gt")

#setwd("/Users/betinacortes/Desktop/Repositorio_taller3")
setwd("C:/Users/Yilmer Palacios/Desktop/Repositorios GitHub/Repositorio_Taller4")


# Importing Dataset ----
# Removing City and operation type as they don't add information 
train <- read.csv("stores/Data_Kaggle/train.csv",header = TRUE,sep=",")
test <- read.csv("stores/Data_Kaggle/test.csv",header = TRUE,sep=",")

text <- stri_trans_general(str = train$text, id = "Latin-ASCII")


texts<-VCorpus(VectorSource(train$text))
texts

rm("dtm_idf") 

#limpiamos el corpus texts
texts <- tm_map(texts, removePunctuation)
texts <- tm_map(texts, content_transformer(tolower))
texts <- tm_map(texts, removeWords, stopwords("spanish"))
texts <- tm_map(texts,content_transformer(removeNumbers))
texts<-tm_map(texts,content_transformer(stripWhitespace))
texts <- tm_map(texts, content_transformer(gsub), pattern = "[!|?|¡|¿|“|”|´|´|‘|’]", replacement = "")
texts <- tm_map(texts, content_transformer(gsub), pattern = "\\b[0-9]+\\b", replacement = "")
texts <- tm_map(texts, content_transformer(gsub), pattern = "\"", replacement = "")
texts <- tm_map(texts, content_transformer(function(x) stri_replace_all_regex(x, "\\p{Emoji}", "")))


dtm_idf<-DocumentTermMatrix(texts,
                            control=list(weighting=weightTfIdf,
                            stopwords=c("con", "si")))

dtm_idf <- removeSparseTerms(dtm_idf, sparse = 0.95)
inspect(dtm_idf[100:103,])

dtm_idf
inspect(dtm_idf)

#revisamos frecuencias
terms_freq <- findFreqTerms(dtm_idf, lowfreq = 1)
freq_table <- data.frame(term = terms_freq, freq = colSums(as.matrix(dtm_idf[,terms_freq])))
print(freq_table)




