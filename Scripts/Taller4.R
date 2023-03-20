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

setwd("/Users/betinacortes/Desktop/Repositorio_Taller4")
# setwd("C:/Users/Yilmer Palacios/Desktop/Repositorios GitHub/Repositorio_Taller4")


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

## Alternativa 2!!

# Loading Libraries ----
rm(list = ls()) 

# install.packages("pacman")
library("pacman")
p_load("tidyverse", "textir", "wordcloud", "tidytext", "tm",
       "SnowballC", "parsnip", "yardstick", "gt", "maptpx", "gamlr")

# Importing Dataset ----
# Loading Training and test dataset

train |> 
  mutate(longitud = str_length(text)) |> 
  arrange(longitud)

## Descriptive statistics

num_tweets <- train |> 
  group_by(name) |> 
  count() |> 
  ungroup() 

longitud_tweets <- train |> 
  group_by(name) |> 
  summarise(longitud = mean(str_length(text), na.rm = T))

contiene_hashtags <- train |> 
  group_by(name) |>
  mutate(tiene_hashtag = str_detect(text, "#")) |> 
  summarise(hashtag = mean(tiene_hashtag, na.rm = T))

tabla_resumen <- num_tweets |> 
  left_join(longitud_tweets) |> 
  left_join(contiene_hashtags) |> 
  gt() |> 
  fmt_number(
    columns = n:hashtag,
    decimals = 2
  ) |> 
  cols_label(
    name = "Autor",
    n = "No.",
    longitud = "Num. Caracteres",
    hashtag = "Pct. que contiene numerales"
  ) |> 
  tab_header(
    title = md("**Descripción de los tweets**"),
    subtitle = "Dataset de entrenamiento"
  ) |> 
  cols_width(
    name ~ px(70),
    n ~ px(100),
    longitud ~ px(120),
    hashtag ~ px(120)
  ) 

gtsave(tabla_resumen, "Tablas/tabla_resumen.html")

## Creating Wordcouds based on the train dataset.

bag_of_words <- train |> 
  unnest_tokens(output = "word", input = text, token = "words") |> 
  anti_join(tibble(word = stopwords(kind = "es"))) |> 
  count(name, word) |> 
  filter(str_length(word) >= 3)

bag_of_words_lopez <- bag_of_words |> filter(name == "Lopez")
bag_of_words_uribe <- bag_of_words |> filter(name == "Uribe")
bag_of_words_petro <- bag_of_words |> filter(name == "Petro")

crear_wordcloud <- function(dataset, vector_colors, seed = 42) {
  set.seed(seed)
  wordcloud(words = dataset$word,
            freq = dataset$n,
            min.freq = 50,
            scale = c(10, 0.5),
            max.words = 300,
            random.order = FALSE,
            rot.per = 0.35,
            colors = colorRampPalette(colors = vector_colors)(3))
}

jpeg("Wordclouds/wordcloud_lopez.jpeg", width = 800, height = 800)
crear_wordcloud(bag_of_words_lopez, c("#74c476", "#006d2c"))
dev.off()

jpeg("Wordclouds/wordcloud_uribe.jpeg", width = 800, height = 800)
crear_wordcloud(bag_of_words_uribe, c("#6baed6", "#08519c"))
dev.off()

jpeg("Wordclouds/wordcloud_petro.jpeg", width = 800, height = 800)
crear_wordcloud(bag_of_words_petro, c("#9e9ac8", "#54278f"))
dev.off()

## Tokenizing and lemmatizing the dataset

test_new <- test |> 
  bind_cols(name = as.character(NA)) |> 
  select(id, name, text)

train_test <- train |> 
  bind_rows(test_new)

train_test_tokens <- train_test |> 
  unnest_tokens(output = "word", input = text, token = "words", drop = F) |> 
  anti_join(tibble(word = stopwords(kind = "es"))) |> 
  filter(str_length(word) >= 3) |> 
  mutate(word = wordStem(word, language = "es"))

train_test_matrix <- train_test_tokens |>  
  count(id, word) |> 
  cast_dtm(document = id, term = word, value = n, weighting = tm::weightTfIdf)

train_test_matrix_less_sparse <- removeSparseTerms(train_test_matrix, sparse = .9997)

train_matrix_less_sparse <- train_test_matrix_less_sparse |> 
  tidy() |> 
  filter(document %in% train$id) |> 
  cast_dtm(document = document, term = term, value = count)

test_matrix_less_sparse <- train_test_matrix_less_sparse |> 
  tidy() |> 
  complete(document, term) |> 
  replace_na(list(count = 0)) |> 
  filter(document %in% test$id) |> 
  cast_dtm(document = document, term = term, value = count)

train_filtered <- train_test_tokens |> 
  distinct(id, name, text) |> 
  filter(id %in% train$id)

test_filtered <- train_test_tokens |> 
  distinct(id, name, text) |> 
  filter(id %in% test$id)

## Split into train_split and validation_split
set.seed(42)
sample_size <- floor(0.75 * nrow(train_matrix_less_sparse))
train_ind <- sample(nrow(train_matrix_less_sparse), size = sample_size)

train_split <- train_matrix_less_sparse[train_ind, ]
validation_split <- train_matrix_less_sparse[-train_ind, ]

## Models - Random forest
rf_classifier <- rand_forest() |> 
  set_engine("ranger") |> 
  set_mode("classification")

rf_fit <- rf_classifier |> 
  fit_xy(x = as.data.frame(as.matrix(train_split)),
         y = as.factor(train_filtered$name[train_ind]))

rf_validation_pred <- predict(rf_fit,
                              new_data = as.data.frame(as.matrix(validation_split))) |> 
  bind_cols(predict(rf_fit, as.data.frame(as.matrix(validation_split)), type= "prob"))

rf_validation_pred


