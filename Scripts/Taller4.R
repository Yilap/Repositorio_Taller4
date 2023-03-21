################################################################
# Problem Set 4: Script
# Authors: Yilmer Palacios, Betina Cortés, Lida Jimena Cárdenas,
# Nelson Fabián López
################################################################

# Loading Libraries ----
rm(list = ls()) 

# install.packages("pacman")
library("pacman")
p_load("tidyverse", "tidytext", "tm", "SnowballC", "topicmodels", 
       "wordcloud", "igraph", "ggraph", "Matrix", "xgboost", "caret",
       "syuzhet", "text2vec", "glmnet", "gt")



#setwd("/Users/betinacortes/Desktop/Repositorio_Taller4")
setwd("C:/Users/Yilmer Palacios/Desktop/Repositorios GitHub/Repositorio_Taller4")

#Cargamos bases de datos

train <- read_csv("Stores/Data_Kaggle/train.csv") |> 
  replace_na(list(text = ".")) |> 
  mutate(len = str_count(text))

test <- read_csv("Stores/Data_Kaggle/test.csv") |> 
  replace_na(list(text = ".")) |> 
  mutate(len = str_count(text))


## Tabla de estadisticas descriptivas

num_tweets <- train |> 
  group_by(name) |> 
  count() |> 
  ungroup() 

longitud_tweets <- train |> 
  group_by(name) |> 
  summarise(longitud = median(str_length(text), na.rm = T))

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

gtsave(tabla_resumen, "Views/tabla_resumen.html")


## Visualizamos número de caracteres para determinar tiempo de actividad en la Twitter
train |> 
  group_by(name) |> 
  summarise(CountMedian = median(len, na.rm = TRUE)) |> 
  ungroup() |> 
  mutate(author = reorder(name, CountMedian)) |> 
  ggplot(aes (x = author, y = CountMedian)) +
  geom_bar(stat = "identity") +
  geom_text(aes(x = name, y = CountMedian + 15, label = CountMedian)) +
  labs(
    title = "Mediana de la longitud de caracteres\nde los tweets por autor",
    x = "Autor", 
    y = "Mediana") +
  theme_minimal() +
  theme(
    plot.background = element_rect(fill = "white", color = "white")
  )

ggsave("Views/Numero_caracteres.png", width = 1500, height = 1500, units = "px")



## Densidad del número de caractéres por político ----

train |> 
  ggplot(aes(x = len)) +
  geom_histogram() +
  labs(
    title = "Distribución de longitud de tweets por autor",
    y = "Conteo", 
    x = "Número de caracteres") +
  facet_wrap(~name) +
  scale_x_continuous(limits = c(15, 280)) +
  theme_minimal() +
  theme(
    plot.background = element_rect(fill = "white", color = "white")
  )

ggsave("Views/Numero_caracteres_histograma.png", width = 1500, height = 1500, units = "px")


## Tokenización ----

train |> 
  unnest_tokens(output = "word", input = text, token = "words") |> 
  anti_join(tibble(word = stopwords(kind = "es")))



## Creación de nubes de palabras ----

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


jpeg("Views/wordcloud_lopez.jpeg", width = 800, height = 800)
crear_wordcloud(bag_of_words_lopez, c("#74c476", "#006d2c"))
dev.off()

jpeg("Views/wordcloud_uribe.jpeg", width = 800, height = 800)
crear_wordcloud(bag_of_words_uribe, c("#6baed6", "#08519c"))
dev.off()

jpeg("Views/wordcloud_petro.jpeg", width = 800, height = 800)
crear_wordcloud(bag_of_words_petro, c("#9e9ac8", "#54278f"))
dev.off()


### TF-IDF ----

## Hacemos conteo de la aparición de las palabras para cada político
## Cabe resaltar que aun no se ha aplicado el stopword
trainWords <- train |> 
  unnest_tokens(output = word, input = text) |> 
  count(name, word, sort = TRUE)


## Calculamos total de palabras por autor.
totalWords <- trainWords |> 
  group_by(name) |> 
  summarize(total = sum(n))

## Agregamos el numero total de palabras a la data original para construir TFIDF
trainWords <- trainWords |>  left_join(totalWords) |> 
  filter(!is.na(name)) |> 
  bind_tf_idf(word, name, n)

plot_trainWords <- trainWords |> 
  arrange(desc(tf_idf)) |> 
  mutate(word = factor(word, levels = rev(unique(word))))


## Graficamos palabras con mayor TFIDF
plot_trainWords |> 
  top_n(20) |> 
  ggplot(aes(word, tf_idf)) +
  geom_col() +
  labs(title = "Top 20 palabras por TF-IDF", x = NULL, y = "TF-IDF") +
  coord_flip() +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5),
    plot.title.position = "plot",
    plot.background = element_rect(fill = "white", color = "white")
  )

ggsave("Views/Top_20_TFIDF.png", width = 1500, height = 1500, units = "px")


## Palabras más importantes por político

create_TFIDF_author_plot <- function(author, fill) {
  plot_trainWords |> 
    filter(name == author) |> 
    top_n(20) |> 
    ggplot(aes(word, tf_idf)) +
    geom_col(fill = fill) +
    labs(
      title = paste("Top 20 palabras con mayor TF-IDF para", author),
      x = "Palabra",
      y = "TF-IDF") +
    coord_flip() +
    theme_minimal() +
    theme(
      plot.title = element_text(hjust = 0.5),
      plot.title.position = "plot",
      plot.background = element_rect(fill = "white", color = "white")
    )
}

create_TFIDF_author_plot("Lopez", "#006d2c")
ggsave("Views/Top_20_TFIDF_Lopez.png", width = 1500, height = 1500, units = "px")
create_TFIDF_author_plot("Petro", "#54278f")
ggsave("Views/Top_20_TFIDF_Petro.png", width = 1500, height = 1500, units = "px")
create_TFIDF_author_plot("Uribe", "#08519c")
ggsave("Views/Top_20_TFIDF_Uribe.png", width = 1500, height = 1500, units = "px")


#Nos parece interesante ver la combinación de palabras para los políticos, no lo agregamos al documento pero 
#va en Anexos

### Bigrams ----

create_bigram_author_plot <- function(author, fill) {
  train  |> 
    filter(name == author) |> 
    unnest_tokens(bigram, text, token = "ngrams", n = 2) |> 
    separate(bigram, c("word1", "word2"), sep = " ") |>
    filter(!word1 %in% stopwords(kind = "es"),
           !word2 %in% stopwords(kind = "es")) |>
    unite(bigramWord, word1, word2, sep = " ") |>
    group_by(bigramWord) |>
    tally() |>
    ungroup() |>
    arrange(desc(n)) |>
    mutate(bigramWord = reorder(bigramWord,n)) |>
    head(10) |>
    ggplot(aes(x = bigramWord,y = n)) +
    geom_bar(stat='identity', fill = fill) +
    geom_text(aes(x = bigramWord, y = 1, label = n),
              hjust=0, vjust=.5, size = 4, color = "white") +
    labs(x = 'Bigramas', 
         y = 'Conteo', 
         title = paste('Conteo de Bigramas más comunes para', author)) +
    coord_flip() + 
    theme_minimal() +
    theme(
      plot.title = element_text(hjust = 0.5),
      plot.title.position = "plot",
      plot.background = element_rect(fill = "white", color = "white")
    )
}

create_bigram_author_plot("Lopez", "#006d2c")
ggsave("Views/Top_10_bigramas_Lopez.png", width = 1500, height = 1500, units = "px")
create_bigram_author_plot("Petro", "#54278f")
ggsave("Views/Top_10_bigramas_Petro.png", width = 1500, height = 1500, units = "px")
create_bigram_author_plot("Uribe", "#08519c")
ggsave("Views/Top_10_bigramas_Uribe.png", width = 1500, height = 1500, units = "px")


### Trigrams ----

create_trigram_author_plot <- function(author, fill) {
  train |>
    filter(name == author) |> 
    unnest_tokens(bigram, text, token = "ngrams", n = 3) |>
    separate(bigram, c("word1", "word2", "word3"), sep = " ") |>
    filter(!word1 %in% stopwords(kind = "es"),
           !word2 %in% stopwords(kind = "es"),
           !word3 %in% stopwords(kind = "es")) |>
    drop_na() |> 
    unite(trigramWord, word1, word2, word3, sep = " ") |>
    group_by(trigramWord) |>
    tally() |>
    ungroup() |>
    arrange(desc(n)) |>
    mutate(trigramWord = reorder(trigramWord,n)) |>
    head(10) |>
    ggplot(aes(x = trigramWord,y = n)) +
    geom_bar(stat='identity', fill = fill) +
    geom_text(aes(x = trigramWord, y = 1, label = n),
              hjust=0, vjust=.5, size = 4, color = "white") +
    labs(x = 'Trigramas', 
         y = 'Conteo', 
         title = paste('Conteo de Trigramas más\ncomunes para', author)) +
    coord_flip() + 
    theme_minimal() +
    theme(
      plot.title = element_text(hjust = 0.5),
      plot.title.position = "plot",
      plot.background = element_rect(fill = "white", color = "white")
    )
}

create_trigram_author_plot("Lopez", "#006d2c")
ggsave("Views/Top_10_trigramas_Lopez.png", width = 1500, height = 1500, units = "px")
create_trigram_author_plot("Petro", "#54278f")
ggsave("Views/Top_10_trigramas_Petro.png", width = 1500, height = 1500, units = "px")
create_trigram_author_plot("Uribe", "#08519c")
ggsave("Views/Top_10_trigramas_Uribe.png", width = 1500, height = 1500, units = "px")


### Análisis de sentimientos
visualize_sentiments <- function(SCWords) {
  
  SCWords_sentiments <- SCWords |> 
    mutate(score = get_sentiment(word, method = "nrc", language = "spanish")) |> 
    group_by(name) |> 
    summarize(score = sum(score * n) / sum(n)) |> 
    arrange(desc(score))
  
  SCWords_sentiments |> 
    mutate(author = reorder(name, score)) |> 
    ggplot(aes(name, score, fill = score>0)) +
    geom_col(show.legend = TRUE) +
    coord_flip() +
    labs(
      y = "Puntaje",
      x = "Autor",
      title = "Puntaje promedio de sentimientos\npor autor",
      caption = "Basado en el dataset nrc"
    ) +
    theme_minimal()+
    theme(
      plot.title = element_text(hjust = 0.5),
      plot.title.position = "plot",
      plot.background = element_rect(fill = "white", color = "white")
    )
}

visualize_sentiments(trainWords)

ggsave("View/score_sentiments.png", width = 1500, height = 1500, units = "px")

create_bar_chart_positive_words <- function(Words, author){
  
  contributions <- Words |> 
    filter(name == author) |> 
    unnest_tokens(word, text) |> 
    count(name, word, sort = TRUE) |> 
    ungroup() |> 
    mutate(score = get_sentiment(word, method = "nrc", language = "spanish")) |> 
    group_by(word) |> 
    summarize(occurences = n(),
              contribution = sum(score))
  
  contributions |> 
    top_n(20, abs(contribution)) |> 
    mutate(word = reorder(word, contribution)) |> 
    head(20) |> 
    ggplot(aes(word, contribution, fill = contribution > 0)) +
    geom_col(show.legend = FALSE) +
    labs(
      title = paste("Top 20 de palabras con mayor\ncontribución al puntaje de\nsentimientos para", author),
      x = "Palabra",
      y = "Contribución"
    ) +
    coord_flip() +
    theme_minimal() +
    theme(
      plot.title = element_text(hjust = 0.5),
      plot.title.position = "plot",
      plot.background = element_rect(fill = "white", color = "white")
    )
}

create_bar_chart_positive_words(train, "Lopez")
ggsave("View/Top_20_contribuciones_sentimientos_Lopez.png", width = 1500, height = 1500, units = "px")
create_bar_chart_positive_words(train, "Uribe")
ggsave("View/Top_20_contribuciones_sentimientos_Uribe.png", width = 1500, height = 1500, units = "px")
create_bar_chart_positive_words(train, "Petro")
ggsave("View/Top_20_contribuciones_sentimientos_Petro.png", width = 1500, height = 1500, units = "px")



### intentamos obtener más información de los signos de puntuación para intentar
### relacionar los políticos con un estilo de escritura, esto lo ingresamos a la data final para darle
### más capacidad predictiva al modelo

implement_fe <- function(dataset) {
  dataset |> 
    mutate(Ncommas = str_count(`text`, ",")) |> 
    mutate(Nsemicolons = str_count(`text`, ";")) |> 
    mutate(Ncolons = str_count(`text`, ":")) |> 
    mutate(Nblank = str_count(`text`, " ")) |> 
    mutate(Nother = str_count(`text`, "[\\.\\.]")) |> 
    mutate(Ncapitalfirst = str_count(`text`," [A-Z][a-z]")) |> 
    mutate(Ncapital = str_count(`text`, "[A-Z]"))
}

train_fe <- implement_fe(train)
test_fe <- implement_fe(test)


make_DTM <- function(train, tfidfFlag = T) {
  
  corpus = Corpus(VectorSource(train$text))
  
  # Pre-process data
  corpus <- tm_map(corpus, tolower)
  corpus <- tm_map(corpus, removePunctuation)
  corpus <- tm_map(corpus, removeWords, stopwords(kind = "es"))
  corpus <- tm_map(corpus, stemDocument)
  
  if(tfidfFlag == 1 ) {
    dtm = DocumentTermMatrix(corpus,control = list(weighting = weightTfIdf))
  } else  {
    dtm = DocumentTermMatrix(corpus)
  }
  
  # Remove sparse terms
  dtm = removeSparseTerms(dtm, 0.997)
  
  # Create data frame
  labeledTerms = as.data.frame(as.matrix(dtm))
  
  return(labeledTerms)
}


make_features <- function(train) {
  labeledTerms <- make_DTM(train, tfidfFlag = 0)
  
  features <- colnames(labeledTerms)
  
  for(f in features) {
    if ((class(labeledTerms[[f]])=="factor") || (class(labeledTerms[[f]])=="character")) {
      levels <- unique(labeledTerms[[f]])
      labeledTerms[[f]] <- as.numeric(factor(labeledTerms[[f]], levels = levels))
    }
  }
  
  return(labeledTerms)
}

labeledTermsTrain <- make_features(train)

labeledTermsTest <- make_features(test)

colnamesSame <- intersect(colnames(labeledTermsTrain), colnames(labeledTermsTest))

filteredTermsTrain <- labeledTermsTrain[, (colnames(labeledTermsTrain) %in% colnamesSame)]
filteredTermsTest <- labeledTermsTest[, (colnames(labeledTermsTest) %in% colnamesSame)]



### Incorporamos las caracteristicas creadas
finalTermsTrain <- bind_cols(
  filteredTermsTrain,
  len = train$len,
  Ncommas = train_fe$Ncommas,
  Nsemicolons = train_fe$Nsemicolons,
  Ncolons = train_fe$Ncolons,
  Nblank = train_fe$Nblank,
  Nother = train_fe$Nother,
  Ncapitalfirst = train_fe$Ncapitalfirst,
  Ncapital = train_fe$Ncapital
)

finalTermsTest <- bind_cols(
  filteredTermsTest,
  len = test$len,
  Ncommas = test_fe$Ncommas,
  Nsemicolons = test_fe$Nsemicolons,
  Ncolons = test_fe$Ncolons,
  Nblank = test_fe$Nblank,
  Nother = test_fe$Nother,
  Ncapitalfirst = test_fe$Ncapitalfirst,
  Ncapital = test_fe$Ncapital
)


finalTermsTrain$name <- as.factor(train$name)
levels(finalTermsTrain$name) <- make.names(unique(finalTermsTrain$name))

## Empezamos a ejecutar modelos:


### Random Forest ----
rf_clf <- caret::train(
  name ~ .,
  data = finalTermsTrain,
  method = "ranger",
  trControl = trainControl(
    method = "cv",
    number = 3,
    verboseIter = TRUE
  )
)

# saveRDS(rf_clf, "Models/random_forest.rds")

rf_pred <- data.frame(
  id = test$id,
  name = predict(rf_clf, finalTermsTest, type = "raw")
)

# write_csv(rf_pred, "Resultado/random_forest.csv")

## XGBoost

xgb_clf <- caret::train(
  name ~ .,
  data = finalTermsTrain,
  method = "xgbTree",
  trControl = trainControl(
    method = "cv",
    number = 3,
    verboseIter = TRUE
  ),
  na.action = na.pass
)

saveRDS(xgb_clf, "Models/xgboost.rds")

xgb_pred <- data.frame(
  id = test$id,
  name = predict(xgb_clf, finalTermsTest, type = "raw")
)

# write_csv(xgb_pred, "Resultado/xgboost.csv")

## Naive Bayes

nb_clf <- caret::train(
  name ~ .,
  data = finalTermsTrain,
  method = "naive_bayes",
  trControl = trainControl(
    method = "cv",
    number = 3,
    verboseIter = TRUE
  ),
  na.action = na.pass
)

saveRDS(nb_clf, "Models/naive_bayes.rds")

nb_pred <- data.frame(
  id = test$id,
  name = predict(nb_clf, finalTermsTest, type = "raw")
)

write_csv(nb_pred, "Resultado/naive_bayes.csv")

## Multinomial Logistic Regression Model

glmnet_clf <- caret::train(
  name ~ .,
  data = finalTermsTrain,
  method = "glmnet",
  trControl = trainControl(
    method = "cv",
    number = 3,
    verboseIter = TRUE
  ),
  na.action = na.pass
)

saveRDS(glmnet_clf, "Models/glmnet.rds")

glmnet_pred <- data.frame(
  id = test$id,
  name = predict(glmnet_clf, as.matrix(finalTermsTest), type = "raw")
)

write_csv(glmnet_pred, "Resultado/glmnet.csv")


## Redes Neuronales
## Trabajaremos la red Neuronal en Google, pero antes, preparamos la data Colab




