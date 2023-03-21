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

