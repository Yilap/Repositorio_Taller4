################################################################
# Problem Set 2: Predicting Poverty
# Authors: Yilmer Palacios, Betina Cortés, Lida Jimena Rivera,
# Nelson Fabián López
################################################################

# Loading Libraries -------------------------------------------------------

rm(list = ls()) 

#install.packages("pacman")
#install.packages("httr")

library("pacman") # para cargar paquetes
p_load("tidyverse",
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


# Importing Dataset -------------------------------------------------------
