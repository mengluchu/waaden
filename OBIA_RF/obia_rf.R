library(sf)
library(caret)
library(ranger)
library(dplyr)
library(vip)
a =st_read("~/Downloads/ObjectShapes/Westerschelde_2016_obj_creation.v2.shp")
cla =st_read("~/Downloads/2016_GMK/e_GMK_Westerschelde2016.shp")
names(a)
#plot(st_geometry(a))
unique(levels(a$OMS_GEOCOD))
cla_f = st_join(cla, a, left =F)


#cla_s= cla_f %>%select(matches("GLCM"), 
#                OMS_GEOCOD,
#                matches("Rel"), matches("Number_of_"), Shape_inde)
#                 Brightness,Compactnes,
cla_s = cla_f%>%select(names(a), OMS_GEOCOD)
#Shape_Leng Shape_Area
st_geometry(cla_s) <- NULL
data = cla_s
training <- sample(1:nrow(data),nrow(data)*0.8)                             #Craete training dataset containing about 90 % of the total observations
testing <- setdiff(1:nrow(data),training)                         #Create testing dataset containing the remaining 10%
training <- data[training,]
testing <- data[testing ,]

r1=ranger(OMS_GEOCOD~., data = training, importance = "impurity")


importance(r1)%>%names
#barplot(sort(importance(r1), decreasing = F), horiz = T, las  =1 )
vip (r1,width = 0.2, num_features = 20, geom = "point", aesthetics = list(fill = "green3", shape = 17, size = 4))
#sort(importance(r1), decreasing = T)
pred = predict(r1, data = testing)
image(table(testing$OMS_GEOCOD, predictions(pred)))

#table(testing$OMS_GEOCOD, predictions(pred))
cm = confusionMatrix(data = predictions(pred), reference = testing$OMS_GEOCOD, mode = "prec_recall")
r1
cm$overall
#cm$byClass