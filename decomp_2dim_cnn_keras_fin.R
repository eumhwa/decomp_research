####################
#######CNN##########
####################

####################################################################################
###Decomposition###
# B : Beef
# F : Fish
# P : Pork
# T : Tissue
# Blank
####################################################################################
rm(list=ls())
#-----------------------------------------------------------------------------------
#Set working Directory
setwd("D:/dyu2017/decomposition_csv")

#load(file='after_binning2')

#Packages
library(keras)
library(kerasR)
library(reticulate)
library(tensorflow)
library(dplyr)
library(ROCR)

#-----------------------------------------------------------------------------------
#initial settings for using keras
#use_condaenv("tensorflow", required = TRUE) 
use_python("C://Users//PC2//Anaconda3//envs//tensorflow//python")
keras_init()
keras_available() #keras 사용가능한지 확인
py_config()


#-----------------------------------------------------------------------------------
#data : rdata (list)
#bin : bret (length:4036)
#opt_bin : bd_t (length:325)

#opt binning with 2-dim data
# 
# bin_out <- matrix(0, 324, 324)
# bindata <- list()
# for(i in 1:length(rdata)){
#   
#   tmp <- rdata[[i]][,-c(1,3)]
#   tmp <- tmp[,1:325]
#   RT <- tmp[,1]
#   
#   for(j in 1:(length(bd_t)-1)){
#     
#     id <- which(RT>=bd_t[j] & RT<bd_t[j+1])
#     sub_tmp <- tmp[id, -1]
#     sub_tmp <- as.vector(apply(sub_tmp, 2, sum))
#     
#     bin_out[j,] <- sub_tmp
#     
#   }
#   
#   bindata[[i]] <- bin_out
#   cat(i," - ")
# }


#-----------------------------------------------------------------------------------
#save(list=ls(),file='two_dim_data.rdata')
load(file='two_dim_data.rdata')
#-----------------------------------------------------------------------------------

bindata_ <- lapply(bindata, t)
newdata <- lapply(bindata_, as.vector)
newdt <- do.call(rbind, newdata)


#model build
{
  mod <- keras_model_sequential()
  mod %>%
    layer_conv_2d(filter=1, kernel_size=c(3,3), padding="same", input_shape=c(324,324,1)) %>%
    layer_activation("relu") %>%
    layer_conv_2d(filter=1, kernel_size=c(3,3)) %>%
    layer_activation("relu") %>%
    layer_max_pooling_2d(pool_size=c(2,2)) %>%
    layer_dropout(0.3) %>%
    layer_batch_normalization() %>%
    
    layer_conv_2d(filter=2, kernel_size=c(3,3), padding="same", input_shape=c(324,324,1)) %>%
    layer_activation("relu") %>%
    layer_conv_2d(filter=2, kernel_size=c(3,3)) %>%
    layer_activation("relu") %>%
    layer_max_pooling_2d(pool_size=c(2,2)) %>%
    layer_dropout(0.3) %>%
    layer_batch_normalization() %>%
    
    layer_conv_2d(filter=4 , kernel_size=c(3,3),padding="same") %>% 
    layer_activation("relu") %>%  
    layer_conv_2d(filter=4,kernel_size=c(3,3) ) %>%  
    layer_activation("relu") %>%  
    layer_max_pooling_2d(pool_size=c(2,2)) %>%  
    layer_dropout(0.3) %>%
    
    #flatten the input  
    layer_flatten() %>%  
    layer_dense(128) %>%  
    layer_activation("relu") %>%  
    layer_dropout(0.3) %>%  
    #output layer-10 classes-2 units  
    layer_dense(2) %>%  
    
    #applying softmax nonlinear activation function to the output layer #to calculate cross-entropy  
    layer_activation("softmax") 
}




#-----------------------------------------------------------------------------------
#initializing mats & vecs
resp2 <- ifelse(resp=="4", 1, 0)
dic_acc <- c()
dic.auc <- c()
dic_tb <- list()

prob <- matrix(0,24,50)


performance <- function(tb){
  sensitivity <- tb[2,2]/sum(tb[2,])
  specificity <- tb[1,1]/sum(tb[1,])
  FDR <- tb[1,2]/(tb[2,2]+tb[1,2])
  Accuracy <- (tb[1,1]+tb[2,2])/(tb[1,1]+tb[1,2]+tb[2,1]+tb[2,2])
  F1.score <- tb[2,2]/(2*tb[2,2]+tb[1,2]+tb[2,1]) 
  
  return(list(sensitivity=sensitivity, specificity=specificity, FDR=FDR, Accuracy=Accuracy, F1.score=F1.score))
}

minmax_scaler <- function(A){
  maxA <- max(A)
  minA <- min(A)
  
  out <- (maxA-A)/(maxA-minA)
  
  return(out)
}
tb_fn <- function(y, p){
  m <- matrix(0,2,2)
  
  m[1,1] <- sum(y==0 & p==0)
  m[1,2] <- sum(y==0 & p==1)
  m[2,1] <- sum(y==1 & p==0)
  m[2,2] <- sum(y==1 & p==1)
  
  return(m)
}

#iteration
for(i in 1:50){
  
  #model build
  {
    m1 <- layer_input(shape = c(324,324,1))
    
    #vgg block
    x2 = layer_conv_2d(m1, filter=8, kernel_size=c(3,3), padding="same")
    x2 = layer_activation_leaky_relu(x2)
    x2 = layer_conv_2d(x2, filter=8, kernel_size=c(3,3), padding="same")
    x2 = layer_activation_leaky_relu(x2)
    x2 = layer_max_pooling_2d(x2, pool_size=c(2,2))
    x2 = layer_dropout(x2, 0.3)
    skip_conn1 = layer_batch_normalization(x2)
    
    #resnet block1
    x2 = layer_conv_2d(skip_conn1, filter=2, padding="same", kernel_size=c(1,1))
    x2 = layer_activation_leaky_relu(x2)
    x2 = layer_batch_normalization(x2)
    x2 = layer_conv_2d(x2, filter=2, padding="same", kernel_size=c(3,3))
    x2 = layer_activation_leaky_relu(x2)
    x2 = layer_batch_normalization(x2)
    x2 = layer_conv_2d(x2, filter=8, kernel_size=c(1,1))
    
    x2 = layer_add(list(x2, skip_conn1))
    x2 = layer_activation_leaky_relu(x2)
    x2 = layer_dropout(x2, 0.3)
    lin_proj = layer_batch_normalization(x2)
    
    #resnet block3 - reduction of feature_map size
    x2 = layer_conv_2d(lin_proj, filter=4, padding="same", kernel_size=c(1,1), stride=2)
    x2 = layer_activation_leaky_relu(x2)
    x2 = layer_batch_normalization(x2)
    x2 = layer_conv_2d(x2, filter=4, padding="same", kernel_size=c(3,3))
    x2 = layer_activation_leaky_relu(x2)
    x2 = layer_batch_normalization(x2)
    x2 = layer_conv_2d(x2, filter=16, kernel_size=c(1,1))
    x2 = layer_activation_leaky_relu(x2)
    skip_conn3 = layer_batch_normalization(x2)
    
    lin_proj = layer_conv_2d(lin_proj, filter=16, padding="same", kernel_size=c(1,1), stride=2)
    
    
    x2 = layer_add(list(lin_proj, skip_conn3))
    x2 = layer_dropout(x2, 0.3)
    x2 = layer_batch_normalization(x2)
    x2 = layer_activation_leaky_relu(x2)
    
    
    #resnet block4
    # x2 = layer_conv_2d(skip_conn4, filter=16, padding="same", kernel_size=c(1,1))
    # x2 = layer_activation_leaky_relu(x2)
    # x2 = layer_batch_normalization(x2)
    # x2 = layer_conv_2d(x2, filter=16, padding="same", kernel_size=c(3,3))
    # x2 = layer_activation_leaky_relu(x2)
    # x2 = layer_batch_normalization(x2)
    # x2 = layer_conv_2d(x2, filter=32, kernel_size=c(1,1))
    
    # x2 = layer_add(list(x2, skip_conn4))
    # x2 = layer_activation_leaky_relu(x2)
    # x2 = layer_batch_normalization(x2)
    
    #last avg_pooling
    x2 = layer_average_pooling_2d(x2, pool_size=c(2,2))
    
    #FC layer
    x2 = layer_flatten(x2)
    x2 = layer_dense(x2, 2)
    x2 = layer_activation(x2, "softmax")
    
    model <- keras_model(inputs = m1, output=x2)
    
    adam <- optimizer_adam(lr=0.001, decay=1e-6)
    
  }
  
  
  set.seed(i)
  
  lab_idx1 <- sample(which(resp=="1"), 6)
  lab_idx2 <- sample(which(resp=="2"), 6)
  lab_idx3 <- sample(which(resp=="3"), 6)
  lab_idx4 <- sample(which(resp=="4"), 6)
  lab_idx  <- c(lab_idx1,lab_idx2,lab_idx3,lab_idx4)
  
  #newdt_scaled <- t(apply(newdt, 1, minmax_scaler))
  
  train_x <- newdt[-lab_idx,]
  train_y <- resp2[-lab_idx]
  
  test_x <- newdt[lab_idx,]
  test_y <- resp2[lab_idx]
  
  train_x <- array_reshape(x=train_x, dim=list(72,324,324,1))
  train_y <- to_categorical(train_y, num_classes = 2)
  
  test_x <- array_reshape(x=test_x, dim=list(24,324,324,1))
  test_y <- to_categorical(test_y, num_classes = 2)
  
  #dim(train_x);dim(test_x)
  cat(i,"-th data setting--\n")
  
  model %>%
    compile(loss="categorical_crossentropy",
            optimizer=adam,
            metrics = "accuracy")
  
  
  early_stop <- callback_early_stopping(monitor = "val_loss", patience = 20)
  
  model %>% fit(train_x, train_y , batch_size=6,
                epochs=100, validation_split=0.1,
                shuffle=TRUE,
                callbacks = list(early_stop)
  )
  
  
  scores <- model %>% evaluate(
    test_x, test_y, verbose = 0
  )
  
  dic_acc[i] <- scores$acc
  
  pred_ <- model %>% predict(test_x)
  pred <- round(pred_)[,2]
  prob[,i] <- pred_[,2]
  cb <- tb_fn(test_y[,2], pred)
  
  auc_pred <- ROCR::prediction(pred_, test_y)
  auc_pref <- ROCR::performance(auc_pred, "tpr", "fpr")
  auc <- ROCR::performance(auc_pred, "auc")
  dic.auc[i] <- unlist(auc@y.values)
  
  dic_tb[[i]] <- cb
  
  k_clear_session()
  
  cat(i," iter ends\n ")
  
}



#-----------------------------------------------------------------------------------
#test error

result <- unlist(sapply(dic_tb, performance))
mean(dic_acc)

res <- matrix(result, 5, 50, 
              dimnames = list(c("sensitivity","specificity", "FDR", "Accuracy", "F1.score"), 
                              c(1:50)))

apply(res,1,mean)
apply(res, 1, sd)/sqrt(50)

tttt <- dic_tb


mean(dic.auc)
sd(dic.auc)/sqrt(50)



#-----------------------------------------------------------------------------------
#ROC curve & FDR curve

tb_fn <- function(y, p){
  m <- matrix(0,2,2)
  
  m[1,1] <- sum(y==0 & p==0)
  m[1,2] <- sum(y==0 & p==1)
  m[2,1] <- sum(y==1 & p==0)
  m[2,2] <- sum(y==1 & p==1)
  
  return(m)
}

thresh <- seq(0.0001, 0.9999, 0.0001)
v <- rep(9999,length(thresh))
SEN <- rep(list(v),50)
SPE <- rep(list(v),50)
FDR <- rep(list(v),50)

for(i in 1:50){
  for(j in 1:length(thresh)){
    cutoff <- thresh[j]
    p <- as.numeric(ifelse(prob[,i]>=cutoff, 1, 0))
    tb <- tb_fn(test_y[,2],p)
    SEN[[i]][j] <- tb[2,2]/sum(tb[2,])
    SPE[[i]][j] <- tb[1,1]/sum(tb[1,])
    if(sum(tb[,2])!=0) FDR[[i]][j] <-  tb[1,2]/(tb[2,2]+tb[1,2])
    else FDR[[i]][j] <- 0
  }
  cat(i,'-')
}

SEN_cnn <- do.call(rbind, SEN)
SPE_cnn <- do.call(rbind, SPE)
FDR_cnn <- do.call(rbind, FDR)

SEN_cnn <- apply(SEN_cnn, 2, mean)
SPE_cnn <- apply(SPE_cnn, 2, mean)
FDR_cnn <- apply(FDR_cnn, 2, mean)



#1)roc curve
ROC_df_cnn <- data.frame(false_positive_rate=(1-SPE_cnn), true_positive_rate=SEN_cnn)

plot(x=ROC_df_cnn$false_positive_rate, ROC_df_cnn$true_positive_rate, xlim=c(0,1), ylim=c(0,1), type="l", lty=1, 
     xlab="False Positive Rate", ylab="True Positive Rate")

#2)fdr curve
plot(x=thresh,y=FDR_cnn , xlim=c(0,1), ylim=c(-0.5,1), type="l", lty=1, 
     xlab="Threshold", ylab="False Discovery Rate")

out_cnn <- data.frame(ROC_df_cnn, FDR_cnn=FDR_cnn)
write.csv(out_cnn,"D://dyu2017//auc_fdr//out_cnn.csv", row.names = F)


#----------------------------------------------------------------------------
#Cut value (youden index, sen_fdr)

best_thr_id <- function(x) apply(x, 1, which.max)

SEN_cnn <- do.call(rbind, SEN)
SPE_cnn <- do.call(rbind, SPE)
FDR_cnn <- do.call(rbind, FDR)

youden_cnn <- SEN_cnn + SPE_cnn
sen_fdr_cnn <- SEN_cnn*(1-FDR_cnn)

youden_id_cnn <- thresh[best_thr_id(youden_cnn)]
sen_fdr_id_cnn <- thresh[best_thr_id(sen_fdr_cnn)]


tb_cnn1 <- list(); tb_cnn2 <- list()

for(i in 1:50){
  
  cutoff_cnn1 <- youden_id_cnn[i]
  cutoff_cnn2 <- sen_fdr_id_cnn[i]
  
  p1 <- as.numeric(ifelse(prob[,i]>=cutoff_cnn1, 1, 0))
  p2 <- as.numeric(ifelse(prob[,i]>=cutoff_cnn2, 1, 0))
  tb_cnn1[[i]] <- tb_fn(test_y[,2], p1) #youden 
  tb_cnn2[[i]] <- tb_fn(test_y[,2], p2) #sen(1-fdr) 
  
  cat(i,'-')
  
}



#youden index --> tb_r1, tb_l1
cnn_result1 <- unlist(sapply(tb_cnn1, performance))

cnn_res1 <- matrix(cnn_result1, 5, 50, 
                  dimnames = list(c("sensitivity","specificity", "FDR", "Accuracy", "F1.score"), 
                                  c(1:50)))

apply(cnn_res1, 1, mean)
apply(cnn_res1, 1, sd)/sqrt(50)


#SEN(1-FDR) --> tb_r2, tb_l2
cnn_result2 <- unlist(sapply(tb_cnn2, performance))

cnn_res2 <- matrix(cnn_result2, 5, 50, 
                  dimnames = list(c("sensitivity","specificity", "FDR", "Accuracy", "F1.score"), 
                                  c(1:50)))

apply(cnn_res2, 1, mean)
apply(cnn_res2, 1, sd)/sqrt(50)


######################################################################################################
#save(list=ls(),file='decomp_cnn_fin_result')
load(file='decomp_cnn_fin_result')


######################################################################################################
