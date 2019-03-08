#######################
##Deep Neural Network##
#######################

####################################################################################
###Decomposition###
# B : Beef
# F : Fish
# P : Pork
# T : Tissue
# Blank
####################################################################################

#-----------------------------------------------------------------------------------
#Set working Directory
setwd("D:/dyu2017/decomposition_csv")

#Packages
library(caret)
library(h2o)
library(ROCR)
search()
#-----------------------------------------------------------------------------------



####################################################################################
rm(list=ls())

load(file='after_binning2')
####################################################################################





#-----------------------------------------------------------------------------------
##data 정리 
o_indx = c(rep(c(rep(T,4),rep(T,4)),3),
           rep(c(rep(T,4),rep(T,4)),3),
           rep(c(rep(T,4),rep(T,4)),3),
           rep(c(rep(T,4),rep(T,4)),3))


nn = sum(o_indx)

chk = apply(alg_dat!=0,2,sum)      #0이 아닌 값을 1(TRUE)로 열 기준 sum
slt_dat = alg_dat[,chk>1]          #값이 존재하는 열만 보기
dim(slt_dat)
sum(chk>1)                         #0이 아닌 값의 갯수
std_dat = slt_dat[o_indx,]         #slt_dat 이랑 같은듯
dim(std_dat)
resp = resp[o_indx]



#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
#Initializing the vecs & mats

h2o.init(nthreads = -1)

performance <- function(tb){
  sensitivity <- tb[2,2]/sum(tb[2,])
  specificity <- tb[1,1]/sum(tb[1,])
  FDR <- tb[1,2]/(tb[2,2]+tb[1,2])
  Accuracy <- (tb[1,1]+tb[2,2])/(tb[1,1]+tb[1,2]+tb[2,1]+tb[2,2])
  F1.score <- tb[2,2]/(2*tb[2,2]+tb[1,2]+tb[2,1]) 
  
  return(list(sensitivity=sensitivity, specificity=specificity, FDR=FDR, Accuracy=Accuracy, F1.score=F1.score))
}

prob <- matrix(0,24,50)

dnn.fit <- list()
tb.list <- list()
dic.auc <- c()
iter <- 1




#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
#DNN

while(iter<51){

  set.seed(iter)
  
  ##data partitioning
  idx1 <- sample(which(resp=="1"),6, replace=FALSE)
  idx2 <- sample(which(resp=="2"),6, replace=FALSE)
  idx3 <- sample(which(resp=="3"),6, replace=FALSE)
  idx4 <- sample(which(resp=="4"),6, replace=FALSE)
  idx <- c(idx1, idx2, idx3, idx4)

  resp2 <- resp


  resp2[which(resp %in% c("1","2","3"))] <- "0"
  resp2[which(resp=="4")] <- "1"
  #table(resp2)



  train_xdt <- std_dat[-idx,]
  train_ydt <- as.numeric(resp2[-idx])

  train <- as.data.frame(cbind(train_ydt, train_xdt))
  train$train_ydt <- as.factor(train$train_ydt)
  train <- as.h2o(train) # h2o형태 맞춰줘야함!

  
  test_xdt <- as.h2o(std_dat[idx,])
  test_ydt <- resp2[idx]



  ##model fitting
  dnn.fit[[iter]] <- h2o.deeplearning(x = 2:ncol(train), #column index
                                      y = 1,
                                      training_frame = train,
                                      distribution = "bernoulli",
                                      activation = "RectifierWithDropout",
                                      hidden = c(ncol(train), ncol(train)*1.6, ncol(train)*1.2, ncol(train)*0.8),
                                      epochs = 300,
                                      loss = "CrossEntropy",
                                      mini_batch_size = 12,
                                      hidden_dropout_ratios = rep(0.5,4), #default:0.5
                                      seed = iter*10
                                      )


  ##prediction
  pred <- h2o.predict(dnn.fit[[iter]], newdata=test_xdt, type="response")
  pred <- as.matrix(pred)
  pp <- c()
  for(i in 1:length(test_ydt)){
    if(as.numeric(pred[i,2]) > as.numeric(pred[i,3])) pp[i] <- "0"
    else pp[i] <- "1"
  }
  
  prob[,iter] <- pred[,3]
  
  auc_pred <- ROCR::prediction(as.numeric(as.numeric(pred[,3])), test_ydt)
  auc_pref <- ROCR::performance(auc_pred, "tpr", "fpr")
  auc <- ROCR::performance(auc_pred, "auc")
  dic.auc[iter] <- unlist(auc@y.values)
  

  tb.list[[iter]] <- table(test_ydt,pp)
  
  cat('End of ',iter,' th iteration')
  
  iter <- iter+1
}

#----------------------------------------------------------------------------
#----------------------------------------------------------------------------




#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
#Test result
tb.list
dnn.res <- unlist(sapply(tb.list, performance))

res <- matrix(dnn.res, 5, 50, 
              dimnames = list(c("sensitivity","specificity", "FDR", "Accuracy", "F1.score"), 
                              c(1:50)))

apply(res,1,mean)
apply(res, 1, sd)/sqrt(50)

dnn.fit[[1]]

mean(dic.auc)
sd(dic.auc)/sqrt(50)

#----------------------------------------------------------------------------
#----------------------------------------------------------------------------



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

prob2 <- apply(prob, 2, FUN=as.numeric)

for(i in 1:50){
  for(j in 1:length(thresh)){
    cutoff <- thresh[j]
    p <- as.numeric(ifelse(prob2[,i]>=cutoff, 1, 0))
    tb <- tb_fn(test_ydt,p)
    SEN[[i]][j] <- tb[2,2]/sum(tb[2,])
    SPE[[i]][j] <- tb[1,1]/sum(tb[1,])
    if(sum(tb[,2])!=0) FDR[[i]][j] <-  tb[1,2]/(tb[2,2]+tb[1,2])
    else FDR[[i]][j] <- 0
  }
  cat(i,'-')
}

SEN_dnn <- do.call(rbind, SEN)
SPE_dnn <- do.call(rbind, SPE)
FDR_dnn <- do.call(rbind, FDR)

SEN_dnn <- apply(SEN_dnn, 2, mean)
SPE_dnn <- apply(SPE_dnn, 2, mean)
FDR_dnn <- apply(FDR_dnn, 2, mean)



#1)roc curve
ROC_df_dnn <- data.frame(false_positive_rate=(1-SPE_dnn), true_positive_rate=SEN_dnn)
ROC_df_dnn <- rbind(c(1,1), ROC_df_dnn, c(0,0))

plot(x=ROC_df_dnn$false_positive_rate, ROC_df_dnn$true_positive_rate, xlim=c(0,1), ylim=c(0,1), type="l", lty=1, 
     xlab="False Positive Rate", ylab="True Positive Rate")

#2)fdr curve
plot(x=thresh,y=FDR_dnn , xlim=c(0,1), ylim=c(-0.5,1), type="l", lty=1, 
     xlab="Threshold", ylab="False Discovery Rate")

out_dnn <- data.frame(ROC_df_dnn, FDR_dnn=FDR_dnn)
write.csv(out_dnn,"D://dyu2017//auc_fdr//out_dnn.csv", row.names = F)



#----------------------------------------------------------------------------
#Cut value (youden index, sen_fdr)

best_thr_id <- function(x) apply(x, 1, which.max)

SEN_dnn <- do.call(rbind, SEN)
SPE_dnn <- do.call(rbind, SPE)
FDR_dnn <- do.call(rbind, FDR)

youden_dnn <- SEN_dnn + SPE_dnn
sen_fdr_dnn <- SEN_dnn*(1-FDR_dnn)


youden_id_dnn <- thresh[best_thr_id(youden_dnn)]
sen_fdr_id_dnn <- thresh[best_thr_id(sen_fdr_dnn)]


tb_dnn1 <- list(); tb_dnn2 <- list()

for(i in 1:50){
  
  cutoff_dnn1 <- youden_id_dnn[i]
  cutoff_dnn2 <- sen_fdr_id_dnn[i]
  
  p1 <- as.numeric(ifelse(prob2[,i]>=cutoff_dnn1, 1, 0))
  p2 <- as.numeric(ifelse(prob2[,i]>=cutoff_dnn2, 1, 0))
  tb_dnn1[[i]] <- tb_fn(test_ydt, p1) #youden 
  tb_dnn2[[i]] <- tb_fn(test_ydt, p2) #sen(1-fdr) 
  
  cat(i,'-')
  
}



#youden index --> tb_r1, tb_l1
dnn_result1 <- unlist(sapply(tb_dnn1, performance))

dnn_res1 <- matrix(dnn_result1, 5, 50, 
                  dimnames = list(c("sensitivity","specificity", "FDR", "Accuracy", "F1.score"), 
                                  c(1:50)))

apply(dnn_res1, 1, mean)
apply(dnn_res1, 1, sd)/sqrt(50)


#SEN(1-FDR) --> tb_r2, tb_l2
dnn_result2 <- unlist(sapply(tb_dnn2, performance))

dnn_res2 <- matrix(dnn_result2, 5, 50, 
                  dimnames = list(c("sensitivity","specificity", "FDR", "Accuracy", "F1.score"), 
                                  c(1:50)))

apply(dnn_res2, 1, mean)
apply(dnn_res2, 1, sd)/sqrt(50)


#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
#save(list=ls(),file='decomp_DNN_result')
load(file='decomp_DNN_result')
