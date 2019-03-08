################
####XGboost####
################
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
setwd("D:/dyu2017/decomposition_csv")

library(xgboost)
library(caret)
library(e1071)
library(ROCR)

#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
rm(list=ls())

load(file='after_binning2')

performance <- function(tb){
  sensitivity <- tb[2,2]/sum(tb[2,])
  specificity <- tb[1,1]/sum(tb[1,])
  FDR <- tb[1,2]/(tb[2,2]+tb[1,2])
  Accuracy <- (tb[1,1]+tb[2,2])/(tb[1,1]+tb[1,2]+tb[2,1]+tb[2,2])
  F1.score <- tb[2,2]/(2*tb[2,2]+tb[1,2]+tb[2,1]) 
  
  return(list(sensitivity=sensitivity, specificity=specificity, FDR=FDR, Accuracy=Accuracy, F1.score=F1.score))
}


#----------------------------------------------------------------------------
#----------------------------------------------------------------------------

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

#fix(std_dat)

#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
iter <- 1
names <- c("1", "0")
m <- matrix(0, 24, 10)
prob <- rep(list(m),50)
tb <- list()
dic.auc <- c()
xgb_finalModel <- list()

#----------------------------------------------------------------------------
#----------------------------------------------------------------------------

while(iter < 51){
  
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

  train_xdt2 <- std_dat[-idx,]
  train_ydt <- resp2[-idx]

  test_xdt2 <- std_dat[idx,]
  test_ydt <- resp2[idx]



  #training
  #mtry <- sqrt(ncol(train_xdt2)) #변수갯수
  metric <- "Accuracy"
  xgb.control <- trainControl(method="cv", 
                              number=10, #10-fold
                              search="grid", 
                              classProbs = TRUE,
                              #summaryFunction = twoClassSummary,
                              allowParallel = TRUE
                              ) 
                              #classProbs, summaryFunction은 ROC를 metric으로 쓸때 필요한 옵션
  
  xgb.grid <- expand.grid(eta=c(0.05, 0.1),
                          gamma= 1,
                          max_depth=c(2,3,5),
                          nrounds=c(80,100,120),
                          colsample_bytree=c(0.6,0.8),
                          subsample=c(0.6,0.8),
                          min_child_weight=1
                          )

  
#when using ROC for metric#
 levels <- unique(train_ydt)
 train_ydt2 <- factor(train_ydt, labels=make.names(levels))

 levels <- unique(test_ydt)
 test_ydt2 <- factor(test_ydt, labels=make.names(levels))

 #order(levels,decreasing = TRUE)



  #XGBoost
  w <- ifelse(as.numeric(train_ydt)==0,1,1.5)
  xgb_fit <- train(x = train_xdt2, 
                   y = train_ydt2, 
                   method="xgbTree", 
                   objective="binary:logistic",
                   metric=metric, 
                   trControl=xgb.control,
                   tuneGrid = xgb.grid,
                   weights = w
                   )

  xgb_finalModel[[iter]] <- xgb_fit$finalModel 
  #xgb_fit$bestTune
  #xgb_fit$results
  #xgb_fit$modelInfo
  
  
  
  #test
  p <- 1-predict(xgb_finalModel[[iter]], test_xdt2, type='response')
  prob[[iter]] <- p
  
  auc_pred <- ROCR::prediction(p, test_ydt)
  auc_pref <- ROCR::performance(auc_pred, "tpr", "fpr")
  auc <- ROCR::performance(auc_pred, "auc")
  dic.auc[iter] <- unlist(auc@y.values)
  
  tb[[iter]] <- table(test_ydt, round(p), useNA = "always")
  
  
  cat(iter,'-th finished! \n')
  iter <- iter+1
}




#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
#Test result

result <- unlist(sapply(tb, performance))
res <- matrix(result, 5, 50, 
              dimnames = list(c("sensitivity","specificity", "FDR", "Accuracy", "F1.score"), 
                              c(1:50)))
res[3,10] <- 0
apply(res,1,mean)
apply(res,1,sd)/sqrt(50)

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

for(i in 1:50){
  for(j in 1:length(thresh)){
    cutoff <- thresh[j]
    p <- as.numeric(ifelse(prob[[i]]>=cutoff, 1, 0))
    tb <- tb_fn(test_ydt,p)
    SEN[[i]][j] <- tb[2,2]/sum(tb[2,])
    SPE[[i]][j] <- tb[1,1]/sum(tb[1,])
    if(sum(tb[,2])!=0) FDR[[i]][j] <-  tb[1,2]/(tb[2,2]+tb[1,2])
    else FDR[[i]][j] <- 0
  }
  cat(i,'-')
}

SEN_xgb <- do.call(rbind, SEN)
SPE_xgb <- do.call(rbind, SPE)
FDR_xgb <- do.call(rbind, FDR)

SEN_xgb <- apply(SEN_xgb, 2, mean)
SPE_xgb <- apply(SPE_xgb, 2, mean)
FDR_xgb <- apply(FDR_xgb, 2, mean)



#1)roc curve
ROC_df_xgb <- data.frame(false_positive_rate=(1-SPE_xgb), true_positive_rate=SEN_xgb)

plot(x=ROC_df_xgb$false_positive_rate, ROC_df_xgb$true_positive_rate, xlim=c(0,1), ylim=c(0,1), type="l", lty=1, 
     xlab="False Positive Rate", ylab="True Positive Rate")

#2)fdr curve
plot(x=thresh,y=FDR_xgb , xlim=c(0,1), ylim=c(-0.5,1), type="l", lty=1, 
     xlab="Threshold", ylab="False Discovery Rate")

out_xgb <- data.frame(ROC_df_xgb, FDR_xgb=FDR_xgb)
write.csv(out_xgb,"D://dyu2017//auc_fdr//out_xgb.csv", row.names = F)


#----------------------------------------------------------------------------
#Cut value (youden index, sen_fdr)

best_thr_id <- function(x) apply(x, 1, which.max)

SEN_xgb <- do.call(rbind, SEN)
SPE_xgb <- do.call(rbind, SPE)
FDR_xgb <- do.call(rbind, FDR)


youden_xgb <- SEN_xgb + SPE_xgb
sen_fdr_xgb <- SEN_xgb*(1-FDR_xgb)


youden_id_xgb <- thresh[best_thr_id(youden_xgb)]
sen_fdr_id_xgb <- thresh[best_thr_id(sen_fdr_xgb)]


tb_xgb1 <- list(); tb_xgb2 <- list()

for(i in 1:50){
  
  cutoff_xgb1 <- youden_id_xgb[i]
  cutoff_xgb2 <- sen_fdr_id_xgb[i]
  
  p1 <- as.numeric(ifelse(prob[[i]]>=cutoff_xgb1, 1, 0))
  p2 <- as.numeric(ifelse(prob[[i]]>=cutoff_xgb2, 1, 0))
  tb_xgb1[[i]] <- tb_fn(test_ydt, p1) #youden 
  tb_xgb2[[i]] <- tb_fn(test_ydt, p2) #sen(1-fdr) 
  
  cat(i,'-')
  
}



#youden index --> tb_r1, tb_l1
xgb_result1 <- unlist(sapply(tb_xgb1, performance))

xgb_res1 <- matrix(xgb_result1, 5, 50, 
                  dimnames = list(c("sensitivity","specificity", "FDR", "Accuracy", "F1.score"), 
                                  c(1:50)))

apply(xgb_res1, 1, mean)
apply(xgb_res1, 1, sd)/sqrt(50)


#SEN(1-FDR) --> tb_r2, tb_l2
xgb_result2 <- unlist(sapply(tb_xgb2, performance))

xgb_res2 <- matrix(xgb_result2, 5, 50, 
                  dimnames = list(c("sensitivity","specificity", "FDR", "Accuracy", "F1.score"), 
                                  c(1:50)))

apply(xgb_res2, 1, mean)
apply(xgb_res2, 1, sd)/sqrt(50)


#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
#save(list=ls(),file='decomp_XGB_result')
load(file='decomp_XGB_result')