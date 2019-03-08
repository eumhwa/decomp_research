################
##RandomForest##
################

#----------------------------------------------------------------------------
#----------------------------------------------------------------------------

setwd("D:/dyu2017/decomposition_csv")

library(caret)
library(e1071)
library(randomForest)
library(ranger)
library(ROCR)

#----------------------------------------------------------------------------
#----------------------------------------------------------------------------

rm(list=ls())

load(file='after_binning2')

#----------------------------------------------------------------------------
#----------------------------------------------------------------------------

performance <- function(tb){
  sensitivity <- tb[2,2]/sum(tb[2,])
  specificity <- tb[1,1]/sum(tb[1,])
  FDR <- tb[1,2]/(tb[2,2]+tb[1,2])
  Accuracy <- (tb[1,1]+tb[2,2])/(tb[1,1]+tb[1,2]+tb[2,1]+tb[2,2])
  F1.score <- tb[2,2]/(2*tb[2,2]+tb[1,2]+tb[2,1]) 
  
  return(list(sensitivity=sensitivity, specificity=specificity, FDR=FDR, Accuracy=Accuracy, F1.score=F1.score))
}


##data 정리 
o_indx = c(rep(c(rep(T,4),rep(T,4)),3),
           rep(c(rep(T,4),rep(T,4)),3),
           rep(c(rep(T,4),rep(T,4)),3),
           rep(c(rep(T,4),rep(T,4)),3))


nn = sum(o_indx)

chk = apply(alg_dat!=0,2,sum)      
slt_dat = alg_dat[,chk>1]          
dim(slt_dat)
sum(chk>1)                         
std_dat = slt_dat[o_indx,]         
dim(std_dat)
resp = resp[o_indx]

#----------------------------------------------------------------------------
#----------------------------------------------------------------------------

#initialization
iter <- 1
tb <- list()
rf_fit <-list()
rf_finalModel.iter <- list()
pred_err <- c()
dic.auc <- c()
prob <- matrix(0, 24, 50)

#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
### 2class###
#data partitioning
while(iter<51){  
  
  set.seed(iter)
  
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
  train_ydt <- resp2[-idx]

  test_xdt <- std_dat[idx,]
  test_ydt <- resp2[idx]


  #training
  rf.metric <- "Accuracy"
  t_grid <- expand.grid(mtry = c(18, 50, 100,150), splitrule="gini", min.node.size=c(1,3,5,10))
  
  control <- trainControl(method="cv", 
                          number=10, 
                          search="grid",
                          classProbs = TRUE
                          #summaryFunction = twoClassSummary 
                          ) #classProbs, summaryFunction은 ROC를 metric으로 쓸때 필요한 옵션


  ### when using ROC for metric#############################
  levels <- unique(train_ydt) 
  train_ydt2 <- factor(train_ydt, labels=make.names(levels))

  levels <- unique(test_ydt) 
  test_ydt2 <- factor(test_ydt, labels=make.names(levels))
  ##########################################################



  #randomforest
  ntrees <- c(100,300,500)
  i <- 1
  for(ntree in ntrees){
    fit <- train(x = train_xdt, 
                 y = train_ydt2, 
                 num.trees=ntree,
                 method = "ranger", 
                 metric = "Accuracy", 
                 tuneGrid = t_grid,
                 trControl = control
    )
    
    rf_fit[[i]] <- fit
    rf_finalModel.iter[[i]] <- fit$finalModel
    i = i+1
    
  }
    


  for(j in 1:length(ntrees)){
    pred_err[j] <- rf_finalModel.iter[[j]]$prediction.error
  }

  best_idx <-which.min(pred_err)
  rf_finalModel <- rf_finalModel.iter[[best_idx]]
  
  params <- rf_finalModel
  
  df <- data.frame(y=train_ydt2, x=train_xdt)
  w <- ifelse(as.numeric(train_ydt)==0,1,1.5)
  ran <- ranger(y~.,data=df, num.trees=ntrees[best_idx], 
                mtry=as.numeric(params$tuneValue[1]), 
                min.node.size=as.numeric(params$tuneValue[3]), 
                case.weights = w, keep.inbag=TRUE, probability=TRUE)
  
  
  
  #test
  test_xdt <- data.frame(test_xdt)
  colnames(test_xdt) <- colnames(df)[-1]
  pred_ <- predict(ran, test_xdt, type='response')
  pred <- as.vector(pred_$predictions[,2])
  prob[,iter] <- pred
  
  auc_pred <- ROCR::prediction(pred, test_ydt)
  auc_pref <- ROCR::performance(auc_pred, "tpr", "fpr")
  auc <- ROCR::performance(auc_pred, "auc")
  dic.auc[iter] <- unlist(auc@y.values)
  
  tb[[iter]] <- table(test_ydt, round(pred), useNA = "always")

  cat(iter,"th model built \n")
  iter <- iter + 1
}




#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
#Test result

result <- unlist(sapply(tb, performance))

res <- matrix(result, 5, 50, 
              dimnames = list(c("sensitivity","specificity", "FDR", "Accuracy", "F1.score"), 
                              c(1:50)))
res[3,33] <- 0 #NaN

apply(res,1,mean)
apply(res,1,sd)/sqrt(50)

mean(dic.auc)
sd(dic.auc)/sqrt(50)

#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
#save(list=ls(),file='decomp_RF_result')
load(file='decomp_RF_result')



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
    p <- as.numeric(ifelse(prob[,i]>=cutoff, 1, 0))
    tb <- tb_fn(test_ydt,p)
    SEN[[i]][j] <- tb[2,2]/sum(tb[2,])
    SPE[[i]][j] <- tb[1,1]/sum(tb[1,])
    if(sum(tb[,2])!=0) FDR[[i]][j] <-  tb[1,2]/(tb[2,2]+tb[1,2])
    else FDR[[i]][j] <- 0
  }
  cat(i,'-')
}

SEN_rf <- do.call(rbind, SEN)
SPE_rf <- do.call(rbind, SPE)
FDR_rf <- do.call(rbind, FDR)

SEN_rf <- apply(SEN_rf, 2, mean)
SPE_rf <- apply(SPE_rf, 2, mean)
FDR_rf <- apply(FDR_rf, 2, mean)



#1)roc curve
ROC_df_rf <- data.frame(false_positive_rate=(1-SPE_rf), true_positive_rate=SEN_rf)

plot(x=ROC_df_rf$false_positive_rate, ROC_df_rf$true_positive_rate, xlim=c(0,1), ylim=c(0,1), type="l", lty=1, 
     xlab="False Positive Rate", ylab="True Positive Rate")

#2)fdr curve
plot(x=thresh,y=FDR_rf , xlim=c(0,1), ylim=c(-0.5,1), type="l", lty=1, 
     xlab="Threshold", ylab="False Discovery Rate")

out_rf <- data.frame(ROC_df_rf, FDR_rf=FDR_rf)
write.csv(out_rf,"D://dyu2017//auc_fdr//out_rf.csv", row.names = F)



#----------------------------------------------------------------------------
#----------------------------------------------------------------------------

#----------------------------------------------------------------------------
#Cut value (youden index, sen_fdr)

best_thr_id <- function(x) apply(x, 1, which.max)

SEN_rf <- do.call(rbind, SEN)
SPE_rf <- do.call(rbind, SPE)
FDR_rf <- do.call(rbind, FDR)



youden_rf <- SEN_rf + SPE_rf
sen_fdr_rf <- SEN_rf*(1-FDR_rf)


youden_id_rf <- thresh[best_thr_id(youden_rf)]
sen_fdr_id_rf <- thresh[best_thr_id(sen_fdr_rf)]


tb_rf1 <- list(); tb_rf2 <- list()

for(i in 1:50){
  
  cutoff_rf1 <- youden_id_rf[i]
  cutoff_rf2 <- sen_fdr_id_rf[i]
  
  p1 <- as.numeric(ifelse(prob[,i]>=cutoff_rf1, 1, 0))
  p2 <- as.numeric(ifelse(prob[,i]>=cutoff_rf2, 1, 0))
  tb_rf1[[i]] <- tb_fn(test_ydt, p1) #youden 
  tb_rf2[[i]] <- tb_fn(test_ydt, p2) #sen(1-fdr) 
  
  cat(i,'-')
  
}



#youden index --> tb_r1, tb_l1
rf_result1 <- unlist(sapply(tb_rf1, performance))

rf_res1 <- matrix(rf_result1, 5, 50, 
                     dimnames = list(c("sensitivity","specificity", "FDR", "Accuracy", "F1.score"), 
                                     c(1:50)))

apply(rf_res1, 1, mean)
apply(rf_res1, 1, sd)/sqrt(50)


#SEN(1-FDR) --> tb_r2, tb_l2
rf_result2 <- unlist(sapply(tb_rf2, performance))

rf_res2 <- matrix(rf_result2, 5, 50, 
                  dimnames = list(c("sensitivity","specificity", "FDR", "Accuracy", "F1.score"), 
                                  c(1:50)))

apply(rf_res2, 1, mean)
apply(rf_res2, 1, sd)/sqrt(50)

