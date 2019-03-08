####################
######glmnet########
####################

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
library(glmnet)
library(ROCR)
library(dplyr)
#-----------------------------------------------------------------------------------



####################################################################################
rm(list=ls())

load(file='after_binning2')
####################################################################################



##Preparing data
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
#Initializing the vecs & mats

grid_lam =  seq(0.01,2,by=0.01) #glmnet lambda
dic.acc <- c()
dic.auc <- matrix(0,2,50)
idx_mat <- matrix(0,24,50)
ridge_pred <- matrix(0,24,50)
lasso_pred <- matrix(0,24,50)
ridge_tb <- list()
lasso_tb <- list()

iter<-1

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
#glmnet + bagging

while(iter<51){   
  
  set.seed(iter)
  
  lab_idx1 <- sample(which(resp=="1"), 6)
  lab_idx2 <- sample(which(resp=="2"), 6)
  lab_idx3 <- sample(which(resp=="3"), 6)
  lab_idx4 <- sample(which(resp=="4"), 6)
  lab_idx  <- c(lab_idx1,lab_idx2,lab_idx3,lab_idx4)
  
  idx_mat[,iter]<- lab_idx

  #3개 class에서 6개씩 tissue에서 6개 뺀 xdata & label
  std_dat2 <- std_dat[-lab_idx,]
  resp2    <- as.numeric(resp[-lab_idx])
  
  resp2[which(resp2!=4)] <- 0
  resp2[which(resp2==4)] <- 1
  
  
  cv_mat = matrix(0,length(grid_lam),2)
  rownames(cv_mat) = grid_lam[length(grid_lam):1] 
  
  
  #LOOCV for tuning lambda
  ridge_cvfit <- cv.glmnet(std_dat2, resp2, family='binomial', alpha=0, lambda=grid_lam,
                           type.measure="deviance", nfold=nrow(std_dat2), standardize=F, grouped=F)       
  
  lasso_cvfit <- cv.glmnet(std_dat2, resp2, family='binomial', alpha=1, lambda=grid_lam,
                           type.measure="deviance", nfold=nrow(std_dat2), standardize=F, grouped=F)      
  
  cv_mat[,1] = ridge_cvfit$cvm   
  cv_mat[,2] = lasso_cvfit$cvm   
  
  
  
  #select lambda 
  ridge_lam <- as.numeric(names(which.min(cv_mat[,1])))
  lasso_lam <- as.numeric(names(which.min(cv_mat[,2])))
  
  
  ##model fitting
  ridge_test <- glmnet(std_dat2, resp2, family='binomial', alpha=0, lambda=ridge_lam, standardize=F)
  lasso_test <- glmnet(std_dat2, resp2, family='binomial', alpha=1, lambda=lasso_lam, standardize=F)
  
  
  #test data
  new_dat <- std_dat[lab_idx,]
  new_lab <- as.numeric(resp[lab_idx])
  new_lab[which(new_lab!=4)] <- 0
  new_lab[which(new_lab==4)] <- 1
  
  
  ridge_p <- predict(ridge_test, newx=new_dat, type="response")
  lasso_p <- predict(lasso_test, newx=new_dat, type="response")
  
  ridge_pred[,iter] <- ridge_p
  lasso_pred[,iter] <- lasso_p
  
  ridge_tb[[iter]] <- table(new_lab, round(ridge_p), useNA="always")
  lasso_tb[[iter]] <- table(new_lab, round(lasso_p), useNA="always")
  
  
  #AUC
  ridge_auc_pred <- ROCR::prediction(ridge_p, new_lab)
  ridge_auc_pref <- ROCR::performance(ridge_auc_pred, "tpr", "fpr")
  ridge_auc <- ROCR::performance(ridge_auc_pred, "auc")
  dic.auc[1,iter] <- unlist(ridge_auc@y.values)
  
  
  lasso_auc_pred <- ROCR::prediction(lasso_p, new_lab)
  lasso_auc_pref <- ROCR::performance(lasso_auc_pred, "tpr", "fpr")
  lasso_auc <- ROCR::performance(lasso_auc_pred, "auc")
  dic.auc[2,iter] <- unlist(lasso_auc@y.values)
  
  
  cat("********** End of ",iter,"th iteration*********\n")
  iter <- iter+1
  
}



#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
#Test result


ridge_result <- unlist(sapply(ridge_tb, performance))
lasso_result <- unlist(sapply(lasso_tb, performance))

ridge_res <- matrix(ridge_result, 5, 50, 
                    dimnames = list(c("sensitivity","specificity", "FDR", "Accuracy", "F1.score"), 
                                    c(1:50)))
lasso_res <- matrix(lasso_result, 5, 50, 
                    dimnames = list(c("sensitivity","specificity", "FDR", "Accuracy", "F1.score"), 
                                    c(1:50)))
lasso_res[3,5] <- 0
lasso_res[3,19] <- 0

apply(ridge_res,1,mean)
apply(ridge_res, 1, sd)/sqrt(50)
#sd(res[1,])/sqrt(50)

apply(lasso_res,1,mean)
apply(lasso_res, 1, sd)/sqrt(50)
#sd(res[1,])/sqrt(50)


#test AUC
apply(dic.auc,1,mean)
apply(dic.auc,1,sd)/sqrt(50)

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
SEN_ridge <- rep(list(v),50)
SPE_ridge <- rep(list(v),50)
FDR_ridge <- rep(list(v),50)

SEN_lasso <- rep(list(v),50)
SPE_lasso <- rep(list(v),50)
FDR_lasso <- rep(list(v),50)


for(i in 1:50){
  
  for(j in 1:length(thresh)){
   
     cutoff <- thresh[j]
    
    p1 <- as.numeric(ifelse(ridge_pred[,i]>=cutoff, 1, 0))
    tb1 <- tb_fn(new_lab, p1)
    SEN_ridge[[i]][j] <- tb1[2,2]/sum(tb1[2,])
    SPE_ridge[[i]][j] <- tb1[1,1]/sum(tb1[1,])
    if(sum(tb1[,2])!=0) FDR_ridge[[i]][j] <-  tb1[1,2]/(tb1[2,2]+tb1[1,2])
    else FDR_ridge[[i]][j] <- 0
    
    p2 <- as.numeric(ifelse(lasso_pred[,i]>=cutoff, 1, 0))
    tb2 <- tb_fn(new_lab, p2)
    SEN_lasso[[i]][j] <- tb2[2,2]/sum(tb2[2,])
    SPE_lasso[[i]][j] <- tb2[1,1]/sum(tb2[1,])
    if(sum(tb2[,2])!=0) FDR_lasso[[i]][j] <-  tb2[1,2]/(tb2[2,2]+tb2[1,2])
    else FDR_lasso[[i]][j] <- 0
    
  }
  cat(i,'-')
  
}

SEN_r <- do.call(rbind, SEN_ridge)
SPE_r <- do.call(rbind, SPE_ridge)
FDR_r <- do.call(rbind, FDR_ridge)

SEN_l <- do.call(rbind, SEN_lasso)
SPE_l <- do.call(rbind, SPE_lasso)
FDR_l <- do.call(rbind, FDR_lasso)

SEN_r <- apply(SEN_r, 2, mean)
SPE_r <- apply(SPE_r, 2, mean)
FDR_r <- apply(FDR_r, 2, mean)

SEN_l <- apply(SEN_l, 2, mean)
SPE_l <- apply(SPE_l, 2, mean)
FDR_l <- apply(FDR_l, 2, mean)


#1)roc curve
ROC_df_ridge <- data.frame(false_positive_rate=(1-SPE_r), true_positive_rate=SEN_r)
ROC_df_lasso <- data.frame(false_positive_rate=(1-SPE_l), true_positive_rate=SEN_l)

plot(x=ROC_df_ridge$false_positive_rate, ROC_df_ridge$true_positive_rate, xlim=c(0,1), ylim=c(0,1), type="l", lty=1, 
     xlab="False Positive Rate", ylab="True Positive Rate")
lines(x=ROC_df_lasso$false_positive_rate, ROC_df_lasso$true_positive_rate, xlim=c(0,1), ylim=c(0,1), type="l", lty=2, 
     xlab="False Positive Rate", ylab="True Positive Rate")



#2)FDR curve

plot(x=thresh,y=FDR_r , xlim=c(0,1), ylim=c(-0.5,1), type="l", lty=1, 
     xlab="Threshold", ylab="False Discovery Rate")
lines(x=thresh,y=FDR_l , xlim=c(0,1), ylim=c(-0.5,1), type="l", lty=2, 
      xlab="Threshold", ylab="False Discovery Rate")


out_ridge <- data.frame(ROC_df_ridge, FDR_r=FDR_r)
out_lasso <- data.frame(ROC_df_lasso, FDR_l=FDR_l)

write.csv(out_ridge,"D://dyu2017//auc_fdr//out_ridge.csv", row.names = F)
write.csv(out_lasso,"D://dyu2017//auc_fdr//out_lasso.csv", row.names = F)



#----------------------------------------------------------------------------
#Cut value (youden index, sen_fdr)

best_thr_id <- function(x) apply(x, 1, which.max)

SEN_r <- do.call(rbind, SEN_ridge)
SPE_r <- do.call(rbind, SPE_ridge)
FDR_r <- do.call(rbind, FDR_ridge)

SEN_l <- do.call(rbind, SEN_lasso)
SPE_l <- do.call(rbind, SPE_lasso)
FDR_l <- do.call(rbind, FDR_lasso)


youden_r <- SEN_r + SPE_r
sen_fdr_r <- SEN_r*(1-FDR_r)

youden_l <- SEN_l + SPE_l
sen_fdr_l <- SEN_l*(1-FDR_l)


youden_id_r <- thresh[best_thr_id(youden_r)]
sen_fdr_id_r <- thresh[best_thr_id(sen_fdr_r)]

youden_id_l <- thresh[best_thr_id(youden_l)]
sen_fdr_id_l <- thresh[best_thr_id(sen_fdr_l)]



tb_r1 <- list(); tb_r2 <- list()
tb_l1 <- list(); tb_l2 <- list()

for(i in 1:50){
    
  cutoff_r1 <- youden_id_r[i]
  cutoff_r2 <- sen_fdr_id_r[i]
  
  p1 <- as.numeric(ifelse(ridge_pred[,i]>=cutoff_r1, 1, 0))
  p2 <- as.numeric(ifelse(ridge_pred[,i]>=cutoff_r2, 1, 0))
  tb_r1[[i]] <- tb_fn(new_lab, p1) #youden ridge
  tb_r2[[i]] <- tb_fn(new_lab, p2) #sen(1-fdr) ridge
  
  cutoff_l1 <- youden_id_l[i]
  cutoff_l2 <- sen_fdr_id_l[i]
  
  p3 <- as.numeric(ifelse(lasso_pred[,i]>=cutoff_l1, 1, 0))
  p4 <- as.numeric(ifelse(lasso_pred[,i]>=cutoff_l2, 1, 0))
  tb_l1[[i]] <- tb_fn(new_lab, p3)  #youden lasso
  tb_l2[[i]] <- tb_fn(new_lab, p4)  #sen(1-fdr) lasso

  cat(i,'-')
  
}



#youden index --> tb_r1, tb_l1
ridge_result1 <- unlist(sapply(tb_r1, performance))
lasso_result1 <- unlist(sapply(tb_l1, performance))

ridge_res1 <- matrix(ridge_result1, 5, 50, 
                    dimnames = list(c("sensitivity","specificity", "FDR", "Accuracy", "F1.score"), 
                                    c(1:50)))
lasso_res1 <- matrix(lasso_result1, 5, 50, 
                    dimnames = list(c("sensitivity","specificity", "FDR", "Accuracy", "F1.score"), 
                                    c(1:50)))


apply(ridge_res1, 1, mean)
apply(ridge_res1, 1, sd)/sqrt(50)

apply(lasso_res1, 1, mean)
apply(lasso_res1, 1, sd)/sqrt(50)



#SEN(1-FDR) --> tb_r2, tb_l2
ridge_result2 <- unlist(sapply(tb_r2, performance))
lasso_result2 <- unlist(sapply(tb_l2, performance))

ridge_res2 <- matrix(ridge_result2, 5, 50, 
                     dimnames = list(c("sensitivity","specificity", "FDR", "Accuracy", "F1.score"), 
                                     c(1:50)))
lasso_res2 <- matrix(lasso_result2, 5, 50, 
                     dimnames = list(c("sensitivity","specificity", "FDR", "Accuracy", "F1.score"), 
                                     c(1:50)))


apply(ridge_res2, 1, mean)
apply(ridge_res2, 1, sd)/sqrt(50)

apply(lasso_res2, 1, mean)
apply(lasso_res2, 1, sd)/sqrt(50)


###########################################################################################################
#save(list=ls(),file='decomp_glmnet_result')
load(file='decomp_glmnet_result')
