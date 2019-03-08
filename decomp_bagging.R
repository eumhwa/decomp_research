####################
##glmnet + bagging##
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

grid_lam = c(seq(0.001,0.01,by=0.001), seq(0.01,2,by=0.01)) #glmnet lambda
grid_al = seq(0,1,by=0.1) 

dic.lam <-matrix(0,11,50)
dic.al  <-matrix(0,11,50)
dic.acc <- c()
dic.auc <- c()
idx_mat <- matrix(0,24,50)

m <- matrix(0,24,11)
prob <- rep(list(m),50)

bhs <- matrix(0, 11,325)
beta_list <- rep(list(bhs),50)

iter<-1

est_cl_ftn  <- function(x) return(names(which.max(table(x))))
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
  
  
  ##build 11 models for bagging
  for(i in 1:11){
    
    set.seed(1000*iter+i)
    #3개 class에서 6개씩 tissue에서 6개 뺀 xdata & label
    std_dat2 <- std_dat[-lab_idx,]
    resp2    <- as.numeric(resp[-lab_idx])
    
    #36x18 binary class data
    idx123 <- sample(which(resp2!=4), 36, replace=T) 
    idx4   <- which(resp2==4)
    idx    <- append(idx123,idx4)
  
    std_dat3 <- std_dat2[idx,]
    resp3    <- as.numeric(resp2[idx])
    resp3[which(resp3!=4)] <- 0
    resp3[which(resp3==4)] <- 1
  
  
    cv_mat = matrix(0,length(grid_lam),length(grid_al))
    colnames(cv_mat) = grid_al
    rownames(cv_mat) = grid_lam[length(grid_lam):1] 
  
    
    #estimate best paramter for 11 models
    for(k in 1:length(grid_al))
    {
      alpha = grid_al[k]
      cvfit=cv.glmnet(std_dat3,resp3,family='binomial',alpha=alpha,lambda=grid_lam,
                      type.measure="deviance",nfold=nrow(std_dat3),standardize=F)               #nfold=nn => loocv
      cv_mat[,k] = cvfit$cvm                                                        #cv-error
      cat("(",i,"-", k,")th parameter searching iteration \n")
    }
  
    
    #select lambda & alpha 
    lam <- as.numeric(names(which.min(apply(cv_mat,1,min))))
    al  <- grid_al[which.min(apply(cv_mat,2,min))]
    dic.lam[i,iter] <- lam
    dic.al[i,iter]  <- al
    
    
    ##model fitting
    test <- glmnet(std_dat3,resp3,family='binomial', alpha=al, lambda=lam, standardize=F)
 
    b0 <- test$a0
    bs <- test$beta
    bs[which(bs==0)] <- 0
    beta_list[[iter]][i,] <- c(as.numeric(b0), as.vector(bs))
    
    #test data
    new_dat <- std_dat[lab_idx,]
    new_lab <- as.numeric(resp[lab_idx])
    new_lab[which(new_lab!=4)] <- 0
    new_lab[which(new_lab==4)] <- 1
  
    
    #calculate probabilties
    for(kk in 1:24)
    {
      new_data <- t(as.matrix(new_dat[kk,]))
      prob[[iter]][kk,i] <- as.numeric(exp(b0+new_data%*%bs)/(1+exp(b0+new_data%*%bs)))
    }
    
    
    
  }
  
  #voting
  prd_result     <- as.numeric(apply(ifelse(prob[[iter]]>=0.5,1,0), 1, est_cl_ftn)) #threshold : 0.5
  dic.acc[iter]  <- mean(new_lab==prd_result)
  
  # auc_pred <- ROCR::prediction(prd_result, new_lab)
  # auc_pref <- ROCR::performance(auc_pred, "tpr", "fpr")
  # auc <- ROCR::performance(auc_pred, "auc")
  # dic.auc[iter] <- unlist(auc@y.values)
  
  cat("********** End of ",iter,"th iteration*********\n")
  iter <- iter+1
  
}


#----------------------------------------------------------------------------
#----------------------------------------------------------------------------


#dic.auc
dic.acc
dic.lam
dic.al
r1 <- as.numeric(apply(ifelse(prob[[1]]>=0.75,1,0), 1, est_cl_ftn))
r2 <- as.numeric(apply(ifelse(prob[[2]]>=0.75,1,0), 1, est_cl_ftn))
r10 <- as.numeric(apply(ifelse(prob[[10]]>=0.75,1,0), 1, est_cl_ftn))



#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
#Test result

tb.list <- rep(list(0),50)
ans <- c(rep(0,18), rep(1,6))
for(i in 1:50){
  p <- as.numeric(apply(ifelse(prob[[i]]>=0.5,1,0), 1, est_cl_ftn))
  tb.list[[i]] <- table(ans,p)
}
p <- as.numeric(apply(round(prob[[5]],0), 1, est_cl_ftn))
tb.list

result <- unlist(sapply(tb.list, performance))
mean(dic.acc)

res <- matrix(result, 5, 50, 
              dimnames = list(c("sensitivity","specificity", "FDR", "Accuracy", "F1.score"), 
                              c(1:50)))

apply(res,1,mean)
apply(res, 1, sd)/sqrt(50)
#sd(res[1,])/sqrt(50)



#----------------------------------------------------------------------------
#test AUC

tb_fn <- function(y, p){
  m <- matrix(0,2,2)
  
  m[1,1] <- sum(y==0 & p==0)
  m[1,2] <- sum(y==0 & p==1)
  m[2,1] <- sum(y==1 & p==0)
  m[2,2] <- sum(y==1 & p==1)
  
  return(m)
}

#auc1 - area under roc curve
thresh <- seq(0.0001, 0.9999, 0.0001)
v <- rep(9999,length(thresh))
SEN <- rep(list(v),50)
SPE <- rep(list(v),50)

for(i in 1:50){
  for(j in 1:length(thresh)){
    cutoff <- thresh[j]
    p <- as.numeric(apply(ifelse(prob[[i]]>=cutoff, 1, 0), 1, est_cl_ftn))
    tb <- tb_fn(ans,p)
    
    SEN[[i]][j] <- tb[2,2]/sum(tb[2,])
    SPE[[i]][j] <- tb[1,1]/sum(tb[1,])
    if(sum(tb[,2])!=0) FDR[[i]][j] <-  tb[1,2]/(tb[2,2]+tb[1,2])
    else FDR[[i]][j] <- 0
  }
  cat(i,'-')
}


AUC_list <- c()
dic_auroc1 <- c()
for(i in 1:50){
  se <- SEN[[i]]
  sp <- SPE[[i]]
  
  tmp <-data.frame(TPR=se, FPR=1-sp)   
  tmp2 <- tmp[order(tmp$FPR, -tmp$TPR),]
  AUC_list <- tmp2[!duplicated(tmp2$FPR),]
  
  S <- c()
  for(kk in 1:(nrow(AUC_list)-1)){
    if(AUC_list$TPR[kk]!=AUC_list$TPR[kk+1]){
      S[kk] <- (AUC_list$TPR[kk]+AUC_list$TPR[kk+1])*(AUC_list$FPR[kk+1]-AUC_list$FPR[kk])/2
    }
    else{
      S[kk] <- (AUC_list$FPR[kk+1]-AUC_list$FPR[kk])*AUC_list$TPR[kk]
    }
    dic_auroc1[i] <- sum(S)
  }
    
}

mean(dic_auroc1)
sd(dic_auroc1)/sqrt(50)



#auc2 - prediction average
dic_auroc2 <-c()
for(i in 1:50){
  p <- prob[[i]]
  p2 <- apply(p,1,mean)
  
  auc_pred <- ROCR::prediction(p2, new_lab)
  auc_pref <- ROCR::performance(auc_pred, "tpr", "fpr")
  auc <- ROCR::performance(auc_pred, "auc")
  
  dic_auroc2[i] <- unlist(auc@y.values)
  
}
mean(dic_auroc2)
sd(dic_auroc2)/sqrt(50)


#auc3 - bagging probability
prob2 <- lapply(prob, round)
cac_prob <- function(x) apply(x,1,sum)/11

prob3 <- lapply(prob2, cac_prob)
dic_auroc3 <-c()
for(i in 1:50){
  pp <- prob3[[i]]
  
  auc_pred <- ROCR::prediction(pp, new_lab)
  auc_pref <- ROCR::performance(auc_pred, "tpr", "fpr")
  auc <- ROCR::performance(auc_pred, "auc")
  
  dic_auroc3[i] <- unlist(auc@y.values)
}

mean(dic_auroc3)
sd(dic_auroc3)/sqrt(50)


#auc4 - coefficients average
avg_fn <- function(x) apply(x, 2, mean)
beta_list2 <- lapply(beta_list, avg_fn)
beta_list2 <- do.call(rbind, beta_list2)

prob4 <- matrix(0,24,50)
dic_auroc4 <- c()

#calculate probabilties
for(ii in 1:50){
  
  idx <- idx_mat[,ii]
  new_dat <- std_dat[idx,]
  new_lab <- as.numeric(resp[idx])
  new_lab[which(new_lab!=4)] <- 0
  new_lab[which(new_lab==4)] <- 1
  
  beta_fin <- beta_list2[ii,]
  for(kk in 1:24)
  {
    new_data <- t(as.matrix(new_dat[kk,]))
    prob4[kk,ii] <- as.numeric(exp(beta_fin[1]+new_data%*%beta_fin[-1])/(1+exp(beta_fin[1]+new_data%*%beta_fin[-1])))
  }
  
  p <- prob4[,ii]
  auc_pred <- ROCR::prediction(p, new_lab)
  auc_pref <- ROCR::performance(auc_pred, "tpr", "fpr")
  auc <- ROCR::performance(auc_pred, "auc")
  
  dic_auroc4[ii] <- unlist(auc@y.values)
}

mean(dic_auroc4)
sd(dic_auroc4)/sqrt(50)
plot(auc_pref)
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
SEN2 <- rep(list(v),50)
SPE2 <- rep(list(v),50)
FDR2 <- rep(list(v),50)

for(i in 1:50){
  for(j in 1:length(thresh)){
    cutoff <- thresh[j]
    p <- as.numeric(ifelse(prob3[[i]]>=cutoff, 1, 0))
    tb <- tb_fn(ans,p)
    SEN2[[i]][j] <- tb[2,2]/sum(tb[2,])
    SPE2[[i]][j] <- tb[1,1]/sum(tb[1,])
    if(sum(tb[,2])!=0) FDR2[[i]][j] <-  tb[1,2]/(tb[2,2]+tb[1,2])
    else FDR2[[i]][j] <- 0
  }
  cat(i,'-')
}

SEN_b <- do.call(rbind, SEN2)
SPE_b <- do.call(rbind, SPE2)
FDR_b <- do.call(rbind, FDR2)

SEN_b <- apply(SEN_b, 2, mean)
SPE_b <- apply(SPE_b, 2, mean)
FDR_b <- apply(FDR_b, 2, mean)



#1)roc curve
ROC_df_bag <- data.frame(false_positive_rate=(1-SPE_b), true_positive_rate=SEN_b)

plot(x=ROC_df_bag$false_positive_rate, ROC_df_bag$true_positive_rate, xlim=c(0,1), ylim=c(0,1), type="l", lty=1, 
     xlab="False Positive Rate", ylab="True Positive Rate")

#2)fdr curve
plot(x=thresh,y=FDR_b , xlim=c(0,1), ylim=c(-0.5,1), type="l", lty=1, 
     xlab="Threshold", ylab="False Discovery Rate")

out_bagging <- data.frame(ROC_df_bag, FDR_b=FDR_b)
write.csv(out_bagging,"D://dyu2017//auc_fdr//out_bagging.csv", row.names = F)




#----------------------------------------------------------------------------
#Cut value (youden index, sen_fdr)

best_thr_id <- function(x) apply(x, 1, which.max)

SEN_b <- do.call(rbind, SEN2)
SPE_b <- do.call(rbind, SPE2)
FDR_b <- do.call(rbind, FDR2)

youden_b <- SEN_b + SPE_b
sen_fdr_b <- SEN_b*(1-FDR_b)



youden_id_b <- thresh[best_thr_id(youden_b)]
sen_fdr_id_b <- thresh[best_thr_id(sen_fdr_b)]




tb_b1 <- list(); tb_b2 <- list()

for(i in 1:50){
  
  cutoff_b1 <- youden_id_b[i]
  cutoff_b2 <- sen_fdr_id_b[i]
  
  p1 <-as.numeric(apply(ifelse(prob[[i]]>=cutoff_b1,1,0), 1, est_cl_ftn))
  p2 <- as.numeric(apply(ifelse(prob[[i]]>=cutoff_b2,1,0), 1, est_cl_ftn))
  
  tb_b1[[i]] <- tb_fn(new_lab, p1) #youden 
  tb_b2[[i]] <- tb_fn(new_lab, p2) #sen(1-fdr) 
  
  cat(i,'-')
  
}


#youden index --> tb_r1, tb_l1
bag_result1 <- unlist(sapply(tb_b1, performance))

bag_res1 <- matrix(bag_result1, 5, 50, 
                     dimnames = list(c("sensitivity","specificity", "FDR", "Accuracy", "F1.score"), 
                                     c(1:50)))

apply(bag_res1, 1, mean)
apply(bag_res1, 1, sd)/sqrt(50)




#SEN(1-FDR) --> tb_r2, tb_l2
bag_result2 <- unlist(sapply(tb_b2, performance))

bag_res2 <- matrix(bag_result2, 5, 50, 
                   dimnames = list(c("sensitivity","specificity", "FDR", "Accuracy", "F1.score"), 
                                   c(1:50)))

apply(bag_res2, 1, mean)
apply(bag_res2, 1, sd)/sqrt(50)



###########################################################################################################
#save(list=ls(),file='decomp_bagging_result')
load(file='decomp_bagging_result')
