########################
##Occlusion Experiment##
########################

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

load(file='two_dim_data.rdata')

#Packages
library(keras)
library(kerasR)
library(reticulate)
library(tensorflow)
library(dplyr)
library(ggplot2)
library(gplots)
library(lattice)
library(gridExtra)
library(RColorBrewer)
#-----------------------------------------------------------------------------------
#initial settings for using keras
#use_condaenv("tensorflow", required = TRUE) 
use_python("C://Users//PC2//Anaconda3//envs//tensorflow//python")
keras_init()
keras_available() #keras 사용가능한지 확인
py_config()



#-----------------------------------------------------------------------------------
#minmax_scaler <- function(A){
  maxA <- max(A)
  minA <- min(A)
  
  out <- (A-minA)/(maxA-minA)
  
  return(out)
}

bindata_ <- lapply(bindata, t)
newdata <- lapply(bindata_, as.vector)
#newdata <- lapply(newdata, sqrt)
#newdata <- lapply(newdata, minmax_scaler)
newdt <- do.call(rbind, newdata)


#-----------------------------------------------------------------------------------
#model build
{
  keras_init()
  mod <- keras_model_sequential()
  mod %>%
    layer_conv_2d(filter=8, kernel_size=c(3,3), padding="same", input_shape=c(324,324,1)) %>%
    layer_activation_leaky_relu() %>%
    layer_conv_2d(filter=8, kernel_size=c(3,3)) %>%
    layer_activation_leaky_relu() %>%
    layer_max_pooling_2d(pool_size=c(2,2)) %>%
    layer_dropout(0.25) %>%
    layer_batch_normalization() %>%
    
    layer_conv_2d(filter=16, kernel_size=c(3,3),padding="same") %>% 
    layer_activation_leaky_relu() %>%  
    layer_conv_2d(filter=16, kernel_size=c(3,3) ) %>%  
    layer_activation_leaky_relu() %>%  
    layer_max_pooling_2d(pool_size=c(2,2)) %>%  
    layer_dropout(0.25) %>%
    layer_batch_normalization() %>%
    
    layer_conv_2d(filter=32, kernel_size=c(3,3),padding="same") %>% 
    layer_activation_leaky_relu() %>%  
    layer_conv_2d(filter=32, kernel_size=c(3,3) ) %>%  
    layer_activation_leaky_relu() %>%  
    layer_max_pooling_2d(pool_size=c(2,2)) %>%  
    layer_dropout(0.25) %>%
    layer_batch_normalization() %>%
    
    #flatten the input  
    layer_flatten() %>%  
    layer_dense(256) %>%  
    layer_activation_leaky_relu() %>%
    layer_batch_normalization() %>%
    
    #output layer-10 classes-2 units  
    layer_dense(2) %>%  
    
    #applying softmax nonlinear activation function to the output layer #to calculate cross-entropy  
    layer_activation("softmax") 
  
  adam <- optimizer_adam(lr=0.001, decay=1e-6)
}

#full data is used to training dataset
train_x <- newdt
train_y <- ifelse(resp!=4, 0,1)

keras_train_x <- array_reshape(x=train_x, dim=list(96,324,324,1))
keras_train_y <- to_categorical(train_y, num_classes = 2)

mod %>%
  compile(loss="categorical_crossentropy",
          optimizer=adam,
          metrics = "accuracy")

EarlyStopping()

mod %>% fit(keras_train_x, keras_train_y , batch_size=12,
            epochs=50, validation_split=0.2,
            shuffle=T
)



#-----------------------------------------------------------------------------------
#Occlusion Experiment

resp2 <- ifelse(resp=="4", 1, 0)

input_shape <- c(324,324)
occluding_size <- 18
occluding_stride <- 6
occluding_pixel <- 0

vec <- c()
outlier_prop <- rep(list(), 96)
occluding_probs <- rep(list(vec), 96)
mask_set <- seq(1, input_shape[1]-(occluding_size-1), occluding_stride)
rotate <- function(x) t(apply(x, 2, rev)) #for levelplot
bluered  <- function(n) colorpanel(n, 'blue','white','red')


#-----------------------------------------------------------------------------------
#iteration

for(iter in 1:96){
  
  test_x <- newdt[iter,]
  test_y <- resp2[iter]

  keras_test_x <- array_reshape(x=test_x, dim=list(1,324,324,1))
  keras_test_y <- to_categorical(test_y, num_classes = 2)
  

  #Occlusion Experiment
  k <- 1
  
  for(i in mask_set){
    
    for(j in mask_set){
      
      #j: horizon moving window index
      #i: vertical moving window index 
      
      test_x2 <- matrix(test_x, input_shape, byrow=T)
      test_x2[i:(i+17), j:(j+17)] <- occluding_pixel
      test_x2 <- t(test_x2)
      test_x_vec <- as.vector(test_x2)
      
      occ_test_x <- array_reshape(x=test_x_vec, dim=list(1,324,324,1))
      
      pred <- mod %>% predict(occ_test_x)
      
      
      if(test_y==0){
        occluding_probs[[iter]][k] <- pred[1,2]
      }else{
        occluding_probs[[iter]][k] <- pred[1,2]
      }
      
      k <- k+1
      
    }
    
  }
  

  #mean and CI
  occ_p <- occluding_probs[[iter]]
  occ_p_avg <- mean(occ_p)
  occ_p_sd <- sd(occ_p)
  
  occ_p[occ_p<occ_p_avg-2*occ_p_sd] <- -1
  occ_p[occ_p>occ_p_avg+2*occ_p_sd] <- 1
  
  outlier_prop[[iter]] <- matrix(occ_p, 52, 52, byrow=T)
  
  
  #saving probability heatmaps
   # prob_mat <- matrix(occluding_probs[[iter]], 52, 52, byrow=T)
   # 
   # 
   # path <- "C://Users//PC2//Desktop//연구//occlusion_heatmap//"
   # 
   # if(test_y==0){
   #   f_name <- paste0(path, "non_tissue//", substr(name[iter], 1, (nchar(name[iter])-4)), ".png" )
   # }else{
   #   f_name <- paste0(path, "tissue//", substr(name[iter], 1, (nchar(name[iter])-4)), ".png" )
   # }
   # 
   # test_dt <- t(sqrt(matrix(newdt[iter,],324,324)))
   # 
   # 
   # raw <- t(apply(rotate(test_dt),1,rev))
   # occ <- t(apply(rotate(prob_mat),1,rev))
   # 
   # png(f_name, width=1280, height=780)
   # p1 <- levelplot(raw, xlab="Root-Intensity of M/Z", ylab="RT_bin",ylim=rev(range(1,324)),
   #           main=paste0("Test data Heatmap - ",substr(name[iter], 1, (nchar(name[iter])-4))),
   #           col.regions =  bluered(100), at=seq(0,200,2))
   # 
   # p2 <- levelplot(occ, xlab="Window of M/Z", ylab="Window of RT_bin", ylim=rev(range(1,52)),
   #                 main=paste0("Occlusion Heatmap - ",substr(name[iter], 1, (nchar(name[iter])-4))),
   #                 col.regions = bluered(1000), at=seq(0,1,0.001))
   # 
   # grid.arrange(p1, p2, ncol=2)
   # dev.off()
  
   cat(iter, "-image saved!\n")

  #k_clear_session()

}




#-----------------------------------------------------------------------------------

beef_id <- which(resp=="1")
fish_id <- which(resp=="2")
pork_id <- which(resp=="3")
tissue_id <- which(resp=="4")
n_tis_id <- which(resp!="4")

beef <- list()
fish <- list()
pork <- list()
tissue <- list()
n_tis <- list()

beef <- outlier_prop[beef_id]
fish <- outlier_prop[fish_id]
pork <- outlier_prop[pork_id]
tissue <- outlier_prop[tissue_id]
n_tissue <- outlier_prop[n_tis_id]

#샘플별로 합치고
beef_v <- lapply(beef, as.vector)
fish_v <- lapply(fish, as.vector)
pork_v <- lapply(pork, as.vector)
tissue_v <- lapply(tissue, as.vector)
n_tissue_v <- lapply(n_tissue, as.vector)
  
BEEF <- do.call(rbind, beef_v)
FISH <- do.call(rbind, fish_v)
PORK <- do.call(rbind, pork_v)
TISSUE <- do.call(rbind, tissue_v)
NONTISSUE <- do.call(rbind, n_tissue_v)

#계산
count_up <- function(v){
  up <- sum(v==1)
  return(up)
}
count_lw <- function(v){
  lw <- sum(v==-1)
  return(lw)
}


beef_ups <- matrix(apply(BEEF, 2, count_up)/24, 52, 52)
beef_lws <- matrix(apply(BEEF, 2, count_lw)/24, 52, 52)
fish_ups <- matrix(apply(FISH, 2, count_up)/24, 52, 52)
fish_lws <- matrix(apply(FISH, 2, count_lw)/24, 52, 52)
pork_ups <- matrix(apply(PORK, 2, count_up)/24, 52, 52)
pork_lws <- matrix(apply(PORK, 2, count_lw)/24, 52, 52)
tissue_ups <- matrix(apply(TISSUE, 2, count_up)/24, 52, 52)
tissue_lws <- matrix(apply(TISSUE, 2, count_lw)/24, 52, 52)
n_tissue_ups <- matrix(apply(NONTISSUE, 2, count_up)/72, 52, 52)
n_tissue_lws <- matrix(apply(NONTISSUE, 2, count_lw)/72, 52, 52)

lev_dt <- function(m) t(apply(rotate(m),1,rev))

beef_ups <- lev_dt(beef_ups)
beef_lws <- lev_dt(beef_lws)
fish_ups <- lev_dt(fish_ups)
fish_lws <- lev_dt(fish_lws)
pork_ups <- lev_dt(pork_ups)
pork_lws <- lev_dt(pork_lws)
tissue_ups <- lev_dt(tissue_ups)
tissue_lws <- lev_dt(tissue_lws)
n_tissue_ups <- lev_dt(n_tissue_ups)
n_tissue_lws <- lev_dt(n_tissue_lws)


b1 <- levelplot(beef_lws, xlab="Window of M/Z", ylab="Window of RT_bin", ylim=rev(range(1,52)),
                main=paste0("Occlusion Heatmap - BEEF_upper"), 
                col.regions = bluered(100), at=seq(0,1,0.01))
b2 <- levelplot(beef_ups, xlab="Window of M/Z", ylab="Window of RT_bin", ylim=rev(range(1,52)),
                main=paste0("Occlusion Heatmap - BEEF_lower"), 
                col.regions = bluered(100), at=seq(0,1,0.05))
grid.arrange(b1, b2, ncol=2)


f1 <- levelplot(fish_lws, xlab="Window of M/Z", ylab="Window of RT_bin", ylim=rev(range(1,52)),
                main=paste0("Occlusion Heatmap - FISH_upper"), 
                col.regions = bluered(100), at=seq(0,1,0.01))
f2 <- levelplot(fish_ups, xlab="Window of M/Z", ylab="Window of RT_bin", ylim=rev(range(1,52)),
                main=paste0("Occlusion Heatmap - FISH_lower"), 
                col.regions = bluered(100), at=seq(0,1,0.01))
grid.arrange(f1, f2, ncol=2)


p1 <- levelplot(pork_lws, xlab="Window of M/Z", ylab="Window of RT_bin", ylim=rev(range(1,52)),
                main=paste0("Occlusion Heatmap - PORK_upper"), 
                col.regions = bluered(100), at=seq(0,1,0.01))
p2 <- levelplot(pork_ups, xlab="Window of M/Z", ylab="Window of RT_bin", ylim=rev(range(1,52)),
                main=paste0("Occlusion Heatmap - PORK_lower"), 
                col.regions = bluered(100), at=seq(0,1,0.01))
grid.arrange(p1, p2, ncol=2)


t1 <- levelplot(tissue_ups, xlab="Window of M/Z", ylab="Window of RT_bin", ylim=rev(range(1,52)),
                main=paste0("Occlusion Heatmap - TISSUE_upper"), 
                col.regions = bluered(100), at=seq(0,1,0.01))
t2 <- levelplot(tissue_lws, xlab="Window of M/Z", ylab="Window of RT_bin", ylim=rev(range(1,52)),
                main=paste0("Occlusion Heatmap - TISSUE_lower"), 
                col.regions = bluered(100), at=seq(0,1,0.01))
grid.arrange(t1, t2, ncol=2)


nt1 <- levelplot(n_tissue_lws, xlab="Window of M/Z", ylab="Window of RT_bin", ylim=rev(range(1,52)),
                main=paste0("Occlusion Heatmap - NON TISSUE_upper"), 
                col.regions = bluered(100), at=seq(0,1,0.01))
nt2 <- levelplot(n_tissue_ups, xlab="Window of M/Z", ylab="Window of RT_bin", ylim=rev(range(1,52)),
                main=paste0("Occlusion Heatmap - NON TISSUE_lower"), 
                col.regions = bluered(100), at=seq(0,1,0.01))
grid.arrange(nt1, nt2, ncol=2)


######################################################################################################
#save(list=ls(),file='decomp_occ_sensitivity')
load(file='decomp_occ_sensitivity')

######################################################################################################


bk <- c(seq(0,200,by=2))
mycols <- c(colorRampPalette(colors = c("blue","red"))(length(bk)-1))

bluered  <- function(n) colorpanel(n, 'blue','white','red')

levelplot(raw, xlab="Root-Intensity of M/Z", ylab="RT_bin",ylim=rev(range(1,324)),
          main=paste0("Test data Heatmap - ",substr(name[iter], 1, (nchar(name[iter])-4))),
          col.regions = bluered(100), at=seq(0,200,2))

levelplot(occ, xlab="Window of M/Z", ylab="Window of RT_bin", ylim=rev(range(1,52)),
          main=paste0("Occlusion Heatmap - ",substr(name[iter], 1, (nchar(name[iter])-4))),
          col.regions = bluered(1000), at=seq(0,1,0.001))



p1 <- levelplot(raw, xlab="Root-Intensity of M/Z", ylab="RT_bin",ylim=rev(range(1,324)),
                main=paste0("Test data Heatmap - ",substr(name[iter], 1, (nchar(name[iter])-4))),
                col.regions = gray(0:100/100), at=seq(0,200,2))

p2 <- levelplot(occ, xlab="Window of M/Z", ylab="Window of RT_bin", ylim=rev(range(1,52)),
                main=paste0("Occlusion Heatmap - ",substr(name[iter], 1, (nchar(name[iter])-4))),
                col.regions = gray(0:100/100), at=seq(0,1,0.05))
