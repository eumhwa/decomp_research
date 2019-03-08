setwd("D:/dyu2017/decomposition_csv")
getwd()
library(dplyr)
######################################
###Decomposition###
# B : Beef
# F : Fish
# P : Pork
# Tissue
# Blank
######################################

#Data load
data <- list()
name <- dir()[1:114]

blank <-substr(name,3,5)=="BLA"
sum(blank)
name2 <- cbind(name, blank)
name <- name2[name2[,2]=='FALSE',1]
nn <- length(name)


symb = substr(name,3,3)
resp = rep(0,nn)
resp[symb=='B'] = 1 # Beef
resp[symb=='F'] = 2 # Fish
resp[symb=='P'] = 3 # Pork
resp[symb=='T'] = 4 # Tissue (Human)


#data load using list
rdata <-list()
for (i in 1:nn){
  rdata[[i]] <- read.table(name[i], sep=";", head=T, stringsAsFactors = T)  
}
length(rdata) #96?? ?????Ͱ? list?? ?????ִ?. 
#a <- (rdata[[2]])
#fix(a)     # ?Ѱ??? row?? ????�� ??, ?ش? retention time ?? ???ҵ?(Xi)???? ???ڷ?


save(list=ls(),file='get_data_decomp.rdata')

#######################################################################################

rm(list=ls())

load(file='get_data_decomp.rdata')



# variables counting

len_x <-c()
len_y <-c()
for(i in 1:length(rdata)){
  len_x[i] <- ncol(rdata[[i]])-3
  len_y[i] <- nrow(rdata[[i]])
}
len_x  #2??° ???????? ???? ũ?? ?ٸ???? 

par(mfrow=c(1,2))
hist(len_x, xlab = "The number of variables(M/Z)", main = "Mass to charge ratio")
legend("center", legend=name[which(len_x<350)], col = 'red', cex=0.85)
hist(len_y, xlab = "The number of observed RT", main = "Retention Time")
legend("center", legend=name[which(len_y!=9643)], col = 'red', cex=0.85)

name[which(len_x<350)] 
name[which(len_y!=9643)]




# define intensities (rowsum of x-variables)
data <-list()
intensities <- c()
for(i in 1:length(rdata)){
  intensities <- apply(rdata[[i]][,c(-1,-2,-3)], 1, sum)  # exclude time variable
  data[[i]] <- cbind(rdata[[i]][,2], intensities)
  colnames(data[[i]]) <- c("r_time", "intensities")
}



#ret_time range
range_mat = matrix(0,nn,2)
for(i in 1:nn){
  range_mat[i,] = range(data[[i]][,1]) #96?? sheet?? r_time??��
}
range_mat
tt_range <- range(range_mat) #??ü r_time?? ??��
tt_range                     #(0.1909667, 41.9930333)



#intensity range
intens_mat <- matrix(0,nn,2)
for(i in 1:nn){
  intens_mat[i,] = range(data[[i]][,2]) #96?? sheet?? r_time??��
}
intens_mat
tt_intens <- range(intens_mat)
tt_intens                      #(1370, 30637313)


#modifying data
for(i in 1:nn){
  data[[i]] <- data[[i]][1:9309,]  
}



# build full dataframe 
tmp_dt <- t(data[[1]])
nc <- length(tmp_dt[1,])
nn <- 96
df <- matrix(0, nn, nc)

for(i in 1:nn){
  
  if(i==4)
  {
    dt <- data[[i]]
    dt <- t(dt)[2,]
    nas <- rep(NA, times=(nc-length(dt)))
    dt <- append(dt, nas)
    df[i,] <- dt  
  }
  
  else
  {
    dt <- data[[i]]
    dt <- t(dt)[2,]
    df[i,] <- dt
  }
  
}
fix(tmp_dt)


#data Plot
seq(1,nn,by=4)
for(i in seq(1,nn,by=4))
{
  st = 3
  if(substr(name[i],3,3)=="T") {
    ed = 10
  } else {
    st = 3
    ed = 5
  } 
  png(file=paste0('fig/max_',substr(name[i],st,ed),'_peaks_norm_avg.png'),
      width = 1024, height = 1280,type="cairo-png",pointsize=25)
  par(mfrow=c(4,1))
  plot(data[[i]][,1],data[[i]][,2],main=substr(name[i],st,ed),xlab='Retention time',
       ylab='Intensity (1st)',col=1,type='h'
       ,xlim=tt_range,ylim=tt_intens)
  plot(data[[i+1]][,1],data[[i+1]][,2],main=substr(name[i+1],st,ed),xlab='Retention time',
       ylab='Intensity (2nd)',col=1,type='h'
       ,xlim=tt_range,ylim=tt_intens)
  plot(data[[i+2]][,1],data[[i+2]][,2],main=substr(name[i+2],st,ed),xlab='Retention time',
       ylab='Intensity (3rd)',col=1,type='h'
       ,xlim=tt_range,ylim=tt_intens)
  plot(data[[i+3]][,1],data[[i+3]][,2],main=substr(name[i+3],st,ed),xlab='Retention time',
       ylab='Intensity (4th)',col=1,type='h'
       ,xlim=tt_range,ylim=tt_intens)
  dev.off()
}

fix(data)


save(list=ls(),file='before_binning')

#######################################################################################

rm(list=ls())

load(file='before_binning')

#######################################################################################


## Binning Procedure ##

#####parameter ����########
bin = 0.01       #initial bin size
le = 0.19        #tt_range
ri = 40.54

length(seq(le, ri,by=bin))   

bret <- seq(le,ri,by=bin)    #binning
nb_rt <- length(bret)        #ret time ?? ?????? binsize?? binning



#binning process

nn=96 
bre_dat <- matrix(0,nn,nb_rt)
bcount <- matrix(0,nn,nb_rt)
for(i in 1:nn)
{
  temp  = data[[i]] 
  for(k in 1:nb_rt)    #bin ????
  {
    indx = which(temp[,1]>= bret[k]-bin/2 & temp[,1] < bret[k]+bin/2)  #???? ?? sum
    #cat(indx)
    if(length(indx)==0) next
    if(length(indx)==1)
      bre_dat[i,k] = temp[indx,2]          #log(temp[indx,2]+1)
    if(length(indx)>1)
    {
      bcount[i,k] = bcount[i,k] + 1
      bre_dat[i,k] = sum(temp[indx,2])     #log(sum(temp[indx,2])+1)
    }
    
  }
}

dim(bre_dat)


#plotting with initial binning (using first data)
bin_forplot <- apply(bre_dat, 2, function(x) sum(x)/length(x))
length(bin_forplot)
plot(x=bret,
     y=bin_forplot, 
     main = "Initial binning",
     xlab = 'Retention Time', 
     ylab = 'Intensitiy',
     xlim = c(0.19,5),
     ylim = c(0,8000000),  
     type='l',
     lwd=2,
     col="darkblue",
     cex.lab=1.5,
     cex.axis=1.5)

bret2 <- seq(le,ri,length=324)
abline(v =bret2, col = 'black', lty=2, lwd=1)




## Optimized Bucketing process ##

rat <- 0.05  #optimize binning ��?? bin?? ?̵? ??�� ??��
rat/bin
ncol(bre_dat)/(rat/bin) 


source("D://dyu2017//EH_4class_data//opt_binning.R")
fit <- opt_binning(bre_dat,
                   nbins=18*18, 
                   slack=0.95)

optbin_dat <- fit$dat

alg_dat <- log(fit$dat+1) 

max_intens <- apply(optbin_dat, 1, max)
alg_dat2 <- (optbin_dat/max_intens)*100


bd_t <- bret[fit$index]
nbd_t <- length(bd_t)
tt_area <- range(alg_dat)
tt_area

dim(alg_dat)



#plotting with opt_binning (using first data)
bret2 <-seq(le, ri, length.out = 18*18)
optbin_forplot <- apply(fit$dat, 2, function(x) sum(x)/length(x))
length(optbin_forplot)
length(bret2)

plot(x=bret2, 
     y=optbin_forplot,  
     main = "Average spectrum of Opt_binning", 
     xlab = 'Retention Time', 
     ylab = 'Intensitiy',
     xlim = c(0,5),
     ylim=c(0,80000000), 
     type='l',
     lwd=2,
     col="darkblue",
     cex.lab=1.5,
     cex.axis=1.5)
abline(v =bd_t, col = 'black', lty=2, lwd=1)
abline(v =bret2, col = 'black', lty=2, lwd=1)

#ggplot version
ggdt <- data.frame(Retention_Time=bret2, Intensity=optbin_forplot)
ggplot(data=ggdt, aes(x=Retention_Time, y=Intensity)) + 
  geom_line(color="blue") +
  xlim(0,5) +
  ylim(0,70000000) +
  theme_classic() +
  geom_vline(xintercept = bd_t[1:39], color="1")




tt <- apply(df,2, max)
tt2 <- apply(df,2, min)
fix(df)
range(tt)
range(tt2)

init_rt <- data[[1]]
init_rt <- init_rt[,1]

#plot with no binning
for(i in 1:nrow(bre_dat)){
  plot(x=init_rt, 
       df[i,], 
       type='l', col=i, 
       xlim=c(0,5), 
       ylim =c(0,55000000),
       main="All spectrums without binning",
       xlab = 'Retention time', 
       ylab = 'Intensitiy',
       cex.lab=1.5,
       cex.axis=1.5)
  
  par(new=TRUE)
}


#init binning plot 
for(i in 1:nrow(bre_dat)){
  plot(x=bret, 
       bre_dat[i,], 
       type='l', col=i, 
       xlim=c(0,5), 
       ylim =c(0,55000000),
       main="All spectrums after Init_binning",
       xlab = 'Retention Time', 
       ylab = 'Intensitiy',
       cex.lab=1.5,
       cex.axis=1.5)
  
  par(new=TRUE)
}



#opt binning plot
for(i in 1:nrow(fit$dat)){
  plot(x=bret2, 
       fit$dat[i,],
       type='l', 
       col=i, 
       xlim=c(0,5), 
       ylim =c(0,550000000),
       main="All spectrums after Opt_binning",
       xlab = 'Retention Time', 
       ylab = 'Intensitiy',
       cex.lab=1.5,
       cex.axis=1.5)
  
  par(new=TRUE)
}


range(apply(fit$dat,1,range))



save(list=ls(),file='after_binning3')

##############################################################

rm(list=ls())

load(file='after_binning3')

