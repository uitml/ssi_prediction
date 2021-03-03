if("data.table" %in% rownames(installed.packages()) == FALSE) {install.packages("data.table")}
if("zoo" %in% rownames(installed.packages()) == FALSE) {install.packages("zoo")}

library(data.table)
library(zoo)
source("data_orig/functions.r")

test_names = c( "Hemoglobin", "Leukocytter", "Natrium", "CRP", "Kalium", "Albumin", "Kreatinin", "Trombocytter", "ALAT", "Bilirubin total", "ASAT", "Glukose", "Amylase", "ALP" )
feat1<-test_names
cutoff1<-c(2,2,1,2,1,2,1,1,1,1,1,3,1,1)
cutoff2<- c(14,10,15,14,13,14,18,16,14,14,13,15,15,15) 
endpoint<-rep(60,length(test_names))
inter_per_test<-data.table(feat1,cutoff1,cutoff2,endpoint)

age_vec<- c(50,65,80)

load_BT_ref_table<-"data_orig/Blood_tests_14_ref_values.csv"
##Preprocess
BT_ref_table<-read.csv(load_BT_ref_table,header=T)[,1:7]
tmp<-unlist(strsplit(as.character(BT_ref_table$Sex),","))
BT_ref_table<-BT_ref_table[rep(1:length(BT_ref_table[,1]),
                               sapply(strsplit(as.character(BT_ref_table$Sex),","),length)),]
BT_ref_table$Sex<-tmp
BT_ref_table$Sex<-factor(BT_ref_table$Sex)
BT_ref_table$TestType<-as.character(BT_ref_table$TestType)


################################################################################################
# Train Set

load_data_lab_res<-"data_orig/WoundInf_Train_Tests.tsv"
load_data_labels<-"data_orig/WoundInf_Train_Labels.tsv"

save_res<-"data/Train_Raw.csv"
save_labels<-"data/Train_Labels.csv"
save_Primoz<-"data/Train_Data_Primoz.tsv"

data_lab_res<-read.delim(load_data_lab_res,header=T)
data_lab_res$Date<-as.POSIXct(strptime(data_lab_res$Date, "%d/%m/%y %H:%M"))
data_lab_res<-data.table(data_lab_res)

data_lab_labels<-read.delim(load_data_labels,header=T)
data_lab_labels<-data.table(data_lab_labels)

data_lab_labels$t.IndexSurgery<-as.POSIXct(strptime(data_lab_labels$t.IndexSurgery, "%Y-%m-%d %H:%M"))
data_lab_labels$t.Infection<-as.POSIXct(strptime(data_lab_labels$t.Infection, "%Y-%m-%d %H:%M"))
data_lab_labels[,':='(Age=as.numeric(format(t.IndexSurgery,"%Y"))-YoB)]
data_lab_labels$AgeGroups<-data_lab_labels[,factor(findInterval(Age, age_vec),labels=c(
                            paste(c(0,age_vec[1:(length(age_vec)-1)]),"-",age_vec[1:(length(age_vec))],sep=""),
                            paste(age_vec[(length(age_vec))],"+",sep="")
                            ))]

data_lab_labels<-data_lab_labels[ order( PID ) ]
write.csv( data_lab_labels, save_labels )
print("Saved Train Labels")

data_lab_res1<-merge(data_lab_res,data_lab_labels[,.(t.IndexSurgery,PID)], by="PID", all.x=T)
data_lab_res1[,':='(DaysFromSurg=floor(as.numeric(difftime(t.IndexSurgery,Date,units="days"))))]
data_lab_res1$AbnVal<-1*grepl("./*",data_lab_res1[,Answer])
data_lab_res1<-merge(data_lab_res1,data_lab_labels[,.(PID,Sex,AgeGroups,Age)],by="PID",all.x=T)
data_lab_res1$AbnVal_kat<-data_lab_res1[,prim_func3(NumAnswer,Sex,Age,TestType),by=1:data_lab_res1[,.N]][,V1]
data_lab_res1$AbnVal_kat[data_lab_res1$AbnVal_kat==9999]<-NA
data_lab_res1$AbnVal_kat<-factor(data_lab_res1$AbnVal_kat)

data_lab_res2<-expand.grid(PID=unique(data_lab_res1$PID),DaysFromSurg=-60:0,TestType=test_names)
data_lab_res2<-data.table(data_lab_res2)
data_lab_res2<-merge(data_lab_res2,data_lab_res1[,.(PID,TestType,NumAnswer,DaysFromSurg)], 
                     by=c("PID","TestType","DaysFromSurg"),all.x=T)

data_lab_res2<-data_lab_res2[ order( PID ) ]
write.csv( data_lab_res2, save_res )
print("Saved Train Raw")

#Primoz Features
tmp1<-data_lab_res2[DaysFromSurg==-60 & is.na(NumAnswer),.(PID,TestType)]
tmp2<-data_lab_res1[TestType %in% test_names & DaysFromSurg < -60,
                         .(NumAnswer=NumAnswer[which.min(NumAnswer)]),by=.(PID,TestType)]
tmp1<-merge(tmp1,tmp2,by=c("PID","TestType"),all.x=T)
tmp3<-data_lab_res1[TestType %in% test_names ,.(NumAnswer=mean(NumAnswer)),by=.(PID,TestType)]
tmp1<-tmp3[tmp1, on=c('PID','TestType')][ is.na(i.NumAnswer), i.NumAnswer:= NumAnswer][,NumAnswer:= i.NumAnswer][,i.NumAnswer:=NULL][]
tmp4<-data_lab_res1[TestType %in% test_names ,.(NumAnswer=mean(NumAnswer)),by=.(TestType)]
tmp1<-tmp4[tmp1, on=c('TestType')][is.na(i.NumAnswer), i.NumAnswer:= NumAnswer][,NumAnswer:= i.NumAnswer][,i.NumAnswer:=NULL][]
tmp1<-merge(tmp1, data_lab_labels[,.(PID, Sex,Age)],by="PID")


data_lab_res3<-tmp1[data_lab_res2, on=c('PID','TestType')][is.na(i.NumAnswer) & 
         DaysFromSurg==-60,i.NumAnswer:= NumAnswer][,
         NumAnswer:= i.NumAnswer][,i.NumAnswer:=NULL][]
data_lab_res3[,NumAnswer:=na.locf(NumAnswer),by=.(PID,TestType)]
data_lab_res3$AbnVal_kat<-data_lab_res3[,prim_func3(NumAnswer,Sex,Age,TestType),by=1:data_lab_res3[,.N]][,V1]

dt_feat_i1<-data.table(PID=data_lab_res1[,unique(PID)])
dt_feat_i1<-slope_val_bt(data_lab_res3,inter_per_test,dt_feat_i1)
dt_feat_i1<-mean_val_bt(data_lab_res3,inter_per_test,dt_feat_i1)
dt_feat_i1<-nmbr_test_bt(data_lab_res1,inter_per_test,dt_feat_i1)
dt_feat_i1<-prop_nmbr_test_bt(data_lab_res1,inter_per_test,dt_feat_i1)
dt_feat_i1<-low_high_val_test_bt(data_lab_res1,inter_per_test,dt_feat_i1)

data_feat_i1<-merge(data_lab_labels[,.(PID, Sex,Age)],dt_feat_i1, by="PID")
# data_feat_i1$Sex<-as.numeric(data_feat_i1$Sex)-1 # Broken I don't  know why
data_feat_i1$Sex<-as.numeric(data_feat_i1$Sex=='M')

data_feat_i1<-data_feat_i1[ order( PID ) ]

write.table( data_feat_i1, save_Primoz, sep="\t" )
print("Saved Train Primoz")

################################################################################################
# Eval Set

load_data_test_lab_res<-"data_orig/WoundInf_Eval_Tests.tsv"
load_data_test_labels<-"data_orig/WoundInf_Eval_Label+.csv"

save_test_res<-"data/Eval_Raw.csv"
save_test_labels<-"data/Eval_Labels.csv"
save_test_Primoz<-"data/Eval_Data_Primoz.tsv"

data_test_lab_res<-read.delim(load_data_test_lab_res,header=T)
data_test_lab_res$Date<-as.POSIXct(strptime(data_test_lab_res$Date, "%d/%m/%y %H:%M"))
data_test_lab_res<-data.table(data_test_lab_res)

data_test_lab_labels<-read.csv(load_data_test_labels,header=T)
data_test_lab_labels<-data.table(data_test_lab_labels)

data_test_lab_labels[,Infected:=NULL]
#colnames(data_test_lab_labels)[2] <- "Infection"

data_test_lab_labels$t.IndexSurgery<-as.POSIXct(strptime(data_test_lab_labels$t.IndexSurgery, "%d.%m.%Y %H:%M"))
data_test_lab_labels$t.Infection<-as.POSIXct(strptime(data_test_lab_labels$t.Infection, "%Y-%m-%d %H:%M:%S"))
data_test_lab_labels[,':='(Age=as.numeric(format(t.IndexSurgery,"%Y"))-YoB)]
data_test_lab_labels$AgeGroups<-data_test_lab_labels[,factor(findInterval(Age, age_vec),labels=c(
                            paste(c(0,age_vec[1:(length(age_vec)-1)]),"-",age_vec[1:(length(age_vec))],sep=""),
                            paste(age_vec[(length(age_vec))],"+",sep="")
                            ))]
data_test_lab_labels<-data_test_lab_labels[ order( PID ) ]
write.csv( data_test_lab_labels, save_test_labels )
print("Saved Eval Labels")


data_test_lab_res1<-merge(data_test_lab_res,data_test_lab_labels[,.(t.IndexSurgery,PID)], by="PID", all.x=T)
data_test_lab_res1[,':='(DaysFromSurg=floor(as.numeric(difftime(t.IndexSurgery,Date,units="days"))))]
data_test_lab_res1$AbnVal<-1*grepl("./*",data_test_lab_res1[,Answer])
data_test_lab_res1<-merge(data_test_lab_res1,data_test_lab_labels[,.(PID,Sex,AgeGroups,Age)],by="PID",all.x=T)
data_test_lab_res1$AbnVal_kat<-data_test_lab_res1[,prim_func3(NumAnswer,Sex,Age,TestType),by=1:data_test_lab_res1[,.N]][,V1]
data_test_lab_res1$AbnVal_kat[data_test_lab_res1$AbnVal_kat==9999]<-NA
data_test_lab_res1$AbnVal_kat<-factor(data_test_lab_res1$AbnVal_kat)

                          
data_test_lab_res2<-expand.grid(PID=unique(data_test_lab_res1$PID),DaysFromSurg=-60:0,TestType=test_names)
data_test_lab_res2<-data.table(data_test_lab_res2)
data_test_lab_res2<-merge(data_test_lab_res2,data_test_lab_res1[,.(PID,TestType,NumAnswer,DaysFromSurg)], 
                     by=c("PID","TestType","DaysFromSurg"),all.x=T)

data_test_lab_res2<-data_test_lab_res2[ order( PID ) ]
write.csv( data_test_lab_res2, save_test_res )
print("Saved Eval Raw")


#Primoz Features
tmp1<-data_test_lab_res2[DaysFromSurg==-60 & is.na(NumAnswer),.(PID,TestType)]
tmp2<-data_test_lab_res1[TestType %in% test_names & DaysFromSurg < -60,
                         .(NumAnswer=NumAnswer[which.min(NumAnswer)]),by=.(PID,TestType)]
tmp1<-merge(tmp1,tmp2,by=c("PID","TestType"),all.x=T)
tmp3<-data_test_lab_res1[TestType %in% test_names ,.(NumAnswer=mean(NumAnswer)),by=.(PID,TestType)]
tmp1<-tmp3[tmp1, on=c('PID','TestType')][ is.na(i.NumAnswer), i.NumAnswer:= NumAnswer][,NumAnswer:= i.NumAnswer][,i.NumAnswer:=NULL][]
tmp4<-data_test_lab_res1[TestType %in% test_names ,.(NumAnswer=mean(NumAnswer)),by=.(TestType)]
tmp1<-tmp4[tmp1, on=c('TestType')][is.na(i.NumAnswer), i.NumAnswer:= NumAnswer][,NumAnswer:= i.NumAnswer][,i.NumAnswer:=NULL][]
tmp1<-merge(tmp1, data_test_lab_labels[,.(PID, Sex,Age)],by="PID")

data_test_lab_res3<-tmp1[data_test_lab_res2, on=c('PID','TestType')][is.na(i.NumAnswer) & 
         DaysFromSurg==-60,i.NumAnswer:= NumAnswer][,
         NumAnswer:= i.NumAnswer][,i.NumAnswer:=NULL][]
data_test_lab_res3[,NumAnswer:=na.locf(NumAnswer),by=.(PID,TestType)]
data_test_lab_res3$AbnVal_kat<-data_test_lab_res3[,prim_func3(NumAnswer,Sex,Age,TestType),
                                        by=1:data_test_lab_res3[,.N]][,V1]

dt_test_feat_i1<-data.table(PID=data_test_lab_res1[,unique(PID)])
dt_test_feat_i1<-slope_val_bt(data_test_lab_res3,inter_per_test,dt_test_feat_i1)
dt_test_feat_i1<-mean_val_bt(data_test_lab_res3,inter_per_test,dt_test_feat_i1)
dt_test_feat_i1<-nmbr_test_bt(data_test_lab_res1,inter_per_test,dt_test_feat_i1)
dt_test_feat_i1<-prop_nmbr_test_bt(data_test_lab_res1,inter_per_test,dt_test_feat_i1)
dt_test_feat_i1<-low_high_val_test_bt(data_test_lab_res1,inter_per_test,dt_test_feat_i1)

data_test_feat_i1<-merge(data_test_lab_labels[,.(PID, Sex,Age)],dt_test_feat_i1, by="PID")
# data_test_feat_i1$Sex<-as.numeric(data_test_feat_i1$Sex)-1 # Broken I don't  know why
data_test_feat_i1$Sex<-as.numeric(data_test_feat_i1$Sex=='M')

data_test_feat_i1<-data_test_feat_i1[ order( PID ) ]

write.table( data_test_feat_i1, save_test_Primoz, sep="\t" )
print("Saved Eval Primoz")


print("Done.")
