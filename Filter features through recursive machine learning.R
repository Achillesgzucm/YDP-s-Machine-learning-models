# reference package
library(tidyverse)
library(glmnet)
source('msvmRFE.R') # comes with the folder
library(VennDiagram)
library(sigFeature)
library(e1071)
library(caret)
library(randomForest)
library(limma)

Input file
inputFile="GSE87474.txt" # expression matrix
C="C" # Normal group sample name

Read the input file
rt=read.table(inputFile, header=T, sep="\t", check.names=F)
rt=as.matrix(rt)
rownames(rt)=rt[,1]
exp=rt[,2:ncol(rt)]
dimnames=list(rownames(exp),colnames(exp))
data=matrix(as.numeric(as.matrix(exp)),nrow=nrow(exp),dimnames=dimnames)
data=avereps(data)
data=t(data)
data=data[,read.table("disease.txt", header=F, sep="\t", check.names=F)[,1]]
# Control group is placed first
# Group
sample=read.table("sample.txt",sep="\t",header=F,check.names=F,row.names = 1)
data=data[rownames(sample),]
afcon=sum(sample[,1]==C)
group=c(rep("0",afcon),rep("1",nrow(data)-afcon))
group=as.matrix(as.numeric(group))
rownames(group)=rownames(data)
colnames(group)="Type"
input <- as.data.frame(cbind(group,data))
input$Type=as.factor(input$Type)
# Ten fold cross validation is adopted
svmRFE(input, k = 10, halve.above = 100) # Split data and assign random numbers
nfold = 10
nrows = nrow(input)
folds = rep(1:nfold, len=nrows)[sample(nrows)]
folds = lapply(1:nfold, function(x) which(folds == x))
results = lapply(folds, svmRFE.wrap, input, k=10, halve.above=100) # Feature selection, error reported
top.features = WriteFeatures(results, input, save=F) # View the main variables
head(top.features)
# Save the features found by SVM-REF to a file and AvgRank sorts them by the average ranking of 10 folds
write.csv(top.features,"feature_svm.csv")

The running time depends mainly on the number of variables you select, so you should not choose too many variables on a typical computer
# The first 40 variables were selected for SVM model construction, and then the results that had been running were imported. The results could not run at 1:40
featsweep = lapply(1:20, FeatSweep.wrap, results, input)

# Drawing
no.info = min(prop.table(table(input[,1])))
errors = sapply(featsweep, function(x) ifelse(is.null(x), NA, x$error))

#dev.new(width=4, height=4, bg='white')
pdf("svm-error.pdf",width = 5,height = 5)
PlotErrors(errors, no.info=no.info) # View the error rate
dev.off()

pdf("svm-accuracy.pdf",width = 5,height = 5)
Plotaccuracy(1-errors,no.info=no.info) # Check accuracy
dev.off()

# The position of the red circle in the figure is the lowest error rate
which.min(errors)

library(survival)
library(glmnet)
library(ggplot2)
library(ggsci)
library(patchwork)
library(limma)

inputFile="GSE87474.txt" # Input file
C="C" # Normal group sample name

Read the input file
rt=read.table(inputFile, header=T, sep="\t", check.names=F)
rt=as.matrix(rt)
rownames(rt)=rt[,1]
exp=rt[,2:ncol(rt)]
dimnames=list(rownames(exp),colnames(exp))
data=matrix(as.numeric(as.matrix(exp)),nrow=nrow(exp),dimnames=dimnames)
data=avereps(data)
data=t(data)
data=data[,read.table("DIFF.txt", header=F, sep="\t", check.names=F)[,1]]
sample=read.table("sample.txt",sep="\t",header=F,check.names=F,row.names = 1)
data=data[rownames(sample),]
x=as.matrix(data)

# Control group is placed first
afcon=sum(sample[,1]==C)
group=c(rep("0",afcon),rep("1",nrow(data)-afcon))
group=as.matrix(group)
rownames(group)=rownames(data)
y=as.matrix(group[,1])

set.seed(123)
cvfit = cv.glmnet(x, y,family = "binomial", nlambda=100, alpha=1,nfolds =10) # Here alpha=1 is LASSO regression, if equal to 0 is ridge regression, 10 times cross validation
# Parameter family specifies the type of regression model:
#family="gaussian" is applicable to one-dimensional continuous dependent variables.
#family="mgaussian" It is applied to multi-dimensional continuous dependent variables (multivariate)
#family="poisson" for non-negative degree dependent variables (count)
#family="binomial" for binary discrete dependent variables
# family = "multinomial" applies to multiple discrete dependent variable (category)
Here the outcome index is 2 categorical variable, so binomial is used

fit <- glmnet(x,y,family = "binomial")
cvfit$lambda.min

Extract information and predict risks
coef <- coef(fit, s = cvfit$lambda.min)
index <- which(coef ! = 0)
actCoef <- coef[index]
lassoGene=row.names(coef)[index]
geneCoef=cbind(Gene=lassoGene, Coef=actCoef)
write.table(geneCoef, file="geneCoef.xls", sep="\t", quote=F, row.names=F)
write.table(file="lassoset.txt",lassoGene,sep="\t",quote=F,col.names=F,row.names=F) # File name

# # # # # # #  simple drawing # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
pdf("lasso.pdf",height = 5,width = 7)
layout(matrix(c(1,1,2,2), 2,2, byrow = F)) # Two rows and two columns, Figure 1 occupies the first two grids, Figure 2 occupies the last two grids, arranged in columns
#pdf("lambda.pdf")
plot(fit,xvar = 'lambda')
#dev.off()
#pdf("cvfit.pdf")
plot(cvfit)
abline(v=log(c(cvfit$lambda.min,cvfit$lambda.1se)),lty="dashed")
#dev.off()
dev.off()
# # drawing 

# reference package
library(randomForest)
library(limma)
library(ggpubr)
set.seed(123)

inputFile="GSE87474.txt" # Input file
C="C" # Normal group sample name

Read the input file
rt=read.table(inputFile, header=T, sep="\t", check.names=F)
rt=as.matrix(rt)
rownames(rt)=rt[,1]
exp=rt[,2:ncol(rt)]
dimnames=list(rownames(exp),colnames(exp))
data=matrix(as.numeric(as.matrix(exp)),nrow=nrow(exp),dimnames=dimnames)
data=avereps(data)
data=t(data)
data=data[,read.table("disease.txt", header=F, sep="\t", check.names=F)[,1]]
sample=read.table("sample.txt",sep="\t",header=F,check.names=F,row.names = 1)
data=data[rownames(sample),]
colnames(data)=gsub("-", "afaf", colnames(data))
# Control group is placed first
afcon=sum(sample[,1]==C)
group=c(rep("con",afcon),rep("treat",nrow(data)-afcon))

# Random forest tree
rf=randomForest(as.factor(group)~., data=data, ntree=500)
pdf(file="forest.pdf", width=6, height=6)
plot(rf, main="Random forest", lwd=2)
dev.off()

Find the point with the least error
optionTrees=which.min(rf$err.rate[,1])
optionTrees
rf2=randomForest(as.factor(group)~., data=data, ntree=optionTrees)

# Check the importance of genes
# Map the importance of genes
importance=importance(x=rf2)
importance=as.data.frame(importance)
importance$size=gsub("-", "afaf", importance$size)
importance$size=rownames(importance)
Importance of = importance [, c (2, 1)]
names(importance)=c("Gene","importance")
# Show the importance of the top 20 genes
af=importance[order(importance$importance,decreasing = T),]
af=af[1:20,]
p=ggdotchart(af, x = "Gene", y = "importance",
color = "importance", # Custom color palette
sorting = "descending",                       # Sort value in descending order
add = "segments",                             # Add segments from y = 0 to dots
add.params = list(color = "lightgray", size = 2), # Change segment color and size
dot.size = 6,                        # Add mpg values as dot labels
font.label = list(color = "white", size = 9,
vjust = 0.5), # Adjust label parameters
ggtheme = theme_bw()         ,               # ggplot2 theme
rotate=TRUE)# Flip the axes
p1=p+ geom_hline(yintercept = 0, linetype = 2, color = "lightgray")+
Gradient_color (the palette = c (ggsci: : pal_npg () (2) [2], ggsci: : pal_npg () (2) [1])) + # color
grids()
# Save picture
pdf(file="importance.pdf", width=6, height=6)
print(p1)
dev.off()
Pick disease signature genes
rfGenes=importance[order(importance[,"importance"], decreasing = TRUE),]
write.table(rfGenes, file="rfGenes.xls", sep="\t", quote=F, col.names=T, row.names=F)

