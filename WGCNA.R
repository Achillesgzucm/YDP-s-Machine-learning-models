
# reference package
library(ggplot2)
library(limma)
library(pheatmap)
library(ggsci)
library(dplyr)
lapply(c('clusterProfiler','enrichplot','patchwork'), function(x) {library(x, character.only = T)})
library(org.Hs.eg.db)
library(patchwork)
library(WGCNA)
library(GSEABase)

Parameter setting
GSE="GSE87474" # Indicates the name of the matrix file without suffix
C="C" # Normal control group name
P="P" # Name of the disease group
Ccol="blue" # Heat Map Comment bar normal group color
Pcol="red" # Heat map comment bar Disease group color

rt=read.table(paste0(GSE,".txt"),sep="\t",header=T,check.names=F)
rt=as.matrix(rt)
rownames(rt)=rt[,1]
exp=rt[,2:ncol(rt)]
dimnames=list(rownames(exp),colnames(exp))
rt=matrix(as.numeric(as.matrix(exp)),nrow=nrow(exp),dimnames=dimnames)
rt=avereps(rt)

# # 1. Data preparation
# Group
sample=read.table("sample.txt",sep="\t",header=F,check.names=F,row.names = 1)
rt=rt[,rownames(sample)]
afcon=sum(sample[,1]==C)
# Determine if the raw data went to log
max(rt)
if(max(rt)>50) rt=log2(rt+1) #rt the maximum is greater than 30

# Corrects using normalizeBetweenArrays, which is assigned rt1
rt1=normalizeBetweenArrays(as.matrix(rt))

# Not standardized,mar adjust canvas range left top right, adjust yourself ha
cols=rainbow(ncol(rt)) ### # For 24 samples, set the color, the overall color is rainbow
pdf(file = "1.raw.pdf",width=5,height = 4)
par(cex = 0.7,mar=c(8,8,8))
If (ncol (rt) > 40) par (cex = 0.5, mar = c (8,8,8,8)) # # # set the font size
boxplot(rt,las=2,col =cols) ### Plot
dev.off()

# Standardization
cols=rainbow(ncol(rt1)) ### # For 24 samples, set the color, the overall color is rainbow
pdf(file = "1.nor.pdf",width=5,height = 4.5)
par(cex = 0.5,mar=c(8,8,8))
If (ncol (rt1) > 40) par (cex = 0.5, mar = c (8,8,8,8)) # # # set the font size
boxplot(rt1,las=2,col =cols) ### Plot
dev.off()

# Save normalized results
rt2=rbind(ID=colnames(rt1),rt1)
write.table(rt2,file=paste0("1.","norexp_",GSE,".txt"),sep="\t",quote=F,col.names = F)

# Keep the original result
rt3=rbind(ID=colnames(rt),rt)
write.table(rt3,file=paste0("1.","rawexp_",GSE,".txt"),sep="\t",quote=F,col.names = F)

# 2 . Difference analysis
data=rt1
#data=rt

conData=data[,as.vector(colnames(data)[1:afcon])]
aftreat=afcon+1
treatData=data[,as.vector(colnames(data)[aftreat:ncol(data)])]
rt=cbind(conData,treatData)
conNum=ncol(conData)
treatNum=ncol(treatData)

#limma differential Standard process
Type=c(rep("con",conNum),rep("treat",treatNum))
design <- model.matrix(~0+factor(Type))
colnames(design) <- c("con","treat")
fit <- lmFit(rt,design)
cont.matrix<-makeContrasts(treat-con,levels=design)
fit2 <- contrasts.fit(fit, cont.matrix)
fit2 <- eBayes(fit2)

Diff=topTable(fit2,adjust='fdr',number=length(rownames(data)))
Save the differential results of all the genes
DIFFOUT=rbind(id=colnames(Diff),Diff)
write.table(DIFFOUT,file=paste0("2.","DIFF_all.xls"),sep="\t",quote=F,col.names=F)
diffSig=Diff[with(Diff, (abs(logFC)> 1&ad.p.val < 0.05)),]
diffSigOut=rbind(id=colnames(diffSig),diffSig)
write.table(diffSigOut,file=paste0("2.","DIFF_less.xls"),sep="\t",quote=F,col.names=F)

# Show the top 30 genes with the most differences
Diff=Diff[order(as.numeric(as.vector(Diff$logFC))),]
diffGene=as.vector(rownames(Diff))
diffLength=length(diffGene)
afGene=c()
if(diffLength>(60)){
afGene=diffGene[c(1:30,(diffLength-30+1):diffLength)]
}else{
afGene=diffGene
}
afExp=rt[afGene,]
# Group tag
Type=c(rep(C,conNum),rep(P,treatNum))
names(Type)=colnames(rt)
Type=as.data.frame(Type)
# The comment color of the group tag
anncolor=list(Type=c(C=Ccol,P=Pcol))
names(anncolor[[1]])=c(C,P)

pdf(file=paste0("3.", "DIFF_heatmap.pdf"),height=7,width=6)
pheatmap(afExp, # heatmap data
annotation=Type, # group
color = colorRampPalette(c("blue","white","red"))(50), # heat map color
cluster_cols =F, # Do not add column clustering tree
show_colnames = F, # Displays column names
scale="row",
fontsize = 10,
fontsize_row=6,
fontsize_col=8,
annotation_colors=anncolor
)
dev.off()

# Volcano map variance standard setting
AdjP = 0.05
AflogFC = 0.5
Significant=ifelse((Diff$P.Value<adjP & abs(Diff$logFC)>aflogFC), ifelse(Diff$logFC>aflogFC,"Up","Down"), "Not")
# Start drawing
p = ggplot(Diff, aes(logFC, -log10(P.Value)))+
geom_point(aes(col=Significant),size=3)+
scale_color_manual(values=c(pal_npg()(2)[2], "#838B8B", pal_npg()(1)))+
labs(title = " ")+
theme(plot.title = element_text(size = 16, hjust = 0.5, face = "bold") +
geom_hline(aes(yintercept=-log10(adjP)), colour="gray", linetype="twodash",size=1)+
geom_vline(aes(xintercept=aflogFC), colour="gray", linetype="twodash",size=1)+
geom_vline(aes(xintercept=-aflogFC), colour="gray", linetype="twodash",size=1)
# View, no markup can be directly saved
p
Add gene point markers, according to, can be self-labeled according to the results of difference analysis
Point. Pvalue = 0.01
point.logFc=2
# Continue drawing
Diff$symbol=rownames(Diff)
PDF (paste0 (" 3 ", "DIFF_vol. PDF"), width = 6.5, height = 6)
p=p+theme_bw()
for_label <- Diff %>%
filter(abs(logFC) >point.logFc & P.Value< point.Pvalue )
p+geom_point(size = 1.5, shape = 1, data = for_label) +
ggrepel::geom_label_repel(
aes(label = symbol),
data = for_label,
color="black",
label.size =0.1
)
dev.off()

3.GSEA analysis
deg=Diff
logFC_t=0
Deg $g = ifelse (deg $P.V alue > 0.05, 'stable',
ifelse( deg$logFC > logFC_t,'UP',
ifelse( deg$logFC < -logFC_t,'DOWN','stable') )
)
table(deg$g)

deg$symbol=rownames(deg)
df <- bitr(unique(deg$symbol), fromType = "SYMBOL",
toType = c( "ENTREZID"),
OrgDb = org.Hs.eg.db)
DEG=deg
DEG=merge(DEG,df,by.y='SYMBOL',by.x='symbol')
data_all_sort <- DEG %>%
arrange(desc(logFC))

geneList = data_all_sort$logFC # Extract foldchange from largest to smallest
names(geneList) < -data_all_sort $ENTREZID # Add the corresponding ENTREZID to the foldchange extracted above
head(geneList)

Start GSEA enrichment analysis
kk2 < -gseKEGG (geneList = geneList,####fail to download??
organism = 'hsa',# 'hsa'
nPerm        = 10000,
minGSSize    = 10,
maxGSSize    = 200,
pvalueCutoff = 0.05,
pAdjustMethod = "none" )
install.packages("R.utils")
R.utils::setOption("clusterProfiler.download.method","auto")
kk2 < -gseKEGG (geneList = geneList,####fail to download??
organism = 'hsa',# 'hsa'
nPerm        = 10000,
minGSSize    = 10,
maxGSSize    = 200,
pvalueCutoff = 0.05,
pAdjustMethod = "none" )
# Save the GSEA result
GSEAOUT=as.data.frame(kk2@result)
write.table(GSEAOUT,file="4.GSEAOUT.xls",sep="\t",quote=F,col.names=T)


The first 5 and last 5 GSEA results are respectively taken after sorting
num=5
pdf(paste0("4.","down_GSEA.pdf"),width = 8,height = 8)
gseaplot2(kk2, geneSetID = rownames(kk2@result)[head(order(kk2@result$enrichmentScore),num)])
dev.off()
pdf(paste0("4.","up_GSEA.pdf"),width = 8,height = 8)
gseaplot2(kk2, geneSetID = rownames(kk2@result)[tail(order(kk2@result$enrichmentScore),num)])
dev.off()
After sorting, take the first 5 and show the last 5 together
num=5
pdf(paste0("4.","all_GSEA.pdf"),width = 10,height = 10)
gseaplot2(kk2,  geneSetID =  rownames(kk2@result)[c(head(order(kk2@result$enrichmentScore),num),tail(order(kk2@result$enrichmentScore),num))])
dev.off()


# Ridge Map, display 10, save yourself
library(stringr)
#kk2@result$Description=gsub("HALLMARK_","",kk2@result$Description)
pdf(paste0("4.","all_ridgeplot_GSEA.pdf"),width = 6,height = 20)
ridgeplot(kk2)
dev.off()
### Ready
afdir < -paste0 (getwd(),"/5.WGCNA") # The path must be in Chinese
dir.create(afdir)

traitData=sample
traitData[,2]=traitData[,1]
TraitData [1] = ifelse (traitData [1] = = C, 1, 0)
TraitData [2] = ifelse (traitData [2] = = P, 1, 0)
Modify the character name
colnames(traitData)=c(C,P)


############### Formal analysis
# The following setting is important, do not omit.
options(stringsAsFactors = FALSE)
#Read in the female liver data set
fpkm = read.table(paste0("1.rawexp_GSE87474.txt"),sep="\t",header=T,comment.char = "",check.names=F)#########file name  You can be changed##### The data file name. If the working path is not the actual data path, add a correct data path
rownames(fpkm)=fpkm[,1]
dim(fpkm)
names(fpkm)
datExpr0 = as.data.frame(t(fpkm[,-1]))
names(datExpr0) = fpkm[,1];
rownames(datExpr0) = names(fpkm[,-1])

datExpr0

##check missing value
gsg = goodSamplesGenes(datExpr0, verbose = 3)
gsg$allOK

if (! gsg$allOK)
{
# Optionally, print the gene and sample names that were removed:
if (sum(! gsg$goodGenes)>0)
printFlush(paste("Removing genes:", paste(names(datExpr0)[!gsg$goodGenes], collapse = ", ")))
if (sum(! gsg$goodSamples)>0)
printFlush(paste("Removing samples:", paste(rownames(datExpr0)[!gsg$goodSamples], collapse = ", ")))
# Remove the offending genes and samples from the data:
datExpr0 = datExpr0[gsg$goodSamples, gsg$goodGenes]
}

##filter
meanFPKM=0.5 ####the threshold can be changed. Meanfpkm =0.5 #### The threshold can be changed
n=nrow(datExpr0)
datExpr0[n+1,]=apply(datExpr0[c(1:nrow(datExpr0)),],2,mean)
datExpr0=datExpr0[1:n,datExpr0[n+1,] > meanFPKM]  # for meanFpkm in row n+1 and it must be above what you set--select  meanFpkm>opt$meanFpkm(by rp)


filtered_fpkm=t(datExpr0)
filtered_fpkm=data.frame(rownames(filtered_fpkm),filtered_fpkm)
names(filtered_fpkm)[1]="sample"
head(filtered_fpkm)
write.table(filtered_fpkm, file=paste0(afdir,"/FPKM_filter.xls"),row.names=F, col.names=T,quote=FALSE,sep="\t")

sampleTree = hclust(dist(datExpr0), method = "average")


pdf(file =paste0(afdir,"/1_sampleClustering.pdf"), width = 12, height = 9)
par(cex = 0.6)
par(mar = c(0,4,2,0))
plot(sampleTree, main =" Sample clustering to detect outliers", sub="", xlab="", cex.lab = 1.5,
cex.axis = 1.5, cex.main = 2)

### Plot a line to show the cut
##abline(h = 15, col = "red")## Whether to select Clipping
dev.off()


### Determine cluster under the line
##clust = cutreeStatic(sampleTree, cutHeight = 15, minSize = 10)
##table(clust)


### clust 1 contains the samples we want to keep.
##keepSamples = (clust==1)
##datExpr0 = datExpr0[keepSamples, ]


#### Load the trait data
#Loading clinical trait data
for (df in colnames(traitData)) {
traitData[,df]=traitData[,df]/max(traitData[,df])
print(sd(traitData[,df]))
}
max(traitData)
dim(traitData)
names(traitData)
# remove columns that hold information we do not need.
allTraits = traitData
dim(allTraits)
names(allTraits)

# Form a data frame analogous to expression data that will hold the clinical traits.
fpkmSamples = rownames(datExpr0)
traitSamples =rownames(allTraits)
traitRows = match(fpkmSamples, traitSamples)
datTraits = allTraits[traitRows,]
rownames(datTraits)
collectGarbage()

# Re-cluster samples
sampleTree2 = hclust(dist(datExpr0), method = "average")
# Convert traits to a color representation: white means low, red means high, grey means missing entry
traitColors = numbers2colors(datTraits, signed = FALSE)
# Plot the sample dendrogram and the colors underneath.

# sizeGrWindow (12, 14)
pdf(file=paste0(afdir,"/2_Sample dendrogram and trait heatmap.pdf"),width=12,height=11)
plotDendroAndColors(sampleTree2, traitColors,
groupLabels = names(datTraits),
main = "Sample dendrogram and trait heatmap")
dev.off()


enableWGCNAThreads()
# Choose a set of soft-thresholding powers
powers = c(1:30)

# Call the network topology analysis function
sft = pickSoftThreshold(datExpr0, powerVector = powers, verbose = 5)

# Plot the results:
#sizeGrWindow(9, 5)
pdf(file=paste0(afdir,"/3_Scale independence.pdf"),width=9,height=5)
par(mfrow = c(1,2))
cex1 = 0.9
# Scale-free topology fit index as a function of the soft-thresholding power
plot(sft$fitIndices[,1], -sign(sft$fitIndices[,3])*sft$fitIndices[,2],
     xlab="Soft Threshold (power)",ylab="Scale Free Topology Model Fit,signed R^2",type="n",
     main = paste("Scale independence"));
text(sft$fitIndices[,1], -sign(sft$fitIndices[,3])*sft$fitIndices[,2],
     labels=powers,cex=cex1,col="red");
# this line corresponds to using an R^2 cut-off of h
abline(h=0.9,col="red")
# Mean connectivity as a function of the soft-thresholding power
plot(sft$fitIndices[,1], sft$fitIndices[,5],
     xlab="Soft Threshold (power)",ylab="Mean Connectivity", type="n",
     main = paste("Mean connectivity"))
text(sft$fitIndices[,1], sft$fitIndices[,5], labels=powers, cex=cex1,col="red")
dev.off()


######chose the softPower
softPower =sft$powerEstimate

print(softPower)

adjacency = adjacency(datExpr0, power = softPower)

##### Turn adjacency into topological overlap

memory.limit(size = 9999999999)
rm()
gc()
TOM = TOMsimilarity(adjacency);
dissTOM = 1-TOM

# Call the hierarchical clustering function
geneTree = hclust(as.dist(dissTOM), method = "average");
# Plot the resulting clustering tree (dendrogram)

#sizeGrWindow(12,9)
pdf(file=paste0(afdir,"/4_Gene clustering on TOM-based dissimilarity.pdf"),width=12,height=9)
plot(geneTree, xlab="", sub="", main = "Gene clustering on TOM-based dissimilarity",
     labels = FALSE, hang = 0.04)
dev.off()


# We like large modules, so we set the minimum module size relatively high:
minModuleSize = 50
# Module identification using dynamic tree cut:
dynamicMods = cutreeDynamic(dendro = geneTree, distM = dissTOM,
                            deepSplit = 2, pamRespectsDendro = FALSE,
                            minClusterSize = minModuleSize);
table(dynamicMods)

# Convert numeric lables into colors
dynamicColors = labels2colors(dynamicMods)
table(dynamicColors)
# Plot the dendrogram and colors underneath
#sizeGrWindow(8,6)
pdf(file=paste0(afdir,"/5_Dynamic Tree Cut.pdf"),width=8,height=6)
plotDendroAndColors(geneTree, dynamicColors, "Dynamic Tree Cut",
                    dendroLabels = FALSE, hang = 0.03,
                    addGuide = TRUE, guideHang = 0.05,
                    main = "Gene dendrogram and module colors")
dev.off()


# Calculate eigengenes
MEList = moduleEigengenes(datExpr0, colors = dynamicColors)
MEs = MEList$eigengenes
# Calculate dissimilarity of module eigengenes
MEDiss = 1-cor(MEs);
# Cluster module eigengenes
METree = hclust(as.dist(MEDiss), method = "average")
# Plot the result
#sizeGrWindow(7, 6)
pdf(file=paste0(afdir,"/6_Clustering of module eigengenes.pdf"),width=7,height=6)
plot(METree, main = "Clustering of module eigengenes",
     xlab = "", sub = "")
MEDissThres = 0.25###### Module shear height
# Plot the cut line into the dendrogram
abline(h=MEDissThres, col = "red")
dev.off()


# Call an automatic merging function
merge = mergeCloseModules(datExpr0, dynamicColors, cutHeight = MEDissThres, verbose = 3)
# The merged module colors
mergedColors = merge$colors
# Eigengenes of the new merged modules:
mergedMEs = merge$newMEs

#sizeGrWindow(12, 9)
pdf(file=paste0(afdir,"/7_merged dynamic.pdf"), width = 9, height = 6.5)
plotDendroAndColors(geneTree, cbind(dynamicColors, mergedColors),
                    c("Dynamic Tree Cut", "Merged dynamic"),
                    dendroLabels = FALSE, hang = 0.03,
                    addGuide = TRUE, guideHang = 0.05)
dev.off()

# Rename to moduleColors
moduleColors = mergedColors
# Construct numerical labels corresponding to the colors
colorOrder = c("grey", standardColors(50))
moduleLabels = match(moduleColors, colorOrder)-1
MEs = mergedMEs

# Save module colors and labels for use in subsequent parts
#save(MEs, TOM, dissTOM,  moduleLabels, moduleColors, geneTree, sft, file = "networkConstruction-stepByStep.RData")


# Define numbers of genes and samples
nGenes = ncol(datExpr0)
nSamples = nrow(datExpr0)

moduleTraitCor = cor(MEs, datTraits, use = "p")
moduleTraitPvalue = corPvalueStudent(moduleTraitCor, nSamples)

#sizeGrWindow(10,6)
pdf(file=paste0(afdir,"/8_Module-trait relationships.pdf"),width=7,height=7.5)
# Will display correlations and their p-values
textMatrix = paste(signif(moduleTraitCor, 2), "\n(",
                   signif(moduleTraitPvalue, 1), ")", sep = "")

dim(textMatrix) = dim(moduleTraitCor)
par(mar = c(6, 8.5, 3, 3))

# Display the correlation values within a heatmap plot
labeledHeatmap(Matrix = moduleTraitCor,
               xLabels = names(datTraits),
               yLabels = names(MEs),
               ySymbols = names(MEs),
               colorLabels = FALSE,
               colors = blueWhiteRed(50),
               textMatrix = textMatrix,
               setStdMargins = FALSE,
               cex.text = 0.5,
               zlim = c(-1,1),
               main = paste("Module-trait relationships"))
dev.off()


######## Define variable weight containing all column of datTraits

###MM and GS


# names (colors) of the modules
modNames = substring(names(MEs), 3)

geneModuleMembership = as.data.frame(cor(datExpr0, MEs, use = "p"))
MMPvalue = as.data.frame(corPvalueStudent(as.matrix(geneModuleMembership), nSamples))

names(geneModuleMembership) = paste("MM", modNames, sep="")
names(MMPvalue) = paste("p.MM", modNames, sep="")

#names of those trait
traitNames=names(datTraits)

geneTraitSignificance = as.data.frame(cor(datExpr0, datTraits, use = "p"))
GSPvalue = as.data.frame(corPvalueStudent(as.matrix(geneTraitSignificance), nSamples))

names(geneTraitSignificance) = paste("GS.", traitNames, sep="")
names(GSPvalue) = paste("p.GS.", traitNames, sep="")


####plot MM vs GS for each trait vs each module


##########example:royalblue and CK
#module="royalblue"
#column = match(module, modNames)
#moduleGenes = moduleColors==module

#trait="CK"
#traitColumn=match(trait,traitNames)

#sizeGrWindow(7, 7)

######

for (trait in traitNames){
  traitColumn=match(trait,traitNames)
  
  for (module in modNames){
    column = match(module, modNames)
    moduleGenes = moduleColors==module
    
    if (nrow(geneModuleMembership[moduleGenes,]) > 1){
      
      #sizeGrWindow(7, 7)
      pdf(file=paste(afdir,"/9_", trait, "_", module,"_Module membership vs gene significance.pdf",sep=""),width=7,height=7)
      par(mfrow = c(1,1))
      verboseScatterplot(abs(geneModuleMembership[moduleGenes, column]),
                         abs(geneTraitSignificance[moduleGenes, traitColumn]),
                         xlab = paste("Module Membership in", module, "module"),
                         ylab = paste("Gene significance for ",trait),
                         main = paste("Module membership vs. gene significance\n"),
                         cex.main = 1.2, cex.lab = 1.2, cex.axis = 1.2, col = module)
      dev.off()
    }
  }
}

#####
names(datExpr0)
probes = names(datExpr0)


#################export GS and MM############### 

geneInfo0 = data.frame(probes= probes,
                       moduleColor = moduleColors)

for (Tra in 1:ncol(geneTraitSignificance))
{
  oldNames = names(geneInfo0)
  geneInfo0 = data.frame(geneInfo0, geneTraitSignificance[,Tra],
                         GSPvalue[, Tra])
  names(geneInfo0) = c(oldNames,names(geneTraitSignificance)[Tra],
                       names(GSPvalue)[Tra])
}

for (mod in 1:ncol(geneModuleMembership))
{
  oldNames = names(geneInfo0)
  geneInfo0 = data.frame(geneInfo0, geneModuleMembership[,mod],
                         MMPvalue[, mod])
  names(geneInfo0) = c(oldNames,names(geneModuleMembership)[mod],
                       names(MMPvalue)[mod])
}
geneOrder =order(geneInfo0$moduleColor)
geneInfo = geneInfo0[geneOrder, ]

write.table(geneInfo, file = paste0(afdir,"/10_GS_and_MM.xls"),sep="\t",row.names=F)



####################################################Visualizing the gene network#######################################################


nGenes = ncol(datExpr0)
nSamples = nrow(datExpr0)


# Transform dissTOM with a power to make moderately strong connections more visible in the heatmap
plotTOM = dissTOM^7
# Set diagonal to NA for a nicer plot
diag(plotTOM) = NA



# Randomly select the number of genes to display TOM heat map
nSelect = 1000
# For reproducibility, we set the random seed
set.seed(10)
select = sample(nGenes, size = nSelect)
selectTOM = dissTOM[select, select]
# There's no simple way of restricting a clustering tree to a subset of genes, so we must re-cluster.
selectTree = hclust(as.dist(selectTOM), method = "average")
selectColors = moduleColors[select]

# Open a graphical window
#sizeGrWindow(9,9)
# Taking the dissimilarity to a power, say 10, makes the plot more informative by effectively changing
# the color palette; setting the diagonal to NA also improves the clarity of the plot
plotDiss = selectTOM^7
diag(plotDiss) = NA

pdf(file=paste0(afdir,"/13_Network heatmap plot_selected genes.pdf"),width=9, height=9)
TOMplot(plotDiss, selectTree, selectColors, main = "Network heatmap plot, selected genes", col=gplots::colorpanel(250,'red',"orange",'lemonchiffon'))
dev.off()



####################################################Visualizing the gene network of eigengenes####################################################


#sizeGrWindow(5,7.5)
pdf(file=paste0(afdir,"/14_Eigengene dendrogram and Eigengene adjacency heatmap.pdf"), width=5, height=7.5)
par(cex = 0.9)
plotEigengeneNetworks(MEs, "", marDendro = c(0,4,1,2), marHeatmap = c(3,4,1,2), cex.lab = 0.8, xLabelsAngle= 90)
dev.off()
