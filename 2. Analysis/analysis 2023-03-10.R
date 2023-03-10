### Analysis script by David W. Braithwaite
# braithwaite@psy.fsu.edu, baixiwei@gmail.com

options(java.parameters = "-Xmx16000m", scipen = 100)

R = 400
graphics_path = "graphics/"
s = 10.5

library(xlsx)
library(reshape2)
library(plyr)
library(dplyr) # for bind_rows
library(ez)
library(lme4)
library(lmerTest)
library(sciplot)

### Helper functions

toPercent = function(x) {return(round(100*x,1))}

mySummary = function(v) {
    S = summary(v)
    S['Std. Dev.'] = sd(v)
    return(S)
}

cross_paste = function(v, w, sep="_") {
    # given string vectors v and w, return all possible combinations of their elements
    f = function(x, y) {
        return(paste(x, y, sep=sep))
    }
    x = outer(v, w, FUN=f)
    y = t(x)
    dim(y) = NULL
    return(y)
}

factorToColumns <- function( D, v ) {
    x = levels(D[,v])
    for (level in x) {
        D[,level] = 0+(D[,v]==level)
    }
    return(D)
}

collapseOver = function( D, v, f=mean, drop_NA=TRUE, drop_NaN=TRUE ) {
    D = melt( D, measure.vars=v )
    if (drop_NA) {
        D = D[!is.na(D[,'value']),]
    }
    if (drop_NaN) {
        D = D[!is.nan(D[,'value']),]
    }
    D = dcast( D, ...~variable, fun.aggregate=f, value.var='value' )
    return(D)
}

myInteractionBargraph = function( D, dv, xv, zv=NULL, main=NULL, main.adj=0, xlab=NULL, xlim=NULL, xnames=NULL, cex.xlab=NULL, ylab=NULL, ylim=NULL, yat=NULL, percent=FALSE, znames=NULL, zcol=NULL, zpos='top', bar.colors=NULL ) {
    ## calculate margins
    # top margin
    if ( is.null(main) ) {
        top     = 0.75
    } else if ( main=="" ) {
        top     = 0.75
    } else {
        top     = 2
    }
    # bottom margin
    if (is.null(xnames)) {
        xnames  = as.character(levels(D[,xv]))
    }
    if (TRUE%in%grepl("\n",xnames)) {
        x_label_ht = 2
    } else {
        x_label_ht = 1
    }
    if (is.null(xlab)) {
        xlab = ""
    }
    if (xlab=="") {
        x_title_ht  = 0
        x_mgp       = c( 0, x_label_ht - 0.25, 0 )
        x_mar       = 1.1 + x_label_ht - 0.25
    } else {
        x_title_ht  = 1
        x_mgp       = c( x_title_ht + 0.1 + x_label_ht - 0.25, x_label_ht - 0.25, 0 )
        x_mar       = 1.1 + x_title_ht + 0.1 + x_label_ht - 0.25
    }
    print( x_mgp )
    print( x_mar )
    flush.console()
    # left margin
    left        = 3.50
    if ( percent ) {
        left    = left + 0.75
    }
    if ( !is.null(ylab) ) {
        if ( grepl("\n",ylab) ) {
            left = left + 0.75
        }
    }
    # right margin
    right   = 0.75
    ## main plot
    par( mar=c( x_mar, left, top, right )+0.1, mgp=c( left-1.25, 0.75, 0 ) )
    if ( !is.null(ylab) ) {
        if ( grepl("\n",ylab) ) {
            par( mgp=c( left-2.00, 0.75, 0 ) )
        }
    }
    if ( is.null(zv) ) {
        x = bargraph.CI(
            axes        = FALSE,
            response    = D[,dv], ylab="", ylim=ylim,
            x.factor    = D[,xv], xlab="", xaxt="n",
            col         = bar.colors
            )
    } else {
        x = bargraph.CI(
            axes        = FALSE,
            response    = D[,dv], ylab="", ylim=ylim,
            x.factor    = D[,xv], xlab="", xaxt="n",
            group       = D[,zv], legend=FALSE,
            col         = bar.colors
            )
    }
    box()
    ## title and axes
    # main title
    if ( !is.null(main) ) {
        title( main=main, adj=main.adj )
    }
    # y axis
    if ( is.null(ylim) ) {
        ylim        = c( 0, max(x$CI) )
    }
    if ( is.null(yat) ) {
        yat         = seq( ylim[1], ylim[2], 0.05 )
    }
    if ( percent ) {
        yticks      = paste0(round(yat*100,0),"%")
        las         = 1
    } else {
        yticks      = yat
        las         = 0
    }
    idxs_a = seq( 1, length(yat), 2 )
    idxs_b = seq( 2, length(yat), 2 )
    axis( side=2, at=yat[idxs_a], labels=yticks[idxs_a], las=las )
    axis( side=2, at=yat[idxs_b], labels=yticks[idxs_b], las=las )
    title( ylab=ylab )
    # x axis
    if (is.null(zv)) {
        xat    = x$xvals
    } else {
        xat    = colMeans(x$xvals)
    }
    par( mgp=x_mgp )
    idxs_a = seq( 1, length(xat), 2 )
    idxs_b = seq( 2, length(xat), 2 )
    axis( side=1, at=xat[idxs_a], labels=xnames[idxs_a], cex.axis=cex.xlab )
    axis( side=1, at=xat[idxs_b], labels=xnames[idxs_b], cex.axis=cex.xlab )
    title( xlab=xlab )
    ## legend
    if (!is.null(zv)) {
        if (is.null(znames) ) {
            znames  = levels(D[,zv])
        }
        if (is.null(bar.colors)) {
            bar.colors = gray.colors(length(levels(D[,zv])))
        }
        if ( is.null(zcol) ) {
            zcol    = length( unique( D[,zv] ) )
        }
        legend( x=zpos, legend=znames, fill=bar.colors, ncol=zcol, bty="n" )
    }
    return(x)
}

obelus = ":"

extractOperator = function(problem) {
    for ( operator in c( "+", "-", "*", obelus ) ) {
        if ( grepl( operator, problem, fixed=TRUE ) ) {
            return( operator )
        }
    }
    return(NA)
}

extractOperand = function(problem,idx) {
    operands = strsplit( problem, extractOperator(problem), fixed=TRUE )[[1]]
    return( operands[idx] )
}

getOperandType = function(num) {
    if ( grepl("/",num,fixed=TRUE) ) {
        if ( grepl(" ",num,fixed=TRUE) ) {
            x = "mixed"
        } else {
            x = "fraction"
        }
    } else if ( grepl(".",num,fixed=TRUE) ) {
        x = "decimal"
    } else {
        x = "whole"
    }
    return( factor( x, levels=c( "fraction", "mixed", "decimal", "whole" ) ) )
}

text_to_fraction = function(txt) {
    f = function(s) {
        result = tryCatch({
            return(eval(parse(text=s)))
        }, error = function(e) {
            return(NA)
        })
    }
    return( sapply( txt, f ) )
}

getNumericValue = function(resp, replace_list=NULL, replace_vals=NULL) {
    if (length(resp)<=1) {
        r = as.character(resp)
        if (!is.null(replace_list)) {
            r = replace(r, replace_list, replace_vals)
        }
        v = as.numeric(r)
        if (is.na(v)) {
            if (grepl("+",r,fixed=TRUE)|grepl("*",r,fixed=TRUE)) {
                v = NA
            } else if (getOperandType(r)%in%c("mixed","fraction")) {
                v = text_to_fraction(mixedToFrac(r))
            }
        }
        return(v)
    } else {
        return(sapply(resp, getNumericValue))
    }
}

mixedToFrac = function(num) {
    # convert text representing a mixed number into text representing a fraction
    if ( is.na( num ) ) {
        return( NA )
    } else {
        x = getOperandType( num )
        if ( x=="NaN" ) {
            return( NA )
        } else if ( x=="fraction" ) {
            frac = num
        } else if ( x=="mixed" ) {
            L = strsplit( num, " " )[[1]]
            whole = as.numeric(L[1])
            frac = L[2]
            L = strsplit( frac, "/", fixed=TRUE )[[1]]
            num = as.numeric(L[1])
            den = as.numeric(L[2])
            frac = paste0( (num+den*whole), "/", den )
        } else if ( x=="whole" | x=="decimal" ) {
            frac = paste0( num, "/", 1 )
        }
        return( frac )
    }
}

getRespFreqs = function(D, p=NULL, v='resp', min_freq=1, mark_key=NULL, acc=FALSE) {
    if (is.null(p)) {
        p = unique(D$prob)
    }
    if (length(p)==1) {
        D = subset(D, prob==p)
        E = data.frame(table(D[,v]))
        E = E[with(E,order(Freq,decreasing=TRUE)),]
        E$Pct = 100*round(E$Freq/sum(E$Freq),2)
        E = subset(E, Freq>=min_freq)
        if (!is.null(mark_key)) {
            E$Corr = ifelse(E$Var1%in%mark_key, "(*)", "")
        }
        if (acc) {
            for (i in 1:nrow(E)) {
                E[i,'acc'] = mean(subset(D, prob==p & resp==E[i,'Var1'])$acc)
            }
        }
        return(E)
    } else {
        return(sapply(p, function(pp) {return(getRespFreqs(D,pp,v,min_freq,mark_key,acc))}, simplify=FALSE))
    }
}

analyzeRAerrors = function(X) {
    notation = X[1,'notation']
    if (notation=='fraction') {
        vars = c('operation', 'operands', 'prob', 'resp', 'acc')
        X = X[,vars]
        X$resp = with(X, ifelse(
            resp=="1.333/5", "1.3/5", ifelse(
            resp=="1.333/1", "1.3/1", ifelse(
            resp=="1.111/15", "1.1/15", ifelse(
            resp=="0.667/1", "0.6/1", ifelse(
            resp=="1.5/1.667", "1.5/1.6", resp))))))
        Y = FA.data[,vars]
        Y = subset(Y, resp!="?/?")
        D = rbind(X,Y)
        D = unique(D)
        D = adply(D, 1, function(r) {
            x1 = subset(X, prob==r$prob)
            x2 = subset(x1, resp==r$resp)
            y1 = subset(Y, prob==r$prob)
            y2 = subset(y1, resp==r$resp)
            r$in_dat = nrow(y2)>0
            r$dat_freq = nrow(y2)
            r$dat_pct = nrow(y2)/nrow(y1)
            r$in_sim = nrow(x2)>0
            r$sim_freq = nrow(x2)
            r$sim_pct = nrow(x2)/nrow(x1)
            return(r)
        })
    } else if (notation=='decimal') {
        vars = c('operation', 'operands', 'prob', 'key_val', 'resp_val', 'acc')
        X = X[,vars]
        Y = subset(DA.data, !is.na(acc))[,vars]
        D = unique(rbind(X,Y))
        D = adply(D, 1, function(r) {
            x1 = subset(X, prob==r$prob)
            x2 = subset(x1, resp_val==r$resp_val)
            y1 = subset(Y, prob==r$prob)
            y2 = subset(y1, resp_val==r$resp_val)
            r$in_dat = nrow(y2)>0
            r$dat_freq = nrow(y2)
            r$dat_pct = nrow(y2)/nrow(y1)
            r$in_sim = nrow(x2)>0
            r$sim_freq = nrow(x2)
            r$sim_pct = nrow(x2)/nrow(x1)
            return(r)
        })
    }
    D = D[with(D, order(operation, operands, prob, -dat_pct, decreasing=FALSE)),]
    E = subset(D, acc==0)
    head(E); tail(E)
    # F = unique(subset(E, in_dat & in_sim, select=c(prob, resp)))
    L = list()
    # What % of children's errors were generated by UMA?
    L[['pct_child_errs_in_sim']] = sum(subset(E, in_sim & in_dat)$dat_freq)/sum(subset(E, in_dat)$dat_freq)
    # What % of UMA's errors were generated by children
    L[['pct_sim_errs_in_child']] = sum(subset(E, in_sim & in_dat)$sim_freq)/sum(subset(E, in_sim)$sim_freq)
    # Frequency correlation
    F = subset(D, in_sim | in_dat)
    L[['frequency_correlation_all']] = with(F, cor.test(dat_freq, sim_freq))
    F = subset(E, in_sim | in_dat)
    L[['frequency_correlation_errs']] = with(F, cor.test(dat_freq, sim_freq))
    F = subset(E, in_dat)
    L[['frequency_correlation_errs_children_only']] = with(F, cor.test(dat_freq, sim_freq))
    F = subset(E, in_sim & in_dat)
    L[['frequency_correlation_errs_both']] = with(F, cor.test(dat_freq, sim_freq))
    if (notation=='fraction') {
        X = subset(D, dat_pct>=.03, select=-c(in_sim, sim_freq, in_dat, dat_freq))
        L[['example_table']] = subset(X, prob%in%c('2/3+3/5', '3/5-1/4', '4/5*3/5', '3/5:1/5'))
    } else if (notation=='decimal') {
        X = subset(D, dat_pct>=.03, select=-c(key_val, in_sim, sim_freq, in_dat, dat_freq))
        L[['example_table']] = subset(X, prob%in%c('12.3+5.6', '0.826+0.12', '0.415+52', '2.4*1.2', '0.32*2.1', '31*3.2'))
        X = subset(D, sim_pct>=.03 | dat_pct>=.03, select=-c(key_val, in_sim, sim_freq, in_dat, dat_freq))
    }
    return(L)
}

analyzeRAstrategies = function(D) {

    # determine group membership for one subject with data E
    ap_groups = c("CS", "ASP", "MP", "VS", "NO")
    classifyAPgroup = function(E, notation='fraction') {

        N           = nrow(E)
        N_AS        = nrow(subset(E, operation%in%c("add", "sub")))
        N_M         = nrow(subset(E, operation=="mul"))

        # correct strategies: used correct strat on >=75% trials
        group_CS    = 0+(mean(E$strat_corr)>=0.75)

        # M perseverators: used M strat on as many trials as would be
        # if used on all multiplication trials and half of all other trials
        group_MP   = 0+(sum(E$strat_M)>=(N_M+(N-N_M)/2))

        # AS perseverators: used AS strat on as many trials as would be
        # if used on all multiplication trials and half of all other trials
        group_ASP   = 0+(sum(E$strat_AS)>=(N_AS+(N-N_AS)/2))

        # variable strategies: used multiple strategies for >=3/4 operations
        if (notation=="fraction") {
            strat_A     = length(unique(subset(E, operation=="add")$strat))>1
            strat_S     = length(unique(subset(E, operation=="sub")$strat))>1
            strat_M     = length(unique(subset(E, operation=="mul")$strat))>1
            strat_D     = length(unique(subset(E, operation=="div")$strat))>1
            group_VS    = 0+((strat_A+strat_S+strat_M+strat_D)>=3)
        } else if (notation=="decimal") {
            strat_A     = length(unique(subset(E, operation=="add" & strat%in%c("AS", "M"))$strat))>1
            strat_M     = length(unique(subset(E, operation=="mul" & strat%in%c("AS", "M"))$strat))>1
            group_VS    = 0+((strat_A+strat_M)>=2)
        }

        # assign to first group matched, otherwise none
        v = sapply(ap_groups, function(g) {
            if (g=="CS") {return(group_CS)}
            else if (g=="MP") {return(group_MP)}
            else if (g=="ASP") {return(group_ASP)}
            else if (g=="VS") {return(group_VS)}
            else {return(0.5)}
        })
        i = which(v==max(v))[1]
        group = ap_groups[i]
        return(c(group, group_CS, group_ASP, group_MP, group_VS, 0+(group=="NO")))
    }

    # add strategy vars to data to ensure consistency between empirical data and simulation results
    addStrats = function(E, notation='fraction') {
        if (notation=='fraction') {
            if ('OpNumKeepDen'%in%names(E)) {
                # empirical data
                E$strat_AS = E$OpNumKeepDen
                E$strat_M = E$IndepComp
                E$strat_D = E$InvertOper
            } else if ('KDON_AS'%in%names(E)) {
                # simulation results
                for (s in c('KDON_AS','KDON_OG','CDON_AS','CDON_OG','ONOD_M','ONOD_OG','CROP_M','ICDM_D','ICDM_OG')) {
                    E[,s] = as.numeric(as.character(E[,s]))
                }
                E$strat_AS = E$KDON_AS + E$KDON_OG + E$CDON_AS + E$CDON_OG
                E$strat_M = E$ONOD_M + E$ONOD_OG
                E$strat_D = E$ICDM_D + E$ICDM_OG + E$CROP_M
                E$strat_corr = with(E, ifelse(
                    operation%in%c('add','sub'), E$strat_AS, ifelse(
                    operation=="mul", E$strat_M, ifelse(
                    operation=="div", E$strat_D, 0))))
            }
            E$strat = with(E, factor(ifelse(strat_AS==1, 'AS', ifelse(strat_M==1, 'M', ifelse(strat_D==1, 'D', 'O'))), levels=c('AS', 'M', 'D', 'O')))    
        } else if (notation=='decimal') {
            if ('strat_add'%in%names(E)) {
                # empirical data
                E$strat_AS      = E$strat_add
                E$strat_M       = E$strat_mul
                E$strat_corr    = E$strat_corr
            } else if ('ADBD_AS'%in%names(E)) {
                # simulation results
                for (s in c('ADBD_AS', 'ADBD_OG', 'ARAD_M', 'ARAD_OG')) {
                    E[,s] = as.numeric(as.character(E[,s]))
                }
                E = adply(E, 1, function(r) {
                    if (r$acc==1) {
                        r$strat_AS = 0+(r$operation=='add')
                        r$strat_M = 0+(r$operation=='mul')
                    } else {
                        r$strat_AS = 0+(r$ADBD_AS + r$ADBD_OG)
                        r$strat_M = 0+(r$ARAD_M + r$ARAD_OG)
                    }
                    return(r)
                })
                E$strat_corr = 0+with(E, ifelse(operation=='add',
                    strat_AS==1 & strat_M==0,
                    strat_M==1 & strat_AS==0))
            }
            E$strat = with(E, factor(ifelse(strat_AS==1, 'AS', ifelse(strat_M==1, 'M', 'O')), levels=c('AS', 'M', 'O')))
        }
        return(E)
    }

    # determine group membership for all subjects in data D
    classifyAPgroups = function(D, notation='fraction') {
        S = data.frame(subjid=unique(D$subjid))
        S = adply(S, 1, function(r) {
            E = subset(D, subjid==r$subjid)
            g = classifyAPgroup(E, notation)
            r[,'ap_group'] = factor(g[1], levels=ap_groups)
            r[,ap_groups] = as.numeric(g[2:6])
            return(r)
        })
        return(S)
    }

    notation = D[1,'notation']
    X.simu = addStrats(D, notation=notation)
    S.simu = join(subj.simu, classifyAPgroups(X.simu, notation=notation))
    S.simu$subjid = factor(S.simu$subjid)
    S.sub = droplevels(subset(S.simu, ap_group!="NO"))
    if (notation=='fraction') {
        X.data = FA.data
    } else {
        X.data = DA.data
    }
    X.data = addStrats(X.data, notation=notation)
    S.data = classifyAPgroups(X.data, notation=notation)

    L = list()
    L[["Patterns in children's data"]] = prop.table(table(S.data$ap_group))
    L[["Patterns in UMA's data"]] = prop.table(table(S.simu$ap_group))

    T = data.frame(group=levels(S.simu$ap_group))
    T = adply(T, 1, function(r) {
        S = subset(S.simu, ap_group==r$group)
        for (v in names(var_pars)) {
            r[,v] = paste0(round(mean(S[,v]),3), " (", round(sd(S[,v]),3), ")")
        }
        return(r)
    })
    L[["Param values in UMA groups"]] = T

    return(L)
}

### Load data

loadTestData = function(sim="sim ALL29_BCD", reload=TRUE) {
    if (reload) {
        # load data
        f = paste0("..\\1. Model\\Output\\", sim, ".csv")
        # create long data
        D = read.csv(f)
        D$grade = factor(D$grade)
        D$g_group = with(D, factor(ifelse(g<=.02, ".01-.02", ifelse(g<=.04, ".03-.04", ifelse(g<=.06, ".05-.06", ifelse(g<=.08, ".07-.08", ".09-.10")))), levels=c(".01-.02", ".03-.04", ".05-.06", ".07-.08", ".09-.10")))
        if ("es"%in%names(D)) {
            D$d = D$es # d "error discount" originally called es "error sensitivity"
        }
        if ("init_count"%in%names(D)) {
            D$ice = D$init_count # "ice" originally called "init_count"
        }
        D$notation = with(D, factor(ifelse(notation=="whole", paste(operands, notation, sep="_"), notation), levels=c('SD_whole', 'MD_whole', 'fraction', 'decimal')))
        D$operation = factor(D$operation, levels=c('add', 'sub', 'mul', 'div'))
        l = cross_paste(levels(D$notation), levels(D$operation))
        D$prob_type = with(D, factor(paste(notation, operation, sep="_"), levels=l))
        D$op1 = sapply(D$prob, extractOperand, idx=1)
        D$op2 = sapply(D$prob, extractOperand, idx=2)
        
        EDD_probs   = c("12.3+5.6", "24.45+0.34", "2.4*1.2", "0.41*0.31")
        ED_probs    = c("3/5+1/5", "4/5+3/5", "3/5-1/5", "4/5-3/5", "3/5*1/5", "4/5*3/5", "3/5:1/5", "4/5:3/5")
        fa_operands = c("ED", "UD")
        da_operands = c("EDD", "UDD", "WD")
        D$operands  = with(D,
            ifelse(notation=="fraction", ifelse(D$prob%in%ED_probs, "ED", "UD"),
            ifelse(notation=="decimal", ifelse(operands=="WD", "WD", ifelse(D$prob%in%EDD_probs, "EDD", "UDD")),
            operands)))
        D$operands  = factor(D$operands, levels=c("SD", "MD", fa_operands, da_operands))
        D$resp      = as.character(D$ans)
        D$ans       = NULL
        D$resp_val  = getNumericValue(D$resp)
        D$key_val   = getNumericValue(D$key)
        D$acc       = as.numeric(D$acc)
        subj_vars = intersect(c('subjid', 'g', 'g_group', 'd', 'c', 'rt_mu', 'rt_sd', 'ice'), names(D))
        for (v in intersect(subj_vars, c('g', 'd', 'c', 'rt_mu', 'rt_sd', 'ice'))) {
            D[,v] = as.numeric(D[,v])
        }
        main_vars = c(
            'grade',
            'notation',
            'operation',
            'prob_type',
            'operands',
            'denoms',
            'op1',
            'op2',
            'prob',
            'key',
            'key_val',
            'resp',
            'resp_val',
            'acc',
            'steps',
            'time')
        strat_vars = setdiff(names(D), c(subj_vars, main_vars))
        for (v in strat_vars) {
            D[,v] = as.numeric(D[,v])
        }
        D = D[,c(subj_vars,main_vars,strat_vars)]

        test.simu = D

        # create SD data
        D       = droplevels(subset(test.simu, notation=="SD_whole"))
        D$op1   = as.numeric(D$op1)
        D$op2   = as.numeric(D$op2)
        D       = subset(D, op1<10 & op2<10 & (operation=="add" | (op1>1 & op2>1)))
        D$resp  = as.numeric(D$resp)
        D$prob  = factor(D$prob, levels=unique(D$prob))
        # D       = droplevels(subset(D, operation=="add" | op1!=1 | op2!=1))
        s_ac    = grepl("add_count", D$rules)
        s_re    = grepl("retrieve", D$rules)
        s_mr    = grepl("mul_rep_add", D$rules)
        D$strat = with(D, factor(ifelse(operation=="add",
            ifelse(add_count==1, "backup", ifelse(retrieve==1, "retrieve", "other")),
            ifelse(mul_rep_add==1, "backup", ifelse(retrieve==1, "retrieve", "other"))), 
            levels=c("retrieve", "backup", "other")))
        D = factorToColumns(D, 'strat')
        SD.simu = D
        
        m       = max(as.numeric(as.character(test.simu$grade)))

        # create DA data
        D       = droplevels(subset(test.simu, grade==m & notation=="decimal"))
        D$op1   = as.numeric(D$op1)
        D$op2   = as.numeric(D$op2)
        DA.simu = D
        
        # create FA data
        D       = droplevels(subset(test.simu, grade==m & notation=="fraction"))
        FA.simu = D
        
        # create subj data
        D = test.simu
        
        S = unique(D[,subj_vars])
        test_vars = NULL
        for (G in levels(D$grade)) {
            E = subset(D, grade==G)
            for (n in unique(E$notation)) {
                F = subset(E, notation==n, select=c(subjid, acc))
                if (nrow(F)>0) {
                    F = collapseOver(F, 'acc')
                    v = paste0("G", G, "_", n, "_acc")
                    test_vars = c(test_vars, v)
                    names(F)[2] = v
                    S = join(S, F)
                }
            }
        }
        subj.simu = S

        # record which model parameters were varied in this sim
        pars = setdiff(subj_vars, c('subjid', 'g_group'))
        var_pars = list()
        for (p in pars) {
            if (length(unique(subj.simu[,p]))>1) {
                var_pars[[p]] = sort(unique(subj.simu[,p]))
            }
        }
        
        # save
        test.simu <<- test.simu
        SD.simu   <<- SD.simu
        DA.simu   <<- DA.simu
        FA.simu   <<- FA.simu
        subj.simu <<- subj.simu
        subj_vars <<- subj_vars
        test_vars <<- test_vars
        var_pars  <<- var_pars

        ## Decimal comparison data: Braithwaite, Sprague, & Siegler (2021)
        load("datasets\\DAX6 data 2021-01-27.RData")
        D = calc.data
        D$operation = factor(tolower(as.character(D$operation)), levels=levels(test.simu$operation))
        D$operands = with(D, factor(ifelse(D$operands=="D-W", "WD", as.character(D$operands)), levels=da_operands))
        D$key_val = D$key
        DA.data <<- droplevels(subset(D, !is.na(acc)))
        
        ## Fraction comparison data: Siegler & Pyke (2013)
        load("datasets\\frac_arith_exp_data.RData")
        D = sp.data
        D$operation = with(D, factor(ifelse(oper=="mult", "mul", as.character(oper)), levels=levels(test.simu$operation)))
        D$operands = with(D, factor(ifelse(denom=="same-denom", "ED", "UD"), levels=fa_operands))
        D$resp_val = getNumericValue(D$resp)
        FA.data <<- droplevels(D)
        
        save(test.simu, SD.simu, DA.simu, DA.data, FA.simu, FA.data, subj.simu, subj_vars, test_vars, var_pars, file=paste0(sim," data.Rdata"))
    } else {
        load(paste0(sim," data.Rdata"))
        test.simu <<- test.simu
        SD.simu   <<- SD.simu
        DA.simu   <<- DA.simu
        DA.data   <<- DA.data
        FA.simu   <<- FA.simu
        FA.data   <<- FA.data
        subj.simu <<- subj.simu
        subj_vars <<- subj_vars
        test_vars <<- test_vars
        var_pars  <<- var_pars
    }
}

loadTestData(reload=FALSE)

### General Overview

nrow(subj.simu) # N

var_pars # varied parameters

nrow(subj.simu)/prod(sapply(var_pars, length)) # nPer

### Study 1. Whole Number Arithmetic

# Accuracy

with(SD.simu, tapply(acc, list(operation, grade), mean))

with(SD.simu, tapply(retrieve, list(operation, grade), mean))

plotSDperf = function(dv='acc') {

    ## Raw data
    D = subset(SD.simu, select=c(subjid, grade, operation, acc, retrieve))
    D$operation = factor(ifelse(D$operation=="add", "Addition", "Multiplication"))
    D       = collapseOver(D, c('acc', 'retrieve'))
    D$dv    = D[,dv]
    D$xfact = D$grade
    D$group = D$operation

    ## Graphical parameters except margins
    if (dv=="acc") {
        main = "(A)"
        ylab = "Percent Correct"
        ylim = c(.75,1)
        yat  = seq(.75, 1, .05)
    } else if (dv=="retrieve") {
        main = "(B)"
        ylab = "Percent Retrieval"
        ylim = c(.5,1)
        yat  = seq(.50, 1, .10)
    }
    yatl = paste0(100*yat, '%')
    xat  = as.numeric(as.character(levels(SD.simu$grade)))
    xatl = as.character(levels(SD.simu$grade))
    xlab = "Grade"
    pchs = c(12, 13)
    ltys = c(1,2)
    pct  = TRUE

    ## Margins
    # top margin
    if ( is.null(main) ) {
        top     = 0.75
    } else if ( main=="" ) {
        top     = 0.75
    } else {
        top     = 2
    }
    # bottom margin
    x_label_ht = 1
    if (is.null(xlab)) {
        xlab = ""
    }
    if (xlab=="") {
        x_title_ht  = 0
        x_mgp       = c( 0, x_label_ht - 0.25, 0 )
        x_mar       = 1.1 + x_label_ht - 0.25
    } else {
        x_title_ht  = 1
        x_mgp       = c( x_title_ht + 0.1 + x_label_ht - 0.25, x_label_ht - 0.25, 0 )
        x_mar       = 1.1 + x_title_ht + 0.1 + x_label_ht - 0.25
    }
    # left margin
    left        = 3.50
    if ( pct ) {
        left    = left + 0.75
    }
    # right margin
    right   = 0.75
    par( mar=c( x_mar, left, top, right )+0.1, mgp=c( left-1.25, 0.75, 0 ) )

    ## Main plot
    with(D, lineplot.CI(xfact, dv, group,
        xlab=xlab, xaxt="n",
        ylab=ylab, ylim=ylim, yaxt="n", 
        legend=FALSE, pch=pchs, lty=ltys,
        err.width=0.05, err.col="dark gray"))

    ## Title, axes, and legend
    title( main=main, adj=0 )
    axis(2, at=yat, labels=yatl, las=1)
    par( mgp=x_mgp )
    axis(1, at=xat, labels=xatl)
    title( xlab=xlab )
    legend("bottomright", legend=levels(D$group), pch=pchs, lty=ltys)
}

png( paste0(graphics_path,"Figure 2 SD acc ret.png"), width=6.5, height=3, units='in', res=R, pointsize=s )
par(mfrow=c(1,2))
plotSDperf("acc")
plotSDperf("retrieve")
dev.off()

# Types of Errors

with(SD.simu, tapply(1-acc, operation, mean))

classifyAddErrs = function(cat_cutoff=5) {
    D = droplevels(subset(SD.simu, operation=="add" & acc==0))
    D$err_diff  = D$resp - D$key
    D$err_plot  = factor(
        ifelse(abs(D$err_diff)<=cat_cutoff, D$err_diff,
        ifelse(D$err_diff>cat_cutoff, paste0(">",cat_cutoff),
        paste0("<-",cat_cutoff))),
        levels = c(paste0("<-",cat_cutoff), setdiff(-cat_cutoff:cat_cutoff, 0), paste0(">",cat_cutoff)))
    return(subset(D, select=c(prob, op1, op2, key, resp, err_diff, err_plot)))
}

Dadd = classifyAddErrs()
prop.table(table(abs(Dadd$err_diff)))

classifyMulErrs = function(cat_cutoff=5) {
    D = droplevels(subset(SD.simu, operation=="mul" & acc==0))
    diffs = c(-10:-1, 1:10)
    X = unique(subset(D, select=c(prob, op1, op2)))
    L = sapply(as.character(X$prob), function(p) {
        op1 = X[X$prob==p, 'op1']
        op2 = X[X$prob==p, 'op2']
        M = sapply(as.character(diffs), function(n) {
            x = c()
            a = op1+as.numeric(n)
            b = op2+as.numeric(n)
            # if (a > 0 & a <= 10) {
            if (a>0) {
                x = c(x, a*op2)
            }
            # if (b > 0 & b <= 10) {
            if (b>0) {
                x = c(x, op1*b)
            }
            return(x)
        }, simplify=FALSE)
        return(M)
    }, simplify=FALSE)
    
    v = 1:10
    all_answers = unique(as.vector(outer(v,v)))
    
    D = droplevels(subset(SD.simu, operation=="mul" & acc==0))
    # op_diff
    D$op_diff   = NA
    diff_seq    = as.character(diffs[order(abs(diffs), decreasing=FALSE)])
    D = adply(D, 1, function(r) {
        # op_diff is number with smallest abs val <= 10 such that resp equals either operand times other operand plus or minus op_diff
        for (d in as.character(diff_seq)) {
            if (r$resp%in%L[[as.character(r$prob)]][[d]]) {
                r$op_diff = as.numeric(d)
                break
            }
        }
        return(r)
    })
    # err_cat
    D$err_cat = factor("other", levels=c("operand", "operation", "table", "other"))
    D = adply(D, 1, function(r) {
        if (is.na(r$resp)) {
            r$err_cat = "other"
        } else if (!is.na(r$op_diff)) {
            r$err_cat = "operand"
        } else if (r$resp==(r$op1+r$op2)) {
            r$err_cat = "operation"
        } else if (r$resp%in%all_answers) {
            r$err_cat = "table"
        }
        return(r)
    })
    # err_plot
    D$err_plot = factor(
        ifelse(is.na(D$op_diff), NA,
        ifelse(abs(D$op_diff)<=cat_cutoff, D$op_diff,
        ifelse(D$op_diff>cat_cutoff, paste0(">",cat_cutoff),
        paste0("<-",cat_cutoff)))),
        levels = c(paste0("<-",cat_cutoff), setdiff(-cat_cutoff:cat_cutoff, 0), paste0(">",cat_cutoff)))

    return(D)
}

Dmul = classifyMulErrs()
prop.table(table(Dmul$err_cat))
getRespFreqs(subset(SD.simu, acc==0), "6*9")
prop.table(table(abs(subset(Dmul, err_cat=="operand")$op_diff)))

plotSDerrs = function(D, main, xlab) {
    E = D
    E$freq = 1
    E = collapseOver(E[,c('err_plot', 'freq')], 'freq', sum)
    E$freq = E$freq / sum(E$freq)
    myInteractionBargraph(E, 'freq', 'err_plot', main=main, xlab=xlab, cex.xlab=0.8, ylab="Percent of Errors", ylim=c(0,0.4), yat=seq(0,0.4,0.1), percent=TRUE)
}

png( paste0(graphics_path,"Figure 3 SD errors.png"), width=6.5, height=3, units='in', res=R, pointsize=s )
par(mfrow=c(1,2))
plotSDerrs(Dadd, "(A)", "UMA's Answer Minus Correct Answer")
plotSDerrs(subset(Dmul, err_cat=="operand"), "(B)", "Operand Difference")
dev.off()

# Influences on Problem Difficulty

analyzeSizeEffect = function(D, operation="add", detailed=FALSE) {
D           = D[D$operation==operation,]
D$max_op    = with(D, ifelse(op1>=op2, op1, op2))
D$op_sum    = D$op1+D$op2
D$grade     = as.numeric(as.character(D$grade))
D$retrieve  = as.numeric(D$retrieve)
D_all       = collapseOver(subset(D, select=c(prob, op1, op2, op_sum, max_op, acc, retrieve)), c('acc', 'retrieve'))
D_grd       = collapseOver(subset(D, select=c(grade, prob, op1, op2, op_sum, max_op, acc, retrieve)), c('acc', 'retrieve'))

L = list()
L[['descriptives - accuracy']] = with(D_all, tapply(acc, op_sum, mean))
L[['regression - accuracy']] = summary(lm(acc~op_sum, data=D_all))

if (detailed) {
    L[['descriptives - accuracy - by grade']] = with(D_grd, tapply(acc, list(grade, op_sum), mean))
    L[['regression - accuracy - by grade']] = summary(lmer(acc~op_sum*grade+(1|prob), data=D_grd))

    # textbook frequencies
    D = unique(subset(SD.simu, select=c(prob, operation, op1, op2)))
    D$op_sum = with(D, op1+op2)
    X = read.csv("..\\1. Model\\Problem Sets Training\\go_math.csv")
    X$freq = 1
    X = collapseOver(subset(X, select=c(prob, freq)), 'freq', sum)
    D = join(D, X)
    D$freq = ifelse(is.na(D$freq), 0, D$freq)
    E = D[D$operation==operation,]
    L[['textbook frequency correlation']] = with(E, cor.test(freq, op_sum))
}

return(L)
}

analyzeSizeEffect(SD.simu, 'add', FALSE) # TRUE to add grade to regression

analyzeSizeEffect(SD.simu, 'mul', FALSE) # TRUE to add grade to regression

# Individual Differences in Strategy Use

plotSDretr = function(p='g') {
    ## Raw data
    D = subset(SD.simu, select=c(subjid, g, g_group, d, rt_mu, ice, grade, operation, retrieve))
    D       = collapseOver(D, c('retrieve'))
    D$dv    = D[,'retrieve']
    D$xfact = D$grade

    if (p%in%c("d", "rt_mu", "ice")) {
        D$p = factor(D[,p], levels=sort(unique(as.numeric(as.character(D[,p])))))
    } else if (p%in%c("g", "g_group")) {
        D$p = D$g_group
    }
    if (p%in%c("g", "g_group", "d", "ice")) {
        # show "best" levels on top
        D$p = factor(D$p, levels=rev(levels(D$p)))
    }
    
    ## Create dummy grouping variable that combines operation & params
    lvls    = cross_paste(as.character(levels(D$operation)), as.character(levels(D$p)))
    D$group = factor(paste(D$operation, D$p, sep="_"), levels=lvls)
    n_op    = 2
    n_p     = length(levels(D$p))

    ## Graphical parameters except margins
    main = ifelse(p%in%c("g", "g_group"), "(A)",
        ifelse(p=="d", "(B)",
        ifelse(p=="rt_mu", "(C)", "(D)")))
    ylab = "Percent Retrieval"
    ylim = c(.2,1)
    yat  = seq(.2, 1, .2)
    yatl = paste0(100*yat, '%')
    xat  = as.numeric(as.character(levels(SD.simu$grade)))
    xatl = as.character(levels(SD.simu$grade))
    xlab = "Grade"
    pct  = TRUE

    if (n_p==4) {
        pch_base = c(0,1,5,6)
    } else if (n_p==5) {
        pch_base = c(0,1,5,6,2)
    }
    pchs    = c(pch_base, pch_base)
    ltys    = c(rep(1, n_p), rep(2, n_p))
    leglab  = paste0(p, " = ", as.character(levels(D$p)))

    ## Margins
    # top margin
    if ( is.null(main) ) {
        top     = 0.75
    } else if ( main=="" ) {
        top     = 0.75
    } else {
        top     = 2
    }
    # bottom margin
    x_label_ht = 1
    if (is.null(xlab)) {
        xlab = ""
    }
    if (xlab=="") {
        x_title_ht  = 0
        x_mgp       = c( 0, x_label_ht - 0.25, 0 )
        x_mar       = 1.1 + x_label_ht - 0.25
    } else {
        x_title_ht  = 1
        x_mgp       = c( x_title_ht + 0.1 + x_label_ht - 0.25, x_label_ht - 0.25, 0 )
        x_mar       = 1.1 + x_title_ht + 0.1 + x_label_ht - 0.25
    }
    # left margin
    left        = 3.50
    if ( pct ) {
        left    = left + 0.75
    }
    # right margin
    right   = 0.75
    par( mar=c( x_mar, left, top, right )+0.1, mgp=c( left-1.25, 0.75, 0 ) )

    ## Main plot
    with(D, lineplot.CI(xfact, dv, group,
        xlab=xlab, xaxt="n",
        ylab=ylab, ylim=ylim, yaxt="n", 
        legend=FALSE, pch=pchs, lty=ltys,
        err.width=0.05, err.col="darkgray"))

    ## Title, axes, and legend
    title( main=main, adj=0 )
    axis(2, at=yat, labels=yatl, las=1)
    par( mgp=x_mgp )
    axis(1, at=xat, labels=xatl)
    title( xlab=xlab )
    legend("bottomright", legend=leglab, pch=pch_base)
}

png( paste0(graphics_path,"Figure 4 SD add ret by params.png"), width=6.5, height=6, units='in', res=R, pointsize=s )
par(mfrow=c(2,2))
plotSDretr('g')
plotSDretr('d')
plotSDretr('rt_mu')
plotSDretr('ice')
dev.off()

makeTable4 = function(fn=mean) {
D = collapseOver(droplevels(subset(SD.simu, grade==1 & operation=="add"))[,c('subjid', names(var_pars), 'acc', 'retrieve')], c('acc', 'retrieve'))
D$acc_bin = factor(ifelse(D$acc>=mean(D$acc), "high", "low"), levels=c("high", "low"))
D$ret_bin = factor(ifelse(D$retrieve>=mean(D$retrieve), "high", "low"), levels=c("high", "low"))
D$group = with(D, factor(ifelse(acc_bin=="high", ifelse(ret_bin=="high", "Good", "Perf"), ifelse(ret_bin=="high", "None", "NSG")), levels=c("Good", "NSG", "Perf", "None")))
T = data.frame(table(D$group))
names(T) = c('group', 'freq')
T$freq = T$freq/sum(T$freq)
T = T[with(T, order(group)),]
D = D[,c('group', 'acc', 'retrieve', names(var_pars))]
D = collapseOver(D, setdiff(names(D), 'group'), fn)
D = join(T,D)
return(D)
}
makeTable4(mean)
makeTable4(sd)

### Study 2. Fraction arithmetic

# Accuracy

summary(FA.simu$acc)
summary(FA.data$acc)

# Types of Errors

analyzeRAerrors(FA.simu)

# Influences on Problem Difficulty

plotFAacc = function(src='data') {
    if (src=='data') {
        load("datasets\\frac_arith_exp_data.RData")
        D = sp.data
        D$operation = with(D, factor(ifelse(oper=="mult", "mul", as.character(oper)), levels=levels(FA.simu$operation)))
        D$operands = with(D, factor(ifelse(denom=="same-denom", "ED", "UD"), levels=levels(FA.simu$operands)))
        D$resp_val = getNumericValue(D$resp)
        main = '(A)'
    } else if (src%in%c('sim','simu')) {
        D = FA.simu
        main = '(B)'
    }
    D = subset(D, select=c(subjid, operation, operands, acc))
    D = collapseOver(D, 'acc')
    # y variable
    dv = 'acc'
    ylab = 'Percent Correct'
    ylim = c(0,1)
    yat = seq(0,1,.2)
    pct = TRUE
    # x variable
    xv = 'operation'
    xlab = 'Operation'
    xnames = c('Add','Sub','Mul','Div')
    # z variable
    zv = 'operands'
    znames = c('ED','UD')
    
    myInteractionBargraph(D, dv, xv, zv, main,
        xlab=xlab, xnames=xnames,
        ylab=ylab, ylim=ylim, yat=yat, percent=pct,
        znames=znames)
}

png( paste0(graphics_path,"Figure 5 FA acc.png"), width=6.5, height=3, units='in', res=R, pointsize=s )
par(mfrow=c(1,2))
plotFAacc('data')
plotFAacc('simu')
dev.off()

with(FA.simu, tapply(acc, list(operation, denoms), mean))

with(subset(FA.simu, operation%in%c("add", "sub")), tapply(acc, denoms, mean))

with(subset(FA.simu, operation=="div"), tapply(acc, prob, mean))

with(subset(FA.data, operation=="div"), tapply(acc, prob, mean))

# Individual Differences in Strategy Use

analyzeRAstrategies(FA.simu)

### Study 3. Decimal arithmetic

# Accuracy

summary(DA.simu$acc)
summary(DA.data$acc)

# Errors

analyzeRAerrors(DA.simu)

# Influences on Problem Difficulty

plotDAacc = function(src='data') {
    if (src=='data') {
        D = DA.data
        # load("..\\..\\..\\1.4 Datasets\\Braithwaite Pyke Siegler 2017 FARRA\\frac_arith_exp_data.RData")
        # D = sp.data
        # D$operation = with(D, factor(ifelse(oper=="mult", "mul", as.character(oper)), levels=levels(FA.simu$operation)))
        # D$operands = with(D, factor(ifelse(denom=="same-denom", "ED", "UD"), levels=levels(FA.simu$operands)))
        # D$resp_val = getNumericValue(D$resp)
        main = '(A)'
    } else if (src%in%c('sim','simu')) {
        D = DA.simu
        main = '(B)'
    }
    D$operands = factor(ifelse(D$operands=="WD", "WD", "DD"), levels=c("DD", "WD"))
    D = subset(D, select=c(subjid, operation, operands, acc))
    D = collapseOver(D, 'acc')
    # y variable
    dv = 'acc'
    ylab = 'Percent Correct'
    ylim = c(0,1)
    yat = seq(0,1,.2)
    pct = TRUE
    # x variable
    xv = 'operation'
    xlab = 'Operation'
    xnames = c('Add','Mul')
    # z variable
    zv = 'operands'
    znames = c('DD','WD')
    
    myInteractionBargraph(D, dv, xv, zv, main,
        xlab=xlab, xnames=xnames,
        ylab=ylab, ylim=ylim, yat=yat, percent=pct,
        znames=znames)
}

png( paste0(graphics_path,"Figure 6 DA acc.png"), width=6.5, height=3, units='in', res=R, pointsize=s )
par(mfrow=c(1,2))
plotDAacc('data')
plotDAacc('simu')
dev.off()

D = DA.simu
D$operands = with(D, factor(ifelse(operands=="WD", "WD", "DD"), levels=c("DD", "WD")))
with(D, tapply(acc, operation, mean))
with(D, tapply(acc, list(operation, operands), mean))

# Individual Differences in Strategy Use

analyzeRAstrategies(DA.simu)

### Study 4. Relations Among Basic and Advanced Arithmetic Skills

makeTable9 = function(type="main") {

# prepare data
S = subj.simu[,subj_vars]
for (v in c('g', 'd', 'rt_mu', 'ice')) {
    S[,paste0(v,"2")] = S[,v]**2
}
D = subset(SD.simu, grade==1 & operation=="add", select=c(subjid, acc))
D = collapseOver(D, 'acc')
names(D)[2] = 'G1_SD_add'
S = join(S,D)
D = subset(SD.simu, grade==3 & operation=="mul", select=c(subjid, acc))
D = collapseOver(D, 'acc')
names(D)[2] = 'G3_SD_mul'
S = join(S,D)
D = subset(FA.simu, select=c(subjid, acc))
D = collapseOver(D, 'acc')
names(D)[2] = 'G6_FA'
S = join(S,D)
D = subset(DA.simu, select=c(subjid, acc))
D = collapseOver(D, 'acc')
names(D)[2] = 'G6_DA'
S = join(S,D)
Sstd = S
for (v in c("G1_SD_add", "G3_SD_mul", "G6_FA", "G6_DA")) {
    Sstd[,v] = (Sstd[,v] - mean(Sstd[,v]))/sd(Sstd[,v])
}

D = data.frame(
    predictor   = c("G1_SD_add", "G1_SD_add", "G1_SD_add", "G3_SD_mul", "G3_SD_mul"),
    outcome     = c("G3_SD_mul", "G6_FA", "G6_DA", "G6_FA", "G6_DA"))
if (type=="main") {
    # do analyses
    D = adply(D, 1, function(r) {
        X = S
        X$predictor = X[,r$predictor]
        X$outcome   = X[,r$outcome]
        Xstd = Sstd
        Xstd$predictor = Xstd[,r$predictor]
        Xstd$outcome   = Xstd[,r$outcome]
        fit1 = lm(outcome~predictor, data=X)
        fit1std = lm(outcome~predictor, data=Xstd)
        fit2 = lm(outcome~g+g2+d+d2+rt_mu+rt_mu2+ice+ice2, data=X)
        fit3 = lm(outcome~g+g2+d+d2+rt_mu+rt_mu2+ice+ice2+predictor, data=X)
        fit3std = lm(outcome~g+g2+d+d2+rt_mu+rt_mu2+ice+ice2+predictor, data=Xstd)
        r$pred_only_B   = summary(fit1)$coefficients['predictor', 'Estimate']
        r$pred_only_b   = summary(fit1std)$coefficients['predictor', 'Estimate']
        r$pred_only_p   = round(summary(fit1)$coefficients['predictor', 'Pr(>|t|)'],4)
        r$pred_only_r2  = summary(fit1)$r.squared
        r$with_cont_B   = summary(fit3)$coefficients['predictor', 'Estimate']
        r$with_cont_b   = summary(fit3std)$coefficients['predictor', 'Estimate']
        r$with_cont_p   = round(summary(fit3)$coefficients['predictor', 'Pr(>|t|)'],4)
        r$with_cont_r2  = summary(fit3)$r.squared
        r$with_cont_dr2 = summary(fit3)$r.squared - summary(fit2)$r.squared
        return(r)
    })
    return(D)
} else if (type=="supp") {
    for (i in 1:nrow(D)) {
        print(D[i,c('predictor', 'outcome')])
        X = S
        X$predictor = X[,D[i,'predictor']]
        X$outcome   = X[,D[i,'outcome']]
        print(summary(lm(outcome~g+g2+d+d2+rt_mu+rt_mu2+ice+ice2+predictor, data=X)))
    }
}
}

makeTable9()

### Reviewer questions

# is it reasonable to assume students would switch their learning mode off during assessment?

D = DA.data
D$bin = factor(ifelse(D$idx<=6, "part1", "part2"))
with(D, tapply(acc, bin, mean))
E = dcast(D, subjid~bin, fun.aggregate=mean, value.var='acc')
with(E, t.test(part1, part2, paired=TRUE))

# Was there a bias for under-counting rather than over-counting in the UMA results for whole numbers, or did both occur at the same rate? 

with(subset(SD.simu, acc==0 & !is.na(resp_val)), tapply(resp_val<key_val, operation, mean))
with(subset(SD.simu, acc==0 & !is.na(resp_val)), tapply(resp_val<key_val, list(operation, grade), mean))

# work by van der Maas (which is cited) shows a wide range of errors in multiplication

f = function(p) {
    # compare errors on problem p in Buwalda et al. (2016) and UMA's data
    D = subset(SD.simu, prob==p & acc==0)
    if (p=="3*4") {
        v = c(16, 9, 8, 7, 15, 14, 11, 2, 6, 13)
    } else if (p=="6*9") {
        v = c(45, 56, 63, 36, 53, 51, 64, 52, 72, 42)
    }
    return(list(
        setdiff(v, D$resp_val),
        getRespFreqs(D)[1:20,]))
}

sapply(c("3*4", "6*9"), f, simplify=FALSE)

# textbook problem frequencies in G4-G6 only

D = read.csv("..\\1. Model\\Problem Sets Training\\go_math.csv")
D = subset(D, grade>3 & opSubTypes%in%c("fraction", "decimal"))
table(D$opSubTypes)
