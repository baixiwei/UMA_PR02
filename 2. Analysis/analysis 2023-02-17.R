### Analysis script by David W. Braithwaite
# braithwaite@psy.fsu.edu, baixiwei@gmail.com

options(java.parameters = "-Xmx16000m", scipen = 100)

# source("..\\..\\fractions.R")
# source("..\\..\\utils.R")
library(xlsx)
library(reshape2)
library(plyr)
library(dplyr) # for bind_rows
library(ez)
library(lme4)
library(lmerTest)

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

analyzeSDerrors = function(D, operation="add") {

    D = D[D$operation==operation,]
    D$err = 1-D$acc
    D$err_diff = with(D, abs(key_val-resp_val))
    E = subset(D, err==1)
    if (operation=="add") {
        # E = subset(E, grade==1)
        result = list(
            'summary'   = mySummary(D$err),
            'err_diff'  = with(E, mySummary(err_diff)),
            'err_in_1'  = with(E, mean(err_diff<=1)),
            'err_in_2'  = with(E, mean(err_diff<=2)),
            'err_in_3'  = with(E, mean(err_diff<=3)))
    } else if (operation=="mul") {
        # Multiplication. Compare to Lemaire & Siegler (1995)
        # E = subset(E, grade==3)
        F = adply(E, 1, function(r) {
        
            X = expand.grid(op1=1:10, op2=1:10)
            X = subset(X, !(op1==r$op1 & op2==r$op2) & (op1==r$op1 | op2==r$op2))
            X = X[with(X, order(op1, op2, decreasing=FALSE)),]
            X$product = X$op1 * X$op2
        
            # operand_errors  = c((1:10)*r$op1, (1:10)*r$op2)
            operation_error = r$op1+r$op2
            v = setdiff(1:10, c(r$op1, r$op2))
            table_errors = unique(as.vector(outer(v,v)))
            if (is.na(r$resp)) {
                r$err_type = 'err_other'
            # } else if (r$resp%in%operand_errors) {
            } else if (r$resp%in%X$product) {
                r$err_type = 'err_operand'
                Y = subset(X, product==r$resp)
                Y$op_diff = with(Y, ifelse(op1==r$op1, abs(op2-r$op2), abs(op1-r$op1)))
                Y = Y[with(Y, order(op_diff, decreasing=FALSE)),]
                r$err_diff = Y[1,'op_diff']
                r$err_in_1 = 0+(r$err_diff<=1)
                r$err_in_2 = 0+(r$err_diff<=2)
                r$err_in_3 = 0+(r$err_diff<=3)
            } else if (r$resp_val==operation_error) {
                r$err_type = 'err_operation'
            } else if (r$resp%in%table_errors) {
                r$err_type = 'err_table'
            } else {
                r$err_type = 'err_other'
            }
            r$err_type = factor(r$err_type, levels=c('err_operand', 'err_operation', 'err_table', 'err_other'))
            for (et in levels(r$err_type)) {
                r[,as.character(et)] = 0+(r$err_type==et)
            }
            return(r)
        })
        G = melt(F, measure.vars=c('err_operand', 'err_table', 'err_operation', 'err_other'))
        H = melt(subset(F, err_type=='err_operand'), measure.vars=c('err_in_1', 'err_in_2', 'err_in_3'))

        result = list(
            'summary'   = mySummary(D$err),
            'err_rates' = with(D, tapply(err, grade, mean)),
            'err_types' = with(G, tapply(value, variable, mean)),
            'operand_err_dist' = with(H, tapply(value, variable, mean)),
            'err_types_by_grade' = with(G, tapply(value, list(variable, grade), mean)))
        
        F$freq = 1
        F$prob_resp = with(F, paste(prob, resp, sep="="))
    }
    return(result)
}

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

analyzeSDstrategies = function(D, detailed=FALSE) {
    E = D[,c('grade', subj_vars, 'operation', 'retrieve')]
    E$grade = paste("grade", E$grade)
    E = collapseOver(E, 'retrieve')
    for (v in names(var_pars)) {
        E[,v] = as.numeric(as.character(E[,v]))
    }
    f = function(v, E) {
        E$v = E[,v]
        L = list()
        if (detailed) {
            L[['descriptives']] = with(E, tapply(retrieve, list(v, grade, operation), mean))
        }
        for (o in c("add", "mul")) {
            X = droplevels(subset(E, operation==o))
            L[[o]] = sapply(unique(X$grade), function(this_grade) {
                Y = droplevels(subset(X, grade==this_grade))
                return(round(summary(lm(retrieve~v, data=Y))$coefficients['v',],5))
            }, simplify=FALSE)
        }
        return(L)
    }
    return(sapply(names(var_pars), f, E=E, simplify=FALSE))
}

analyzeRAaccuracies = function(D, p=NULL) {
    if (is.null(p)) {
        R = with(D, list(
            'overall'               =   mean(acc),
            'operation'             =   tapply(acc, operation, mean),
            'operation * operands'  =   tapply(acc, list(operation, operands), mean)))
    } else {
        R = with(D, list(
            'overall'               =   tapply(acc, D[,p], mean),
            'operation'             =   tapply(acc, list(D[,p], operation), mean),
            'operation * operands'  =   tapply(acc, list(operation, operands, D[,p]), mean)))
    }
    return(lapply(R, toPercent))
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
    # Which of children's errors did the sim fail to generate?
    F = subset(E, in_dat & !in_sim)
    L[['child_errs_not_in_sim']] = F[F$dat_freq>=5,]
    # Which of UMA's errors were not in children's data?
    F = subset(E, !in_dat & in_sim)
    L[['sim_errs_not_in_child']] = F[F$sim_freq>=5,]
    # Most common errors?
    X = data.frame(prob=unique(D$prob))
    X = adply(X,1,function(r) {
        Y = subset(E, prob==r$prob)
        return(Y[1,])
    })
    L[['most_common_child_errs_by_prob']] = X
    D2 = D[with(D, order(operation, operands, prob, -sim_pct, decreasing=FALSE)),]
    E2 = subset(D2, acc==0)
    X = adply(X,1,function(r) {
        Y = subset(E2, prob==r$prob)
        return(Y[1,])
    })
    L[['most_common_sim_errs_by_prob']] = X
    if (notation=='fraction') {
        X = subset(D, dat_pct>=.03, select=-c(in_sim, sim_freq, in_dat, dat_freq))
        L[['example_table']] = subset(X, prob%in%c('2/3+3/5', '3/5-1/4', '4/5*3/5', '3/5:1/5'))
        X = subset(D, select=-c(in_sim, sim_freq, in_dat, dat_freq))
        L[['example_table_full']] = subset(X, prob%in%c('2/3+3/5', '3/5-1/4', '4/5*3/5', '3/5:1/5'))
    } else if (notation=='decimal') {
        X = subset(D, dat_pct>=.03, select=-c(key_val, in_sim, sim_freq, in_dat, dat_freq))
        L[['example_table']] = subset(X, prob%in%c('12.3+5.6', '0.826+0.12', '0.415+52', '2.4*1.2', '0.32*2.1', '31*3.2'))
        X = subset(D, sim_pct>=.03 | dat_pct>=.03, select=-c(key_val, in_sim, sim_freq, in_dat, dat_freq))
        L[['example_table_full']] = subset(X, prob%in%c('12.3+5.6', '0.826+0.12', '0.415+52', '2.4*1.2', '0.32*2.1', '31*3.2'))
        # '2.46+4.1',
    }
    return(L)
}

analyzeRAstrategies = function(D, detailed=FALSE) {

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
            # r[,'ap_label'] = ap_labels[which(ap_groups==g[1])]
            # r[,'ap_color'] = ap_colors[which(ap_groups==g[1])]
            # r[,'ap_pch'] = ap_pchs[which(ap_groups==g[1])]
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

    if (detailed) {
    L[["Logistic regressions"]] = list(
        'CS' = summary(glm(CS~g+es+rt_mu+init_count, data=S.simu, family=binomial)),
        'ASP' = summary(glm(ASP~g+es+rt_mu+init_count, data=S.simu, family=binomial)),
        'MP' = summary(glm(MP~g+es+rt_mu+init_count, data=S.simu, family=binomial)),
        'VS' = summary(glm(VS~g+es+rt_mu+init_count, data=S.simu, family=binomial)))
    L[["ANOVAs"]] = list(
        'g' = ezANOVA(data=S.sub, wid=subjid, dv=g, between=ap_group, type=3),
        'es' = ezANOVA(data=S.sub, wid=subjid, dv=es, between=ap_group, type=3),
        'rt_mu' = ezANOVA(data=S.sub, wid=subjid, dv=rt_mu, between=ap_group, type=3),
        'init_count' = ezANOVA(data=S.sub, wid=subjid, dv=init_count, between=ap_group, type=3))
    }

    return(L)
}

analyzeCorrelations = function(detailed=FALSE) {

# prepare data
S = subj.simu[,subj_vars]
for (v in c('g', 'es', 'rt_mu', 'init_count')) {
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
vars = c("G1_SD_add", "G3_SD_mul", "G6_FA", "G6_DA")

# do analyses
D = data.frame(
    predictor   = c("G1_SD_add", "G1_SD_add", "G1_SD_add", "G3_SD_mul", "G3_SD_mul"),
    outcome     = c("G3_SD_mul", "G6_FA", "G6_DA", "G6_FA", "G6_DA"))
D = adply(D, 1, function(r) {
    X = S
    # X$predictor = X[,r$predictor]
    # X$outcome   = X[,r$outcome]
    X$predictor = X[,predictor]
    X$outcome   = X[,outcome]
    fit1 = lm(outcome~predictor, data=X)
    fit2 = lm(outcome~g+g2+es+es2+rt_mu+rt_mu2+init_count+init_count2, data=X)
    fit3 = lm(outcome~g+g2+es+es2+rt_mu+rt_mu2+init_count+init_count2+predictor, data=X)
    r$pred_only_B   = summary(fit1)$coefficients['predictor', 'Estimate']
    r$pred_only_p   = round(summary(fit1)$coefficients['predictor', 'Pr(>|t|)'],4)
    r$pred_only_r2  = summary(fit1)$r.squared
    r$with_cont_B   = summary(fit3)$coefficients['predictor', 'Estimate']
    r$with_cont_p   = round(summary(fit3)$coefficients['predictor', 'Pr(>|t|)'],4)
    r$with_cont_r2  = summary(fit3)$r.squared
    r$with_cont_dr2 = summary(fit3)$r.squared - summary(fit2)$r.squared
    if (detailed) {
        # determine variance uniquely explained by each parameter
        for (param in c('g', 'es', 'rt_mu', 'init_count')) {
            if (param=='g') {
                fitx = lm(outcome~es+es2+rt_mu+rt_mu2+init_count+init_count2+predictor, data=X)
            } else if (param=='es') {
                fitx = lm(outcome~g+g2+rt_mu+rt_mu2+init_count+init_count2+predictor, data=X)
            } else if (param=='rt_mu') {
                fitx = lm(outcome~g+g2+es+es2+init_count+init_count2+predictor, data=X)
            } else if (param=='init_count') {
                fitx = lm(outcome~g+g2+es+es2+rt_mu+rt_mu2+predictor, data=X)
            }
        r[,paste0(param,'_p')] = round(anova(fit3,fitx)["Pr(>F)"][2,1], 4)
        r[,paste0(param,'_r2')] = summary(fit3)$r.squared - summary(fitx)$r.squared
        }
    }
    return(r)
})
D

return(D)
}

### Graphics parameters and functions

R = 400
graphics_path = "graphics/"
s = 10.5

myLinePlot = function(
    D, dv, main, 
    ylab, ylim, yat, yatl, pct,
    xv, xlab, xat, xatl,
    zv, zlab, pchs, legpos) {
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
    ## main plot
    par( mar=c( x_mar, left, top, right )+0.1, mgp=c( left-1.25, 0.75, 0 ) )

    # plot
    L = levels(D[,zv])
    for (i in 1:length(L)) {
        val = levels(D[,zv])[i]
        pch = pchs[i]
        if (i==1) {
            with(D[D[,zv]==val,], plot(
                grade, dv, type='b', pch=pch,
                xaxt="n", xlab=xlab,
                yaxt="n", ylab=ylab, ylim=ylim
                ))
        } else {
            with(D[D[,zv]==val,], points(
                grade, dv, type='b', pch=pch))
        }
    }

    # title
    title( main=main, adj=0 )

    # y axis
    axis(2, at=yat, labels=yatl, las=1)

    # x axis
    par( mgp=x_mgp )
    axis(1, at=xat, labels=xatl)
    title( xlab=xlab )

    # legend
    legend(legpos, legend=zlab, pch=pchs)
}

plotSDperf = function(dv='acc', D=SD.simu) {
    if (dv=="acc") {
        D = subset(D, select=c(grade, operation, acc))
        D$grade = as.numeric(as.character(D$grade))
        D = collapseOver(D, 'acc')
        D$dv = D[,dv]
        main = "(A)"
        ylab = "Percent Correct"
        ylim = c(.75,1)
        yat  = seq(.75, 1, .05)
        yatl = paste0(100*yat, '%')
    } else if (dv=="retrieve") {
        D = subset(D, select=c(grade, operation, retrieve))
        D$grade = as.numeric(as.character(D$grade))
        D = collapseOver(D, dv)
        D$dv = D[,dv]
        main = "(B)"
        ylab = "Percent Retrieval"
        ylim = c(.5,1)
        yat  = seq(.50, 1, .10)
        yatl = paste0(100*yat, '%')
    }
    zv   = "operation"
    zlab = c("Addition", "Multiplication")
    pchs = c(3,4)
    legpos = "bottomright"
    pct  = TRUE
    xv   = 'grade'
    xlab = "Grade"
    xat  = as.numeric(as.character(levels(SD.simu$grade)))
    xatl = as.character(levels(SD.simu$grade))
    # print(tapply(D$dv, list(D$operation, D$grade), mean))
    # flush.console()
    myLinePlot(
        D, dv, main, 
        ylab, ylim, yat, yatl, pct,
        xv, xlab, xat, xatl,
        zv, zlab, pchs, legpos)
}

plotSDretr = function(p='rt_mu', main="(A)", D=SD.simu) {
    D = D[,c('operation', 'grade',p,'retrieve')]
    D$grade = as.numeric(as.character(D$grade))
    D$dv = D[,'retrieve']
    D$retrieve = NULL
    if (p%in%c("d", "rt_mu", "ice")) {
        D[,p] = factor(D[,p], levels=sort(unique(as.numeric(as.character(D[,p])))))
    }
    if (p%in%c("g_group", "d", "ice")) {
        # show "best" levels on top
        D[,p] = factor(D[,p], levels=rev(levels(D[,p])))
    }
    E = collapseOver(subset(D, operation=="mul", select=-operation), 'dv')
    D = collapseOver(subset(D, operation=="add", select=-operation), 'dv')
    ylab = "Percent Retrieval"
    ylim = c(.2,1)
    yat  = seq(.2, 1, .2)
    yatl = paste0(100*yat, '%')
    zv   = p
    zlab = paste0(ifelse(p=="g_group", "g", p), " = ", as.character(levels(D[,p])))
    pchs = c(0,1,5,6,2)
    legpos = "bottomright"
    pct  = TRUE
    xv   = 'grade'
    xlab = "Grade"
    xat  = as.numeric(as.character(levels(SD.simu$grade)))
    xatl = as.character(levels(SD.simu$grade))
    myLinePlot(
        D, dv, main, 
        ylab, ylim, yat, yatl, pct,
        xv, xlab, xat, xatl,
        zv, zlab, pchs, legpos)

    # plot
    L = levels(E[,zv])
    for (i in 1:length(L)) {
        val = levels(E[,zv])[i]
        pch = pchs[i]
        with(E[E[,zv]==val,], points(
            grade, dv, type='b', pch=pch, lty='dashed'))
    }
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

png( paste0(graphics_path,"Figure 1 SD acc ret.png"), width=6.5, height=3, units='in', res=R, pointsize=s )
par(mfrow=c(1,2))
plotSDperf("acc")
plotSDperf("retrieve")
dev.off()

# Types of Errors

analyzeSDerrors(SD.simu, 'add')

analyzeSDerrors(SD.simu, 'mul')

getRespFreqs(subset(SD.simu, acc==0), "6*5")

# Influences on Problem Difficulty

analyzeSizeEffect(SD.simu, 'add', FALSE) # TRUE to add grade to regression

analyzeSizeEffect(SD.simu, 'mul', FALSE) # TRUE to add grade to regression

# Individual Differences in Strategy Use

analyzeSDstrategies(SD.simu, FALSE)

png( paste0(graphics_path,"Figure 2 SD add ret by params.png"), width=6.5, height=6, units='in', res=R, pointsize=s )
par(mfrow=c(2,2))
plotSDretr('g_group', main="(A)")
plotSDretr('d', main="(B)")
plotSDretr('rt_mu', main="(C)")
plotSDretr('ice', main="(D)")
dev.off()

D = collapseOver(droplevels(subset(SD.simu, grade==1 & operation=="add"))[,c('subjid', names(var_pars), 'acc', 'retrieve')], c('acc', 'retrieve'))
D$acc_bin = factor(ifelse(D$acc>=mean(D$acc), "high", "low"), levels=c("high", "low"))
D$ret_bin = factor(ifelse(D$retrieve>=mean(D$retrieve), "high", "low"), levels=c("high", "low"))
D$group = with(D, factor(ifelse(acc_bin=="high", ifelse(ret_bin=="high", "Good", "Perf"), ifelse(ret_bin=="high", "None", "NSG")), levels=c("Good", "NSG", "Perf", "None")))
T = data.frame(table(D$group))
names(T) = c('group', 'freq')
T$freq = T$freq/sum(T$freq)
T = T[with(T, order(group)),]
D = D[,c('group', 'acc', 'retrieve', names(var_pars))]
D = collapseOver(D, setdiff(names(D), 'group'))
D = join(T,D)
D

### Study 2. Fraction arithmetic

## Accuracy
D = FA.simu
D$operation = with(D, factor(ifelse(operation%in%c("add", "sub"), "add_sub", as.character(operation)), levels=c("add_sub", "mul", "div")))
analyzeRAaccuracies(D)
analyzeRAaccuracies(FA.data)

D = subset(FA.simu, select=c(operation, operands, prob, acc))
D = D[with(D, order(operation, operands, prob)),]
collapseOver(D, 'acc')

D = subset(FA.data, select=c(operation, operands, prob, acc))
D = D[with(D, order(operation, operands, prob)),]
collapseOver(D, 'acc')


## Errors
analyzeRAerrors(FA.simu)

## Strategies
analyzeRAstrategies(FA.simu, detailed)


### Study 3. Decimal arithmetic

## Accuracy
analyzeRAaccuracies(DA.simu)
D = DA.simu
D$operands = with(D, factor(ifelse(operands=="WD", "WD", "DD"), levels=c("DD", "WD")))
analyzeRAaccuracies(D)
analyzeRAaccuracies(DA.data)

with(DA.simu, tapply(acc, operation, mean))
with(DA.simu, tapply(acc, list(g, operation), mean))
with(DA.simu, tapply(acc, list(es, operation), mean))

## Errors
analyzeRAerrors(DA.simu)

## Strategies

analyzeRAstrategies(DA.simu, detailed)


### Study 4. Relations Among Basic and Advanced Arithmetic Skills

analyzeCorrelations(detailed)

### Graphics

plotFAacc = function(src='data') {
    if (src=='data') {
        load("..\\..\\..\\1.4 Datasets\\Braithwaite Pyke Siegler 2017 FARRA\\frac_arith_exp_data.RData")
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

png( paste0(p,"Figure 3 FA acc.png"), width=6.5, height=3, units='in', res=R, pointsize=s )
par(mfrow=c(1,2))
plotFAacc('data')
plotFAacc('simu')
dev.off()

png( paste0(p,"Figure 4 DA acc.png"), width=6.5, height=3, units='in', res=R, pointsize=s )
par(mfrow=c(1,2))
plotDAacc('data')
plotDAacc('simu')
dev.off()


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