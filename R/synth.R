# functions to generate synthetic test data for driving BrainSignals
# in order to fit its behaviour to simpler forms for the BSRF models

# as far as possible I shall try to make these generic, so as to be
# usable with other models, but the default args will produce a
# "standard" BrainSignals configuration


# simple scaling of a signal into a range
rescale <- function ( x, lo=0, hi=1 )
{
	if ( is.null(lo) || is.null(hi) )
	{
		return(x)
	}
	
	return ( lo + (hi - lo) * (x-min(x)) / (max(x)-min(x)) )
}

# our standard signals are all 600 seconds at 10 Hz sample rate
TT <- 0.1 * 1:6000

# basic elementary signals
noise <- function ( tt=TT, mean=0, sd=1, lo=NULL, hi=NULL )
{
	invisible(rescale(rnorm(length(tt), mean, sd), lo, hi))
}

sine <- function ( tt=TT, f=1, phi=0, lo=NULL, hi=NULL )
{
	invisible(rescale(sin(tt * 2 * pi * f + phi), lo, hi))
}

square <- function ( tt=TT, f=1, duty=0.5, phi=0, lo=NULL, hi=NULL )
{
	# special case duty limits, which aren't well catered for by the cosine hack below
	if ( duty <= 0 )
	{
		invisible(rep(ifelse(is.null(lo), 0, lo[1]), length(tt)))
	}
	else if ( duty >= 1 )
	{
		invisible(rep(ifelse(is.null(hi), 1, hi[1]), length(tt)))
	}
	else
	{
		invisible(rescale(as.numeric(cos(tt * 2 * pi * f + phi + duty * pi) >= cos(duty * pi)), lo, hi))
	}
}

# sawtooth wave -- unlike the python code, we don't support other shapes here
saw <- function ( tt=TT, f=1, phi=0, lo=0, hi=1 )
{
	# for consistency with above, phi is in radians
	# but here we just care about cycles, so convert
	tx <- tt + phi/(2 * pi * f)
	
	# just divide time into cycles and rescale
	invisible(rescale(tx %% (1/f), lo, hi))
}

walk <- function ( tt=TT, mean=0, sd=0.1, lo=NULL, hi=NULL )
{
	invisible(rescale(cumsum(rnorm(length(tt), mean, sd)), lo, hi))
}

# trivial constant signal
const <- function ( tt=TT, value=0 )
{
	invisible(rep(value, length(tt)))
}

# in this case we just rescale tt, so ramp will be irregular if sample times are
ramp <- function ( tt=TT, lo=0, hi=1, descending=FALSE )
{
	invisible( rescale(ifelse(descending, -1, 1) * tt, lo, hi) )
}

# the multi-osc sim shape from Felix
nirs.mix <- function ( tt=TT, lo=0, hi=1 )
{
	invisible( rescale( sine(tt, f=1, lo=-0.6, hi=-0.6)
						+ sine(tt, f=0.25, lo=-0.2, hi=0.2)
						+ sine(tt, f=0.1, lo=-0.9, hi=0.9)
						+ sine(tt, f=0.04, lo=-1, hi=1)
						+ noise(tt, sd=0.053),
						lo, hi) )
}

# function returning a matrix of signals of various types
# all using the same timebase and scaled into the same range
signals <- function ( tt=TT, lo=0, hi=1, short=FALSE )
{
	result <- matrix(0, nrow=ifelse(short,4,8), ncol=length(tt))
	
	result[1,] <- sine(tt, f=0.01, lo=lo, hi=hi)
	result[2,] <- noise(tt, lo=lo, hi=hi)
	result[3,] <- square(tt, f=0.01, duty=0.2, lo=lo, hi=hi)
	result[4,] <- nirs.mix(tt, lo=lo, hi=hi)
  
	
	if ( !short )
	{
		result[5,] <- (lo + hi) / 2
		result[6,] <- walk(tt, lo=lo, hi=hi)
		result[7,] <- saw(tt, f=0.1, phi=pi, lo=lo, hi=hi)
		result[8,] <- ramp(tt, lo=lo, hi=hi)
		
		rownames(result)<-c("Sine","Noise","Square","NIRS Mix",
		                         "Constant Mean","Walk","Saw","Ramp")
	}
	if (short){
	  rownames(result)<-c("Sine","Noise","Square","NIRS Mix")
	}
	invisible(result)
}

# signal sets for some vaguely useful parameter ranges
pa.bs.good <- signals(TT, lo=80, hi=100)
pa.bs.bad <- signals(TT, lo=40, hi=150)
pa.pre.good <- signals(TT, lo=20, hi=40)
pa.pre.bad <- signals(TT, lo=8, hi=70)
pa.term.good <- signals(TT, lo=35, hi=55)
pa.term.bad <- signals(TT, lo=10, hi=80)

co2.good <- signals(TT, lo=37, hi=45)
co2.bad <- signals(TT, lo=30, hi=50)
o2.good <- signals(TT, lo=0.9, hi=1)
o2.bad <- signals(TT, lo=0.6, hi=1)
u.good <- signals(TT, lo=0.8, hi=1.2)
u.bad <- signals(TT, lo=0.5, hi=1.5)

# function generating data frames of input values for combos of input signals
# defaults to standard BrainSignals inputs over "reasonable" ranges
# (this produces a lot of combos, many uninteresting -- may want to reduce this)
combos <- function ( tt=TT,
					 fields=list(SaO2sup=o2.good, Pa_CO2=co2.good, P_a=pa.bs.good) )
{
	keys <- names(fields)
	
	# in calculating the end of the last step we sort-of assume
	# that the steps are all the same length
	ends <- c(tt[-1], tt[length(tt)] + diff(tt[1:2]))
	
	# start with just the time base, which is the same for all
	tables <- list(data.frame(t0=tt, t1=ends))
	
	for ( name in keys )
	{
		M <- fields[[name]]
		nsigs <- dim(M)[1]
		new.tables <- vector("list", length(tables) * nsigs)
		
		tabnum <- 1

		for ( ii in 1:nsigs )
		{
			newcol <- data.frame(M[ii,])
			colnames(newcol) <- name
			
			for ( prev in tables )
			{
				new.tables[[tabnum]] <- cbind(prev, newcol)
				tabnum <- tabnum + 1
			}
		}
		
		tables <- new.tables
	}
	
	invisible(tables)
}

abbrev.fields <- list(SaO2sup=signals(TT, lo=0.8, hi=1, short=TRUE),
                      Pa_CO2=signals(TT, lo=37, hi=43, short=TRUE),
                      P_a=signals(TT, lo=80, hi=120, short=TRUE),
                      u=signals(TT, lo=0.5, hi=1.5, short=TRUE))

# dump a data frame (as generated above) to a BCMD input file
# first two columns are assumed to be start and end times for each step
# init specifies whether to initialise data params before equilibration
# steady specifies how long to equilibrate for (if NULL or < 0 then don't equilibrate)
write.input <- function ( xx, filename, init=TRUE, steady=1000 )
{
	ff <- file(filename, "wt")
	cat("# BCMD input file generated from R by write.input\n", file=ff)
	
	M <- as.matrix(xx)
	fields <- names(xx)[-(1:2)]
	
	if ( !is.null(steady) && steady > 0 )
	{
		cat("@", dim(M)[1] + 1, "\n", sep=" ", file=ff)
		cat("# disable output for equilibration\n", file=ff)
		cat("!0\n>>> 0\n", file=ff)
		
		if ( init )
		{
			cat(":", dim(M)[2] - 2, fields, "\n", sep=" ", file=ff)
			cat("=", M[1,1]-steady, M[1,1], M[1,-(1:2)], "\n", sep=" ", file=ff)
		}
		else
		{
			cat(": 0\n", file=ff)
			cat("=", M[1,1]-steady, M[1,1], "\n", file=ff)
		}
		
		cat("# restore output\n", file=ff)
		cat("!!!\n>>> *\n", file=ff)
		
		if ( ! init )
		{
			cat(":", dim(M)[2] - 2, fields, "\n", sep=" ", file=ff)
		}
	}
	else
	{
		cat("@", dim(M)[1], "\n", sep=" ", file=ff)		
		cat(":", dim(M)[2] - 2, fields, "\n", sep=" ", file=ff)
	}
	
	for ( ii in 1:(dim(M)[1]) )
	{
		cat("=", M[ii,], "\n", sep=" ", file=ff)
	}
	
	close(ff)
}

# do the same for a list of jobs
write.combos <- function ( xx, basename="BS_combo", init=TRUE, steady=1000 )
{
	for ( ii in 1:length(xx) )
	{
		write.input( xx[[ii]], sprintf("%s_%04d.input", basename, ii), init, steady )
	}
}

# utility to load all the simulated data from a bunch of .out files
# can optionally specify columns to include or discard
# (if both are args are provided, discard is ignored)
# if merging, we assume the columns wind up the same for all files
read.sims <- function ( path=".",
						columns=NULL,
						discard=c("ERR", "Vmca_d", "TOI_d", "HbT_d", "CCO_d"),
						merge=TRUE )
{
	tables <- list()
	src <- list.files(path, pattern="\\.out$", full.names=TRUE)
	
	for ( ss in src )
	{
		tab <- read.table(ss, header=TRUE)
		
		if ( ! is.null(columns) )
		{
			tab <- tab[,intersect(names(tab), columns)]
		}
		else if ( !is.null(discard) )
		{
			tab <- tab[,setdiff(names(tab), discard)]
		}
		
		tables[[basename(ss)]] <- tab
	}
	
	if ( merge )
	{
		result <- NULL
		for ( nn in names(tables) )
		{
			tab <- tables[[nn]]
			tab <- cbind( data.frame(Simulation=rep(nn, dim(tab)[1])), tab)
			result <- rbind(result, tab)
		}
		
		invisible(result)
	}
	else
	{
		invisible(tables)
	}
}

read.eFast <- function ( path=".")
{
  tables <- list()
  src <- list.files(path, pattern="\\.csv$", full.names=TRUE)
  
  for ( ss in src )
  {
    tab <- read.csv(ss, header=TRUE)
    for ( ii in seq(65)){
      
      tables[[basename(ss)]] <- tab
    }
  }
  
  
  invisible(tables)
  
}

# export selected columns from data frame to a BCMD input file
#
# this is very similar to write.input in synth.R, with the following modifications:
# - only a single time column is expected, rather than start and end times
# - the time column is explicitly specified
# - a subset of columns can be chosen
# - columns from the data can be mapped to different names in the model
#
# the first specified column is taken to be time
# outnames specifies the parameter names to be used in the file
# -- the first name is ignored, since time is part of the file syntax, not a parameter
# init specifies whether to initialise data params before equilibration
# steady specifies how long to equilibrate for (if NULL or < 0 then don't equilibrate)
write.selected <- function ( xx, filename, cols=names(xx), outnames=cols, init=TRUE, steady=1000 )
{
  t0 <- xx[,cols[1]]
  N <- length(t0)
  
  # we assume that the last step is the same length as the one before it
  t1 <- c(t0[-1], t0[N] + t0[N] - t0[N-1])
  
  params <- outnames[-1]
  P <- length(params)
  
  M <- as.matrix(xx[,cols[-1]])
  
  ff <- file(paste(filename,".input", sep=""), "wt")
  cat("# BCMD input file generated from R by write.selected\n", file=ff)
  
  
  if ( !is.null(steady) && steady > 0 )
  {
    cat("@", N + 1, "\n", sep=" ", file=ff)
    cat("# disable output for equilibration\n", file=ff)
    cat("!0\n>>> 0\n", file=ff)
    
    if ( init )
    {
      cat("# equilibrate to initial parameter values\n", file=ff)
      cat(":", P, params, "\n", sep=" ", file=ff)
      cat("=", t0[1]-steady, t0[1], M[1,], "\n", sep=" ", file=ff)
    }
    else
    {
      cat("# equilibrate to default parameter values\n", file=ff)
      cat(": 0\n", file=ff)
      cat("=", t0[1]-steady, t0[1], "\n", file=ff)
    }
    
    cat("# restore output\n", file=ff)
    cat("!!!\n>>> *\n", file=ff)
    
    if ( ! init )
    {
      cat(":", P, params, "\n", sep=" ", file=ff)
    }
  }
  else
  {
    cat("@", N, "\n", sep=" ", file=ff)		
    cat(":", P, params, "\n", sep=" ", file=ff)
  }
  
  for ( ii in 1:N )
  {
    cat("=", t0[ii], t1[ii], M[ii,], "\n", sep=" ", file=ff)
  }
  
  close(ff)
}

table.gen<-function(params,t,par.vals){
  #parameters = vector of parameters as strings
  #t = vector of time steps t_n long
  #par.vals = matrix of parameter values at each time step.
  #           Should be t_n by len(params)
  if(missing(t)||missing(params)||missing(par.vals)){
    stop("Please check provided parameters.")
  }
  
  if(dim(par.vals)[1]!=length(t)){
    stop("Please provide parameter values for exactly each time step.")
  }
  
  if(dim(par.vals)[2]!=length(params)){
    stop("Please provide values for exactly each named parameter.")
  }
  
  
  input.table <- data.frame(t,par.vals)
  colnames(input.table)[-1]<-c(params)
  return(input.table)
}

t=c(0,100,200)

input.norm <- table.gen(params=c("u","Xtot","Xtot_n"), t=c(0,100,200),
                        par.vals=matrix(c(1,1.2,1,rep(7.8,length(t)),rep(7.8,length(t))),
                               nrow=(length(t)),ncol=3))
write.csv(input.norm, file = "~/repos/bcmd/MiniProj1/inputNorm.csv", row.names= FALSE)
write.selected(input.norm,"~/repos/bcmd/MiniProj1/norm")

input.mild <- table.gen(params=c("u","Xtot","Xtot_n"), t=c(0,100,200),
                        par.vals=matrix(c(1,1.2,1,rep(6.5,length(t)),rep(6.5,length(t))),
                                        nrow=(length(t)),ncol=3))
write.csv(input.mild, file = "~/repos/bcmd/MiniProj1/inputMild.csv", row.names= FALSE)
write.selected(input.mild,"~/repos/bcmd/MiniProj1/mild")

input.moderate <- table.gen(params=c("u","Xtot","Xtot_n"), t=c(0,100,200),
                        par.vals=matrix(c(1,1.2,1,rep(5.5,length(t)),rep(5.5,length(t))),
                                        nrow=(length(t)),ncol=3))
write.csv(input.moderate, file = "~/repos/bcmd/MiniProj1/inputMod.csv", row.names= FALSE)
write.selected(input.moderate,"~/repos/bcmd/MiniProj1/moderate")

input.severe <- table.gen(params=c("u","Xtot","Xtot_n"), t=c(0,100,200),
                        par.vals=matrix(c(1,1.2,1,rep(3.5,length(t)),rep(3.5,length(t))),
                                        nrow=(length(t)),ncol=3))
write.csv(input.severe, file = "~/repos/bcmd/MiniProj1/inputSev.csv", row.names= FALSE)
write.selected(input.severe,"~/repos/bcmd/MiniProj1/severe")

v = c(1,10)
for (ii in seq(2,1,length.out=100)){
     v = c(v,1,ii)
}
for (ii in seq(1.1,2,length.out=100)){
    v = c(v,1,ii)
}
t = seq(0,by=100, length.out=length(v))
input.step <- table.gen(params=c("u"), t,
                          par.vals=matrix(v,nrow=(length(t)),ncol=1))
write.csv(input.step, file = "~/repos/bcmd/MiniProj1/inputStep_reduced.csv", row.names= FALSE)
write.selected(input.step,"~/repos/bcmd/MiniProj1/Step_reduced.input")