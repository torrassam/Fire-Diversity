{ # big problem: non mi fa scaricare correttamente i pacchetti R, stesso problema che mi dava quest'estate ma non ricordo come modificarlo
    library("Rcpp")
    library("hypervolume")
    library("BAT")
    # Setting the working directory
    setwd("")

    #Create a Empty DataFrame with 0 rows and n columns
    columns <- c("N", "ncom", "alt_stat", "frt", "srichness", "isimpson", "fd_rich", "fd_div")
    dfout <- data.frame(matrix(nrow = 0, ncol = length(columns)))
    colnames(dfout) <- columns

    # write the new data in a .csv files
    write.csv(dfout, "temp.csv", row.names = FALSE)

    # Estimating the hypervolumes and the functional diversity metrics of the communities
    datafile <- read.csv("coms-fire-bioindex.csv")
    # dfin <- datafile
    dfin <- datafile[datafile$N == 10, ]
    n <- nrow(dfin)

    for (i in 1:n) {

        dfout <- read.csv("temp.csv")

        sr <- as.numeric(dfin[i, "srichness"])
        if (sr > 1) {

        biome <- dfin[i, "biome"]
        NP <- dfin[i, "N"]
        X <- dfin[i, "ncom"]
        AT <- dfin[i, "alt_stat"]

        fname <- paste("comp10/coms-n", NP, "-", biome, "-", X, "-", AT, ".csv", sep = "")
        df <- read.csv(fname)

        # df$C <- log(df$C)
        df$I <- log(df$I)

        # Select the dimension to compute the hypervolume on
        data_no_a <- as.data.frame(na.omit(df))
        data_c <- data_no_a[c("I", "C", "R", "L")]
        hv <- hypervolume(data_c, method = "gaussian")

        # Functional Diversity metrics
        fd_rich <- kernel.alpha(hv)
        fd_div <- kernel.dispersion(hv)
        fd_reg <- kernel.evenness(hv)

    } else {

# Functional Diversity metrics set to zero in case it is a single-standing species system
        fd_rich <- 0.0
        fd_div <- 0
        fd_reg <- 0.0
        }

        # Save the results in a new dataframe built on the input one
        dfi <- dfin[i,]
        dfi$fd_rich <- fd_rich
        dfi$fd_div <- fd_div
        dfi$fd_reg <- fd_reg

        # write the new data in a .csv file
        dfout <- rbind(dfout,dfi)
        write.csv(dfout, "temp.csv", row.names = FALSE)
    }

}