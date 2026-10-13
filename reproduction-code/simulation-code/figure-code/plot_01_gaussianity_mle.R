#!/usr/bin/env Rscript
## Render saved simulation results using the shared KEF-style R template.
script_argument <- grep("^--file=", commandArgs(FALSE), value = TRUE)[[1L]]
script_directory <- dirname(normalizePath(sub("^--file=", "", script_argument)))
source(file.path(script_directory, "_paper_plot.R"))
paper_plot_main("01_gaussianity_mle")
