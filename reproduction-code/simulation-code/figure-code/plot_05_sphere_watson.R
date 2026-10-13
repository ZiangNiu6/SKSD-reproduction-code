#!/usr/bin/env Rscript
## Render the spherical appendix experiment using the main-text R template.
script_argument <- grep("^--file=", commandArgs(FALSE), value = TRUE)[[1L]]
script_directory <- dirname(normalizePath(sub("^--file=", "", script_argument)))
source(file.path(script_directory, "_paper_plot.R"))
paper_plot_main("05_sphere_watson", title_override = "Fixed axis: scalar score-matching estimator")
