#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 3) {
  stop("usage: Rscript convert_data.R Y_LUAD.rds X_LUAD.rds output.csv")
}

y <- readRDS(args[[1]])
x <- readRDS(args[[2]])
if (!is.data.frame(y) || !is.data.frame(x)) {
  stop("Both RDS objects must be data.frames")
}
if (!identical(dim(y), c(360L, 41L)) || !identical(dim(x), c(360L, 10L))) {
  stop("Unexpected LUAD dimensions")
}

y_participant <- substr(rownames(y), 1, 12)
x_participant <- substr(rownames(x), 1, 12)
if (anyDuplicated(y_participant) || anyDuplicated(x_participant)) {
  stop("LUAD participant identifiers are not unique")
}
match_index <- match(y_participant, x_participant)
if (anyNA(match_index) || !setequal(y_participant, x_participant)) {
  stop("RPPA and RNA participant identifiers do not match")
}
x <- x[match_index, , drop = FALSE]
if (!identical(y_participant, substr(rownames(x), 1, 12))) {
  stop("Explicit participant join failed")
}

required_x <- c("T cells", "Monocytes", "Neutrophils")
if (!all(required_x %in% colnames(x))) {
  stop("Required immune signatures are missing")
}
if (any(!is.finite(as.matrix(y))) || any(!is.finite(as.matrix(x)))) {
  stop("Non-finite values found")
}

output <- data.frame(
  participant_id = y_participant,
  rppa_barcode = rownames(y),
  rna_barcode = rownames(x),
  t_cell_score = x[, "T cells"],
  monocyte_score = x[, "Monocytes"],
  neutrophil_score = x[, "Neutrophils"],
  check.names = FALSE,
  stringsAsFactors = FALSE
)
output <- cbind(output, y)
dir.create(dirname(args[[3]]), recursive = TRUE, showWarnings = FALSE)
write.csv(output, args[[3]], row.names = FALSE, quote = TRUE)
message("Wrote ", nrow(output), " joined LUAD patients to ", args[[3]])
