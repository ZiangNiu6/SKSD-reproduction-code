#!/usr/bin/env Rscript

## Plot the TCGA-LUAD model evaluation using the manuscript simulation style.

suppressPackageStartupMessages({
  library(ggplot2)
  library(cowplot)
})

arguments <- commandArgs(trailingOnly = TRUE)
if (length(arguments) < 1L || length(arguments) > 2L) {
  stop(
    "Usage: plot_tcga_rppa.R OUTPUT_DIR [COMPARISON_OUTPUT_DIR]",
    call. = FALSE
  )
}

output_dir <- normalizePath(arguments[[1L]], mustWork = TRUE)
comparison_output_dir <- if (length(arguments) == 2L) {
  normalizePath(arguments[[2L]], mustWork = TRUE)
} else {
  NULL
}
read_checked_csv <- function(path, required_columns, label) {
  if (!file.exists(path)) stop("Missing ", label, ": ", path, call. = FALSE)
  data <- read.csv(path, stringsAsFactors = FALSE, check.names = FALSE)
  missing_columns <- setdiff(required_columns, names(data))
  if (nrow(data) == 0L || length(missing_columns) > 0L) {
    stop(
      label, " is empty or missing columns: ",
      paste(missing_columns, collapse = ", "),
      call. = FALSE
    )
  }
  data
}

model_levels <- c("independent", "shared", "varying")
numeric_columns <- c(
  "p_value_influence", "p_value_influence_holm",
  "p_value_brueck", "p_value_brueck_holm"
)
model_axis_labels <- c(
  independent = expression(M[0]),
  shared = expression(M[S]),
  varying = expression(M[V])
)

read_model_evaluation <- function(directory) {
  model_data <- read_checked_csv(
    file.path(directory, "model_evaluation.csv"),
    c("model", "edge_count", numeric_columns),
    "model_evaluation.csv"
  )
  if (anyDuplicated(model_data$model) ||
      any(!model_data$model %in% model_levels)) {
    stop(
      "model_evaluation.csv has unexpected or duplicated models",
      call. = FALSE
    )
  }
  if (any(!vapply(model_data[numeric_columns], is.numeric, logical(1))) ||
      any(!is.finite(as.matrix(model_data[numeric_columns]))) ||
      any(as.matrix(model_data[numeric_columns]) <= 0) ||
      any(as.matrix(model_data[numeric_columns]) > 1) ||
      !is.numeric(model_data$edge_count) ||
      any(!is.finite(model_data$edge_count)) ||
      length(unique(model_data$edge_count)) != 1L) {
    stop(
      "model_evaluation.csv contains invalid edge counts or p-values",
      call. = FALSE
    )
  }
  model_data$model <- factor(model_data$model, levels = model_levels)
  list(
    data = model_data,
    panel = sprintf("%d-edge support", unique(model_data$edge_count))
  )
}

model_evaluations <- list(read_model_evaluation(output_dir))
if (!is.null(comparison_output_dir)) {
  model_evaluations[[2L]] <- read_model_evaluation(comparison_output_dir)
}
panel_names <- vapply(model_evaluations, `[[`, character(1), "panel")
if (anyDuplicated(panel_names)) {
  stop("The requested model evaluations use the same edge support", call. = FALSE)
}
preferred_panel_order <- c("18-edge support", "45-edge support")
panel_levels <- c(
  intersect(preferred_panel_order, panel_names),
  setdiff(panel_names, preferred_panel_order)
)

p_value_spec <- data.frame(
  column = c(
    "p_value_influence", "p_value_influence_holm",
    "p_value_brueck", "p_value_brueck_holm"
  ),
  procedure = c(
    "Influence-adjusted V", "Influence-adjusted V",
    "Midpoint refitted U", "Midpoint refitted U"
  ),
  adjustment = c("Raw", "Holm", "Raw", "Holm"),
  stringsAsFactors = FALSE
)
make_p_value_data <- function(evaluation) {
  model_data <- evaluation$data
  do.call(
    rbind,
    lapply(seq_len(nrow(p_value_spec)), function(index) {
      data.frame(
        model = model_data$model,
        p_value = model_data[[p_value_spec$column[[index]]]],
        procedure = p_value_spec$procedure[[index]],
        adjustment = p_value_spec$adjustment[[index]],
        panel = evaluation$panel,
        stringsAsFactors = FALSE
      )
    })
  )
}
p_value_data <- do.call(rbind, lapply(model_evaluations, make_p_value_data))
p_value_data$model <- factor(p_value_data$model, levels = model_levels)
p_value_data$procedure <- factor(
  p_value_data$procedure,
  levels = c("Influence-adjusted V", "Midpoint refitted U")
)
p_value_data$adjustment <- factor(
  p_value_data$adjustment,
  levels = c("Raw", "Holm")
)
p_value_data$panel <- factor(p_value_data$panel, levels = panel_levels)
p_value_lower_limit <- min(0.025, min(p_value_data$p_value) / 2)

procedure_colors <- c(
  "Influence-adjusted V" = "#9467bd",
  "Midpoint refitted U" = "#ff7f0e"
)
procedure_labels <- c(
  "Influence-adjusted V" = "SKSD-IA",
  "Midpoint refitted U" = "Brück et al."
)

paper_theme <- theme_bw(base_size = 16) +
  theme(
    axis.line = element_line(colour = "black", linewidth = 0.4),
    axis.text = element_text(size = 14, colour = "black"),
    axis.title = element_text(size = 16),
    legend.box = "horizontal",
    legend.direction = "horizontal",
    legend.text = element_text(size = 13),
    legend.title = element_text(size = 13),
    panel.grid.minor = element_blank(),
    panel.grid.major.x = element_blank(),
    strip.background = element_rect(
      fill = "grey92", colour = "black", linewidth = 0.5
    ),
    strip.text = element_text(size = 16, face = "bold"),
    plot.margin = margin(3, 3, 3, 3)
  )

p_value_plot <- ggplot(
  p_value_data,
  aes(
    x = model,
    y = p_value,
    colour = procedure,
    shape = adjustment,
    group = interaction(procedure, adjustment)
  )
) +
  geom_hline(yintercept = 0.05, linetype = "dashed", colour = "red") +
  geom_point(
    position = position_dodge(width = 0.34),
    size = 3.2,
    stroke = 0.9
  ) +
  facet_wrap(~panel, nrow = 1L) +
  scale_x_discrete(labels = model_axis_labels, drop = FALSE) +
  scale_colour_manual(
    values = procedure_colors,
    labels = procedure_labels,
    drop = FALSE
  ) +
  scale_shape_manual(values = c(Raw = 16, Holm = 4), drop = FALSE) +
  scale_y_log10(
    limits = c(p_value_lower_limit, 1),
    breaks = c(0.0001, 0.001, 0.01, 0.05, 0.1, 1),
    labels = c("0.0001", "0.001", "0.01", "0.05", "0.1", "1"),
    expand = expansion(mult = c(0.03, 0.06))
  ) +
  labs(
    x = "Candidate model",
    y = "P-value (log scale)",
    colour = "Method",
    shape = "Adjustment"
  ) +
  paper_theme +
  theme(legend.position = "bottom") +
  guides(
    colour = guide_legend(order = 1, nrow = 1),
    shape = guide_legend(order = 2, nrow = 1)
  )

final_plot <- cowplot::ggdraw() +
  cowplot::draw_plot(p_value_plot, x = 0.015, y = 0.015, width = 0.97, height = 0.97)

pdf_path <- file.path(output_dir, "model_evaluation.pdf")
png_path <- file.path(output_dir, "model_evaluation.png")
ggsave(
  pdf_path,
  final_plot,
  width = 12.5,
  height = 4.8,
  device = grDevices::pdf,
  useDingbats = FALSE,
  bg = "white"
)
ggsave(
  png_path,
  final_plot,
  width = 12.5,
  height = 4.8,
  dpi = 180,
  bg = "white"
)

message("wrote ", pdf_path)
message("wrote ", png_path)
