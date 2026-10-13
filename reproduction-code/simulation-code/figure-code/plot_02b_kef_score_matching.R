#!/usr/bin/env Rscript
## Plot the optional 1D KEF score-matching study at its nominal cutoff.
## Input rejection_rate is the raw proportion with p <= alpha. This script
## performs no method-specific size calibration.

script_argument <- grep("^--file=", commandArgs(FALSE), value = TRUE)
if (length(script_argument) != 1L)
  stop("run with Rscript so the sibling plotting template can be found", call. = FALSE)
script_directory <- dirname(normalizePath(sub("^--file=", "", script_argument[[1L]])))
source(file.path(script_directory, "_paper_plot.R"))

arguments <- commandArgs(trailingOnly = TRUE)
if (!length(arguments) %in% c(3L, 4L))
  stop("Usage: Rscript plot_02b_kef_score_matching.R summary.csv figure.pdf figure.png [alpha]",
    call. = FALSE)
alpha <- if (length(arguments) == 4L) suppressWarnings(as.numeric(arguments[[4L]])) else 0.05
if (length(alpha) != 1L || !is.finite(alpha) || alpha <= 0 || alpha >= 1)
  stop("alpha must lie strictly between zero and one", call. = FALSE)

data <- read.csv(normalizePath(arguments[[1L]], mustWork = TRUE),
  stringsAsFactors = FALSE, check.names = FALSE)
required <- c("setup", "signal", "method", "rejection_rate")
missing <- setdiff(required, names(data))
if (length(missing))
  stop("summary.csv is missing columns: ", paste(missing, collapse = ", "), call. = FALSE)
if (!nrow(data)) stop("summary.csv contains no numerical results", call. = FALSE)

setup_levels <- c("p1", "p2", "p3")
method_levels <- c("SKSD-IA", "SKSD-NWB")
if (anyNA(data$setup) || !setequal(unique(data$setup), setup_levels))
  stop("summary.csv must contain exactly p1, p2, and p3 setups", call. = FALSE)
if (anyNA(data$method) || !setequal(unique(data$method), method_levels))
  stop("summary.csv must contain exactly SKSD-IA and SKSD-NWB", call. = FALSE)
if (!is.numeric(data$signal) || any(!is.finite(data$signal)) || any(data$signal < 0))
  stop("summary.csv contains invalid signals", call. = FALSE)
if (!is.numeric(data$rejection_rate) || any(!is.finite(data$rejection_rate)) ||
    any(data$rejection_rate < 0 | data$rejection_rate > 1))
  stop("summary.csv contains invalid rejection rates", call. = FALSE)

if ("rejections" %in% names(data) || "replications" %in% names(data)) {
  if (!all(c("rejections", "replications") %in% names(data)))
    stop("rejections and replications must be supplied together", call. = FALSE)
  if (!is.numeric(data$rejections) || !is.numeric(data$replications) ||
      any(!is.finite(data$rejections)) || any(!is.finite(data$replications)) ||
      any(data$replications < 1 | data$replications != floor(data$replications)) ||
      any(data$rejections < 0 | data$rejections > data$replications |
          data$rejections != floor(data$rejections)) ||
      any(abs(data$rejections / data$replications - data$rejection_rate) > 1e-10))
    stop("summary.csv contains invalid or inconsistent rejection counts", call. = FALSE)
}

signal_grid <- sort(unique(data$signal[data$setup == setup_levels[[1L]]]))
if (length(signal_grid) < 2L)
  stop("summary.csv needs at least two signal points per panel", call. = FALSE)
for (setup in setup_levels) {
  if (!identical(sort(unique(data$signal[data$setup == setup])), signal_grid))
    stop("the three panels must have the same signal grid", call. = FALSE)
}
if (anyDuplicated(data[c("setup", "signal", "method")]))
  stop("summary.csv contains duplicate setup/signal/method rows", call. = FALSE)
expected_rows <- length(setup_levels) * length(signal_grid) * length(method_levels)
if (nrow(data) != expected_rows)
  stop("summary.csv has incomplete setup/signal/method rows", call. = FALSE)

panel_labels <- c(
  p1 = '1~"fitted coefficient:"~phi[0]',
  p2 = '2~"fitted coefficients:"~phi[0]*","~phi[1]',
  p3 = '3~"fitted coefficients:"~phi[0]*","~phi[1]*","~phi[2]'
)
data$panel <- factor(unname(panel_labels[data$setup]), levels = unname(panel_labels))
data$method <- factor(data$method, levels = method_levels)

## Keep the shared Figure 1 font, color, strip, line, and legend conventions.
grDevices::pdf(file = NULL)
on.exit(grDevices::dev.off(), add = TRUE)
power <- ggplot(data, aes(signal, rejection_rate, colour = method, group = method)) +
  geom_hline(yintercept = alpha, linetype = "dashed", colour = "red") +
  geom_line(linewidth = style_number("line_width") / ggplot2::.pt) +
  geom_point(size = style_number("marker_size") / ggplot2::.pt) +
  facet_wrap(~ panel, nrow = 1L, scales = "fixed", labeller = label_parsed) +
  scale_x_continuous(breaks = signal_grid, limits = range(signal_grid)) +
  scale_y_continuous(limits = c(0, 1), breaks = c(0, 0.25, 0.5, 0.75, 1)) +
  scale_colour_manual(values = method_colors, breaks = method_levels, drop = FALSE) +
  labs(x = "Signal magnitude", y = "Rejection rate", colour = NULL) +
  paper_theme()
legend_plot <- power + theme(legend.position = "bottom", legend.box = "horizontal",
  legend.title = element_blank(),
  legend.text = element_text(size = style_number("legend_font_size"))) +
  guides(colour = guide_legend(nrow = 1L, byrow = TRUE,
    override.aes = list(linewidth = style_number("line_width") / ggplot2::.pt)))
legend <- cowplot::get_plot_component(legend_plot, "guide-box-bottom", return_all = FALSE)
title <- sprintf("Nuisance estimator: score matching (nominal %g%% cutoff)", 100 * alpha)
title_plot <- cowplot::ggdraw() + cowplot::draw_label(title, x = 0.5,
  hjust = 0.5, size = style_number("figure_title_font_size"))
figure <- cowplot::plot_grid(title_plot, power, legend, ncol = 1L,
  rel_heights = c(1.1, 11, 1.6))
figure <- cowplot::ggdraw() + cowplot::draw_plot(figure,
  x = 0.02, y = 0.02, width = 0.96, height = 0.96)

for (path in arguments[2:3]) {
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  ggsave(path, figure, width = style_number("figure_width"),
    height = style_number("figure_height"), dpi = style_number("dpi"), bg = "white")
  message("wrote ", path)
}
