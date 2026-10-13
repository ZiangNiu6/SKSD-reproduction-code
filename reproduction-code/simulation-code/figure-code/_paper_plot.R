## Shared R/ggplot2 template. Python produces results; R only reads and plots.
suppressPackageStartupMessages({ library(ggplot2); library(cowplot) })
paper_directory <- dirname(normalizePath(sys.frame(1)$ofile))
style_table <- read.csv(file.path(paper_directory, "_paper_plot_style.csv"), stringsAsFactors = FALSE)
paper_style <- setNames(style_table$value, style_table$key)
style_number <- function(key) as.numeric(paper_style[[key]])
method_colors <- c(
  "SKSD-PA" = "#1f77b4", "SKSD-IA" = "#9467bd",
  "SKSD-NWB" = "#8c564b",
  "SKS" = "#2ca02c", "SKS (Lilliefors)" = "#2ca02c",
  "SMMD (Key et al.)" = "#d62728", "Brück et al." = "#ff7f0e"
)

prepare_paper_data <- function(data, study) {
  columns <- switch(study,
    "01_gaussianity_mle" = c("alternative", "signal", "method", "mean_seconds"),
    "02_kef_minimum_ksd" = c("dimension", "theta2_base", "method", "display_method", "mean_seconds_per_replication", "nuisance_estimator"),
    "03_conditional_gaussian_score_matching" = c("alternative", "delta", "method", "mean_seconds"),
    "04_rbm_score_matching" = c("alternative", "signal", "method", "total_seconds_per_dataset"),
    "05_sphere_watson" = c("family", "dispersion_path", "sample_size", "kappa0", "signal", "method", "mean_total_seconds", "ci_low", "ci_high"),
    stop("unknown simulation study: ", study, call. = FALSE))
  missing <- setdiff(c(columns, "rejection_rate"), names(data))
  if (length(missing)) stop("summary.csv is missing columns: ", paste(missing, collapse = ", "), call. = FALSE)
  if (!nrow(data)) stop("summary.csv contains no numerical results", call. = FALSE)
  seconds_column <- intersect(columns, c("mean_seconds", "mean_seconds_per_replication", "total_seconds_per_dataset", "mean_total_seconds"))
  runtimes <- data[[seconds_column]]
  if (!is.numeric(runtimes) || any(!is.finite(runtimes)) || any(runtimes <= 0))
    stop("timing panels require finite, strictly positive runtimes", call. = FALSE)
  numeric_columns <- setdiff(c(columns, "rejection_rate"),
    c("alternative", "method", "display_method", "nuisance_estimator", "family", "dispersion_path"))
  if (any(!vapply(data[numeric_columns], function(x) is.numeric(x) && all(is.finite(x)), logical(1))))
    stop("summary.csv contains invalid numerical results", call. = FALSE)
  if (any(data$rejection_rate < 0 | data$rejection_rate > 1))
    stop("summary.csv contains invalid rejection rates", call. = FALSE)
  method_labels <- c("SKSD-parametric" = "SKSD-PA", "SKSD-influence" = "SKSD-IA",
    "SKSD-naive-wild" = "SKSD-NWB", "SKS" = "SKS (Lilliefors)",
    "SMMD" = "SMMD (Key et al.)", "Brueck-corrected" = "Brück et al.")
  x_label <- "Signal strength"
  title <- "Nuisance estimator: score matching"
  if (study == "02_kef_minimum_ksd") {
    if (any(data$dimension < 1 | data$dimension != floor(data$dimension)))
      stop("summary.csv contains invalid dimensions", call. = FALSE)
    if (anyNA(data$nuisance_estimator) || !all(data$nuisance_estimator == "minimum KSD"))
      stop("this plotter accepts only the minimum-KSD nuisance fit", call. = FALSE)
    ## Preserve KEF's publication crop; custom simulation grids remain complete.
    publication_grid <- c(-4, -3, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 3, 4)
    observed_grid <- sort(unique(round(data$theta2_base, 10)))
    if (isTRUE(all.equal(observed_grid, publication_grid, tolerance = 1e-8)))
      data <- data[abs(data$theta2_base) <= 1.5 + 1e-10, , drop = FALSE]
    panel_levels <- paste0("d = ", sort(unique(data$dimension)))
    panel <- paste0("d = ", data$dimension)
    signal <- data$theta2_base
    kef_method_labels <- method_labels
    kef_method_labels[["SKS"]] <- "SKS"
    method <- unname(kef_method_labels[data$method])
    method_levels <- unname(kef_method_labels[names(kef_method_labels) %in% data$method])
    seconds <- data$mean_seconds_per_replication
    x_label <- expression(theta[2] %*% sqrt(d))
    title <- "Nuisance estimator: minimum KSD"
    timing_label <- paste0("Timing (", panel_levels[[1L]], ")")
  } else if (study == "05_sphere_watson") {
    if (anyNA(data$family) || anyNA(data$dispersion_path) ||
        any(!data$family %in% c("bingham", "crossing")) ||
        any(!data$dispersion_path %in% c("fixed_kappa", "matched_odi")) ||
        any(data$family == "crossing" & data$dispersion_path != "fixed_kappa"))
      stop("summary.csv contains an unknown spherical design", call. = FALSE)
    if (any(data$sample_size < 2 | data$sample_size != floor(data$sample_size)) ||
        any(data$kappa0 <= 0) || any(data$signal < 0) ||
        any(data$signal[data$family == "bingham"] >= 1) ||
        any(data$signal[data$family == "crossing"] > 0.5))
      stop("summary.csv contains invalid spherical settings", call. = FALSE)
    if (any(data$ci_low < 0 | data$ci_high > 1 |
            data$ci_low > data$rejection_rate | data$ci_high < data$rejection_rate))
      stop("summary.csv contains invalid Monte Carlo intervals", call. = FALSE)
    allowed <- c("SKSD-parametric", "SKSD-influence", "Brueck-corrected")
    if (anyNA(data$method) || any(!data$method %in% allowed))
      stop("summary.csv contains an unknown spherical method", call. = FALSE)
    method_levels <- unname(method_labels[allowed[allowed %in% data$method]])
    method <- unname(method_labels[data$method])
    order_index <- order(match(data$family, c("bingham", "crossing")),
      match(data$dispersion_path, c("fixed_kappa", "matched_odi")),
      data$kappa0, data$sample_size)
    number_text <- function(x) format(x, trim = TRUE, scientific = FALSE)
    panel <- paste0("kappa[0] == ", number_text(data$kappa0))
    if (length(unique(data$sample_size)) > 1L)
      panel <- paste0("atop(", panel, ", n == ", number_text(data$sample_size), ")")
    if (any(data$family != "bingham" | data$dispersion_path != "fixed_kappa")) {
      scenario <- ifelse(data$family == "crossing", "Crossing fibers",
        ifelse(data$dispersion_path == "matched_odi", "Bingham: matched ODI", "Bingham: fixed kappa"))
      panel <- paste0('atop("', scenario, '", ', panel, ')')
    }
    panel_levels <- unique(panel[order_index])
    signal <- data$signal
    seconds <- data$mean_total_seconds
    x_label <- if (all(data$family == "bingham")) expression(r == beta / kappa) else
      if (all(data$family == "crossing")) expression(epsilon) else "Signal strength"
    title <- "Nuisance estimator: principal axis and score matching"
    timing_label <- paste0('paste("Timing (", ', panel_levels[[1L]], ', ")")')
  } else {
    panel_labels <- switch(study,
      "01_gaussianity_mle" = c("t" = "Student-t", "mixture" = "Gaussian mixture", "gen_chi" = "Generalized chi-square"),
      "03_conditional_gaussian_score_matching" = c("rewire" = "Second-neighbor rewiring", "community" = "Within-block triadic closure", "matching" = "Diametric matched pairs"),
      "04_rbm_score_matching" = c("loading-rotation" = "Loading-direction rotation", "omitted-hidden-unit" = "Omitted hidden unit", "heavy-tailed-visible" = "Heavy-tailed visible noise"))
    if (anyNA(data$alternative) || !all(data$alternative %in% names(panel_labels)))
      stop("summary.csv contains an unknown alternative", call. = FALSE)
    panel_levels <- unname(panel_labels[names(panel_labels) %in% data$alternative])
    panel <- unname(panel_labels[data$alternative])
    method <- unname(method_labels[data$method])
    if (study == "01_gaussianity_mle") {
      method_levels <- unname(method_labels[names(method_labels) %in% data$method])
      signal <- data$signal
      seconds <- data$mean_seconds
      title <- "Nuisance estimator: maximum likelihood"
      x_label <- "Signal magnitude"
      timing_label <- paste0("Timing (", panel_levels[[1L]], ")")
    } else {
      method_levels <- unname(method_labels[c("SKSD-influence", "Brueck-corrected")])
      if (!setequal(data$alternative, names(panel_labels)))
        stop("summary.csv does not contain the three expected alternatives", call. = FALSE)
      if (study == "03_conditional_gaussian_score_matching") {
        signal <- data$delta
        seconds <- data$mean_seconds
        timing_label <- "Timing (rewiring)"
      } else {
        signal <- data$signal
        seconds <- data$total_seconds_per_dataset
        timing_label <- "Timing (rotation)"
      }
    }
  }
  if (anyNA(method) || !setequal(method, method_levels))
    stop("summary.csv does not contain the expected method labels", call. = FALSE)
  if (any(!is.finite(signal)) || any(!is.finite(data$rejection_rate)) || any(data$rejection_rate < 0 | data$rejection_rate > 1))
    stop("summary.csv contains invalid numerical results", call. = FALSE)
  if (any(!is.finite(seconds)) || any(seconds <= 0))
    stop("timing panels require finite, strictly positive runtimes", call. = FALSE)
  plot_data <- data.frame(panel = factor(panel, levels = panel_levels), signal = signal,
    method = factor(method, levels = method_levels), rejection_rate = data$rejection_rate,
    seconds_per_100_reps = 100 * seconds)
  if (study == "05_sphere_watson") {
    plot_data$ci_low <- data$ci_low
    plot_data$ci_high <- data$ci_high
  }
  if (anyDuplicated(plot_data[c("panel", "signal", "method")]))
    stop("summary.csv contains a duplicate method/cell row", call. = FALSE)
  counts <- table(interaction(plot_data$panel, plot_data$signal, drop = TRUE), plot_data$method)
  if (any(counts != 1L)) stop("summary.csv has incomplete method/cell rows", call. = FALSE)
  list(data = plot_data, method_levels = method_levels, panel_levels = panel_levels,
    title = title, x_label = x_label, timing_label = timing_label,
    parse_panels = study == "05_sphere_watson")
}

paper_theme <- function() {
  theme_bw(base_size = style_number("axis_tick_font_size")) + theme(
    axis.line = element_line(colour = "black"), panel.grid = element_blank(),
    panel.border = element_blank(), panel.background = element_blank(),
    axis.title = element_text(size = style_number("axis_title_font_size")),
    axis.text = element_text(size = style_number("axis_tick_font_size")),
    strip.background = element_rect(fill = paper_style[["strip_background"]],
      colour = paper_style[["strip_border"]], linewidth = style_number("strip_border_width") / ggplot2::.pt),
    strip.text = element_text(size = style_number("strip_font_size"), margin = margin(4, 3, 4, 3), lineheight = 1),
    plot.title = element_text(hjust = 0.5, size = style_number("figure_title_font_size")), legend.position = "none")
}

timing_box_data <- function(timing_data) {
  ## Compute quartiles and 1.5-IQR whiskers in seconds, before log display.
  boxes <- lapply(split(timing_data, timing_data$method, drop = TRUE), function(group) {
    values <- group$seconds_per_100_reps
    q <- unname(quantile(values, c(0.25, 0.5, 0.75), type = 7))
    inlier <- values >= q[1L] - 1.5 * (q[3L] - q[1L]) & values <= q[3L] + 1.5 * (q[3L] - q[1L])
    data.frame(method = as.character(group$method[[1L]]), ymin = min(q[1L], values[inlier]),
      lower = q[1L], middle = q[2L], upper = q[3L], ymax = max(q[3L], values[inlier]))
  })
  result <- do.call(rbind, boxes)
  result$method <- factor(result$method, levels = levels(timing_data$method))
  rownames(result) <- NULL
  result
}

draw_paper_figure <- function(prepared, alpha = 0.05) {
  if (length(alpha) != 1L || !is.finite(alpha) || alpha <= 0 || alpha >= 1)
    stop("alpha must lie strictly between zero and one", call. = FALSE)
  ## Avoid an implicit Rplots.pdf while cowplot measures grobs.
  grDevices::pdf(file = NULL)
  on.exit(grDevices::dev.off(), add = TRUE)
  data <- prepared$data
  panel_labeller <- if (isTRUE(prepared$parse_panels)) label_parsed else label_value
  rejection <- ggplot(data, aes(signal, rejection_rate, colour = method, group = method)) +
    geom_hline(yintercept = alpha, linetype = "dashed", colour = "red")
  if (all(c("ci_low", "ci_high") %in% names(data)))
    rejection <- rejection + geom_errorbar(aes(ymin = ci_low, ymax = ci_high),
      width = 0, linewidth = 0.3, alpha = 0.75, show.legend = FALSE)
  rejection <- rejection +
    geom_line(linewidth = style_number("line_width") / ggplot2::.pt) +
    geom_point(size = style_number("marker_size") / ggplot2::.pt) +
    facet_wrap(~ panel, nrow = 1L, scales = "free_x", labeller = panel_labeller) +
    scale_colour_manual(values = method_colors, breaks = prepared$method_levels, drop = FALSE) +
    scale_y_continuous(limits = c(0, 1), breaks = c(0, 0.25, 0.5, 0.75, 1)) +
    labs(x = prepared$x_label, y = "Rejection rate", colour = NULL) + paper_theme()
  timing_data <- data[data$panel == prepared$panel_levels[[1L]], , drop = FALSE]
  boxes <- timing_box_data(timing_data)
  boxes$timing_panel <- prepared$timing_label
  bounds <- boxes[match(timing_data$method, boxes$method), ]
  outliers <- timing_data[timing_data$seconds_per_100_reps < bounds$ymin |
    timing_data$seconds_per_100_reps > bounds$ymax, , drop = FALSE]
  outliers$timing_panel <- rep(prepared$timing_label, nrow(outliers))
  timing <- ggplot(boxes, aes(x = method, fill = method, colour = method)) +
    geom_boxplot(aes(ymin = ymin, lower = lower, middle = middle, upper = upper, ymax = ymax),
      stat = "identity", width = 0.62, linewidth = 0.65) +
    geom_point(data = outliers, aes(y = seconds_per_100_reps), size = 1.4, stroke = 0.25) +
    facet_wrap(~ timing_panel, nrow = 1L, labeller = panel_labeller) +
    scale_fill_manual(values = method_colors, drop = FALSE) + scale_colour_manual(values = method_colors, drop = FALSE) +
    scale_y_log10(expand = expansion(mult = c(0.08, 0.12))) +
    labs(x = NULL, y = "Mean seconds per 100 reps") + paper_theme() +
    theme(axis.text.x = element_blank(), axis.ticks.x = element_blank())
  legend_plot <- rejection + theme(legend.position = "bottom", legend.box = "horizontal",
    legend.title = element_blank(), legend.text = element_text(size = style_number("legend_font_size"))) +
    guides(colour = guide_legend(nrow = 1L, byrow = TRUE,
      override.aes = list(linewidth = style_number("line_width") / ggplot2::.pt)))
  legend <- cowplot::get_plot_component(legend_plot, "guide-box-bottom", return_all = FALSE)
  title_plot <- cowplot::ggdraw() + cowplot::draw_label(prepared$title, x = 0.5,
    hjust = 0.5, size = style_number("figure_title_font_size"))
  rejection_grob <- ggplotGrob(rejection)
  timing_grob <- ggplotGrob(timing)
  shared_heights <- grid::unit.pmax(rejection_grob$heights, timing_grob$heights)
  rejection_grob$heights <- shared_heights
  timing_grob$heights <- shared_heights
  panels <- cowplot::plot_grid(rejection_grob, timing_grob, nrow = 1L,
    rel_widths = c(length(prepared$panel_levels), 1), align = "h", axis = "tb", greedy = FALSE)
  final_plot <- cowplot::plot_grid(title_plot, panels, legend, ncol = 1L, rel_heights = c(1.1, 11, 1.6))
  final_plot <- cowplot::ggdraw() + cowplot::draw_plot(final_plot, x = 0.02, y = 0.02, width = 0.96, height = 0.96)
  attr(final_plot, "paper_components") <- list(rejection = rejection, timing = timing, legend = legend)
  final_plot
}

paper_plot_main <- function(study, title_override = NULL) {
  arguments <- commandArgs(trailingOnly = TRUE)
  if (!length(arguments) %in% c(3L, 4L))
    stop("Usage: Rscript plot_STUDY.R summary.csv figure.pdf figure.png [alpha]", call. = FALSE)
  data <- read.csv(normalizePath(arguments[[1L]], mustWork = TRUE), stringsAsFactors = FALSE, check.names = FALSE)
  if (study == "01_gaussianity_mle" &&
      all(c("alt", "nuisance", "reject", "t_per_rep") %in% names(data))) {
    data <- data[data$nuisance == "MLE", , drop = FALSE]
    data$alternative <- data$alt
    data$rejection_rate <- data$reject
    data$mean_seconds <- data$t_per_rep
  }
  alpha <- if (length(arguments) == 4L) as.numeric(arguments[[4L]]) else 0.05
  prepared <- prepare_paper_data(data, study)
  if (!is.null(title_override)) prepared$title <- title_override
  final_plot <- draw_paper_figure(prepared, alpha)
  for (path in arguments[2:3]) {
    dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
    ggsave(path, final_plot, width = style_number("figure_width"), height = style_number("figure_height"),
      dpi = style_number("dpi"), bg = "white")
    message("wrote ", path)
  }
  invisible(final_plot)
}
