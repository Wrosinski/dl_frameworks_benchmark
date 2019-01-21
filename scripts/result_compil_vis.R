library('ggplot2')
library('ggrepel')

acc_plot_wf <- function(df, save=FALSE, title='', plot_name='') {
  
  g_full <- ggplot(df, aes(x=reorder(model_name, compilation_time), y=compilation_time)) + 
    # geom_point(aes(color='framework're), size=4) +
    geom_col(fill='navy') +
    geom_label_repel(aes(label=compilation_time)) +
    scale_color_brewer(palette='Set2') +
    labs(x='Model name', y='Epoch time =[s]', title=title) +
    theme_minimal() +
    theme(axis.text.x = element_text(angle=90, hjust=1, size=11))
  g_full
  
  if (save) {
    ggsave(sprintf('figures/ggplot_%s.png', plot_name), g_full, dpi=400, width=8, height=8)
  }
}

acc_plot_facet <- function(df, save=FALSE, title='', plot_name='') {
  
  g_full <- ggplot(df, aes(x=reorder(model_name, compilation_time), y=compilation_time)) + 
    # geom_point(aes(color='framework're), size=4) +
    geom_col(aes(fill=model_name)) +
    geom_label_repel(aes(label=compilation_time)) +
    scale_color_brewer(palette='Set2') +
    labs(x='Model name', y='Epoch time [s]', title=title) +
    theme_minimal() +
    facet_grid(. ~ framework, drop=TRUE) +
    theme(axis.text.x = element_text(angle=90, hjust=1, size=11))
  g_full
  
  if (save) {
    ggsave(sprintf('figures/ggplot_%s.png', plot_name), g_full, dpi=400, width=14, height=8)
  }
}

df <- read.csv('df_compilation_times.csv')

# df$epoch_time <- round(df$epoch_time, 0)
df$compilation_time <- round(df$compilation_time, 1)

df[, 'framework'] = as.factor(df[, 'framework'])
df[, 'model_name'] = as.factor(df[, 'model_name'])
# df[, 'image_size'] = as.factor(df[, 'image_size'])
# df[, 'batch_size'] = as.factor(df[, 'batch_size'])
# df[, 'trial_num'] = as.factor(df[, 'trial_num'])

df_keras <- df[which(df$framework == 'keras'), ]
df_torch <- df[which(df$framework == 'pytorch'), ]

# Plot all:
acc_plot_facet(df, TRUE, 'Compilation times', 'compil_times')  # OK
