library('ggplot2')
library('ggrepel')

acc_plot_wf <- function(df, save=FALSE, title='', plot_name='') {
  
  g_full <- ggplot(df, aes(x=reorder(model_name, epoch_time), y=epoch_time)) + 
    # geom_point(aes(color='framework're), size=4) +
    geom_col(fill='navy') +
    geom_label_repel(aes(label=epoch_time)) +
    scale_color_brewer(palette='Set2') +
    labs(x='Model name', y='Epoch time [s]', title=title) +
    theme_minimal() +
    theme(axis.text.x = element_text(angle=90, hjust=1, size=11))
  g_full
  
  if (save) {
    ggsave(sprintf('figures/ggplot_%s.png', plot_name), g_full, dpi=400, width=8, height=8)
  }
}

acc_plot_facet <- function(df, save=FALSE, title='', plot_name='') {
  
  g_full <- ggplot(df, aes(x=reorder(model_name, epoch_time), y=epoch_time)) + 
    # geom_point(aes(color='framework're), size=4) +
    geom_col(aes(fill=model_name)) +
    geom_label_repel(aes(label=epoch_time)) +
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

df <- read.csv('df_first_epoch.csv')
# df <- read.csv('df_non_first.csv')


df$epoch_time <- round(df$epoch_time, 0)

df[, 'framework'] = as.factor(df[, 'framework'])
df[, 'model_name'] = as.factor(df[, 'model_name'])


df_keras <- df[which(df$framework == 'keras'), ]
df_torch <- df[which(df$framework == 'pytorch'), ]

# Plot all:
acc_plot_facet(df, TRUE, 'First epoch times', 'first_epoch_times')  # OK
