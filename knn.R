library(tidyverse)
library(tidymodels)
library(kknn)

# get the penguins dataset
penguins_clean <- as_tibble(datasets::penguins) |> na.omit()
head(penguins_clean, 3)

##################################################################
# Example 1: a first KNN workflow
# Goal: predict penguin `sex` from `bill_len` and `bill_dep` with KNN

# Step 1: specify the model, with k = 5 neighbors
knn_mod <- nearest_neighbor(mode = "classification", neighbors = 5) |>
  set_engine("kknn")

# Step 2: specify the pre-processing recipe
# KNN is based on distances, so scale the predictors
knn_rec <- recipe(sex ~ bill_len + bill_dep, data = penguins_clean) |>
  step_normalize(all_predictors())

# Step 3: build a workflow
knn_wf <- workflow() |> add_recipe(knn_rec) |> add_model(knn_mod)

# Step 4: fit the model
knn_fit <- knn_wf |> fit(data = penguins_clean)

# Step 5: predict
knn_pred <- augment(knn_fit, penguins_clean)
knn_pred |> select(sex, .pred_class, .pred_female, .pred_male) |> head(5)

# Step 6: calculate classification metrics
conf_mat(knn_pred, truth = sex, estimate = .pred_class)
accuracy(knn_pred, truth = sex, estimate = .pred_class)


##################################################################
# k is a hyperparameter: each value of k gives a different model
# predict on a grid of bill length and depth to draw the decision boundary
grid_df <- crossing(
  bill_len = seq(min(penguins_clean$bill_len), max(penguins_clean$bill_len), length.out = 120),
  bill_dep = seq(min(penguins_clean$bill_dep), max(penguins_clean$bill_dep), length.out = 120)
)

boundary_k <- function(k) {
  fit_k <- knn_wf |>
    update_model(nearest_neighbor(mode = "classification", neighbors = k) |> set_engine("kknn")) |>
    fit(data = penguins_clean)
  augment(fit_k, grid_df) |> mutate(k = paste("k =", k))
}

boundaries <- map_dfr(c(1, 5, 25), boundary_k) |>
  mutate(k = fct_inorder(k))

boundaries |>
  ggplot(aes(x = bill_len, y = bill_dep)) +
  geom_raster(aes(fill = .pred_class), alpha = 0.35) +
  geom_point(data = penguins_clean, aes(color = sex), size = 1) +
  facet_wrap(vars(k)) +
  scale_fill_brewer(palette = "Dark2", name = "Predicted") +
  scale_color_brewer(palette = "Dark2", name = "Observed") +
  labs(x = "Bill length (mm)", y = "Bill depth (mm)") +
  theme_bw() +
  theme(aspect.ratio = 1, legend.position = "bottom")


##################################################################
# Which k is best? Not on the training data
# fit k = 1, ..., 20 and compute the accuracy on the same data we fit on
train_acc <- map_dfr(1:20, \(k) {
  knn_wf |>
    update_model(nearest_neighbor(mode = "classification", neighbors = k) |> set_engine("kknn")) |>
    fit(data = penguins_clean) |>
    augment(penguins_clean) |>
    accuracy(truth = sex, estimate = .pred_class) |>
    mutate(k = k)
})

train_acc |>
  ggplot(aes(x = k, y = .estimate)) +
  geom_line() +
  geom_point() +
  scale_x_continuous(breaks = 1:20) +
  labs(y = "Accuracy on training data")


##################################################################
# Example 2: KNN workflow with tuning and cross-validation

# Step 0: training-testing split
set.seed(123)
penguins_split <- initial_split(penguins_clean, prop = 0.75)
penguins_split
penguins_train <- training(penguins_split)
penguins_test  <- testing(penguins_split)
dim(penguins_train)
dim(penguins_test)

# Step 1: specify the model, with the number of neighbors left to tune
knn_tune_mod <- nearest_neighbor(mode = "classification", neighbors = tune()) |> set_engine("kknn")

# Step 2: specify the pre-processing recipe
knn_rec <- recipe(sex ~ bill_len + bill_dep, data = penguins_train) |> step_normalize(all_predictors())

# Step 3: build a workflow
knn_tune_wf <- workflow() |> add_recipe(knn_rec) |> add_model(knn_tune_mod)

# Step 4a: create the cross-validation folds from the training data
penguins_folds <- vfold_cv(penguins_train, v = 10)
penguins_folds

# Step 4b: set up the hyperparameter grid: the values of k to try
knn_grid <- grid_regular(neighbors(range = c(1, 20)), levels = 20)
knn_grid

# Step 4c: fit the model for every k on every fold
knn_tune_res <- tune_grid(knn_tune_wf, resamples = penguins_folds, grid = knn_grid)
knn_tune_res
knn_tune_res |> tidyr::unnest(.metrics)

# Step 4d: average the metrics over the 10 folds
collect_metrics(knn_tune_res) |> filter(.metric == "accuracy")
autoplot(knn_tune_res, metric = "accuracy")

# Step 4e: find the best k
show_best(knn_tune_res, metric = "accuracy", n = 5)
best_k <- select_best(knn_tune_res, metric = "accuracy")
best_k

# Step 4f: plug in the best k and refit on the whole training data
final_wf <- finalize_workflow(knn_tune_wf, best_k)
final_wf
final_fit <- fit(final_wf, data = penguins_train)

# Step 5/6: predict and calculate classification metrics on the testing data, once
res_final <- augment(final_fit, penguins_test)
metrics(res_final, truth = sex, estimate = .pred_class)

# ROC curve on the testing data
# male is the positive class (1): yardstick uses the first level (female) by default,
# so set event_level = "second"
roc_curve(res_final, truth = sex, .pred_male, event_level = "second") |> autoplot()
roc_auc(res_final, truth = sex, .pred_male, event_level = "second")
