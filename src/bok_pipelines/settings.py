"""Kedro project settings."""

CONFIG_LOADER_ARGS = {
    "base_env": "base",
    "default_run_env": "local",
    # conf/local/parameters.yml may change single values, e.g. only training.epochs
    "merge_strategy": {"parameters": "soft"},
}
