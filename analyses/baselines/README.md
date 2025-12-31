# Prespecified model baselines

All models predict the same log-trans-response outcome under target-grouped cross-validation. The graph baseline uses only covariate similarity within each fold and never propagates outcomes. Test-fold covariates are used to construct the test graph, so its result is reported as transductive rather than inductive.
