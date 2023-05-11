# Random Forest, Elastic Net and Linear Regression training scripts
This API is product specific. It is not applicable for use in other products than those it was designed for. 

## Description of Packages
### data_import:
- query data from the database
- cache data in files
- encode feature values
### utils:
- objects for representing feature tables, featuring transformations and export to csv
- library for parallel task execution
- csv library
### sklearn_extension:
- CategoricalRegressionClassifier allowing classification using scikit-learn regression models
### experiment_suite:
- API for building model training tasks
- API for executing model training tasks
### experiment_definition:
- Experiments used in this work
  - train Random Forest with different tree_depth values
  - train Elastic Net
  - train Linear Regression
  - train Random Forest with reduced feature set based on Random Forest feature importance
  - train Random Forest with reduced feature set based on Elastic Net feature selection