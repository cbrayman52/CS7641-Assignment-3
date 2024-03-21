import numpy as np
import pandas as pd

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

import matplotlib.pyplot as plt

def pooled_var(stds):
    return np.sqrt(sum((4)*(stds**2))/ len(stds)*(4))

def learning_curves(dataset, x_train, x_test, y_train, y_test, rs=None):
    mlp = MLPClassifier(max_iter=100, random_state=rs)

    # Train the classifier and store accuracy at each iteration
    train_accuracies = []
    test_accuracies = []

    for i in range(1, mlp.max_iter + 1):
        mlp.partial_fit(x_train, y_train, classes=np.unique(y_train))
        y_train_pred = mlp.predict(x_train)
        y_test_pred = mlp.predict(x_test)
        
        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        
        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)

    plt.plot()
    plt.plot(range(1, mlp.max_iter + 1), train_accuracies, label='Training Accuracy')
    plt.plot(range(1, mlp.max_iter + 1), test_accuracies, label='Test Accuracy')
    plt.xlabel('Number of Iterations')
    plt.ylabel('Accuracy')
    plt.legend(loc="best")
    plt.title('Neural Networks Learning Curve - Wine Quality')
    plt.savefig(f'Images/{dataset}/Learning Curves.png')
    plt.close()
    return

def validation_curves(dataset, x, y, rs=None):
    # Define the parameters to optimize
    params = {'hidden_layer_sizes': [(50,), (20,), (10,)],
              'activation': ['tanh', 'relu', 'identity', 'logistic'],
              'solver': ['sgd', 'adam', 'lbfgs'],
              'learning_rate': ['constant','adaptive', 'invscaling'],
              'max_iter': [10, 50, 100, 500, 1000]}
    
    # Conduct Grid Search over the parameter space to find the best parameters
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3)
    grid_search = GridSearchCV(estimator=MLPClassifier(random_state=rs), param_grid=params, cv=cv, n_jobs=-1, return_train_score=True)
    grid_search.fit(x, y)
    best_params = grid_search.best_params_
    print('NN Best Parameters: ' + str(best_params))
    
    # Generate Validation Curves using results from Grid Search
    df = pd.DataFrame(grid_search.cv_results_)
    results = ['mean_test_score',
            'mean_train_score',
            'std_test_score', 
            'std_train_score']

    fig, axes = plt.subplots(1, len(params), figsize = (5*len(params), 7))
    axes[0].set_ylabel('Score', fontsize=25)

    for idx, (param_name, param_range) in enumerate(params.items()):
        grouped_df = df.groupby(f'param_{param_name}')[results].agg({'mean_test_score' : 'mean',
                                                                     'mean_train_score': 'mean',
                                                                     'std_test_score'  : pooled_var,
                                                                     'std_train_score' : pooled_var})
        if param_name == 'hidden_layer_sizes':
            grouped_df = grouped_df.reset_index()
            grouped_df['param_hidden_layer_sizes'] = grouped_df['param_hidden_layer_sizes'].apply(
                lambda x: 'x'.join(map(str, x)))
            grouped_df.set_index('param_hidden_layer_sizes', inplace=True)
            param_range = ['x'.join(map(str, x)) for x in param_range]

        axes[idx].set_xlabel(param_name, fontsize=30)
        if any(isinstance(element, str) for element in param_range):
            index = np.arange(len(param_range))

            axes[idx].bar(index, grouped_df['mean_train_score'], width=0.3)
            axes[idx].errorbar(index, 
                            grouped_df['mean_train_score'],
                            yerr=grouped_df['std_train_score'],
                            fmt='o',
                            color='r')
            axes[idx].bar(index+0.3, grouped_df['mean_test_score'], width=0.3)
            axes[idx].errorbar(index+0.3, 
                            grouped_df['mean_test_score'],
                            yerr=grouped_df['std_test_score'],
                            fmt='o',
                            color='r')
            axes[idx].set_xticks(index + 0.3 / 2)
            axes[idx].set_xticklabels(param_range)
        else:
            if param_name != 'index':
                axes[idx].plot(param_range, 
                            grouped_df['mean_train_score'],
                            label='Training score',
                            lw=2)
                axes[idx].fill_between(param_range,
                            grouped_df['mean_train_score'] - grouped_df['std_train_score'],
                            grouped_df['mean_train_score'] + grouped_df['std_train_score'],
                            alpha=0.2,
                            color='navy',
                            lw=2)
                axes[idx].plot(param_range,
                            grouped_df['mean_test_score'],
                            label='Cross-validation score',
                            lw=2)
                axes[idx].fill_between(param_range,
                                grouped_df['mean_test_score'] - grouped_df['std_test_score'],
                                grouped_df['mean_test_score'] + grouped_df['std_test_score'],
                                alpha=0.2,
                                color='darkorange',
                                lw=2)

    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc=8, ncol=2, fontsize=20)

    fig.subplots_adjust(bottom=0.25, top=0.85)
    fig.suptitle('Neural Networks Validation Curves', fontsize=40)
    plt.savefig(f'Images/{dataset}/Validation Curves.png')
    plt.close()
    return

def accuracy_curves(dataset, x, y, rs=None):

    mlp = MLPClassifier(
        hidden_layer_sizes=(10,), 
        activation='relu', 
        solver='adam', 
        alpha=0.0001, 
        batch_size='auto', 
        learning_rate='constant', 
        learning_rate_init=0.01, 
        power_t=0.5, 
        max_iter=100, 
        shuffle=True, 
        random_state=4, 
        tol=1e-4, 
        verbose=False, 
        warm_start=False, 
        momentum=0.9, 
        nesterovs_momentum=True, 
        early_stopping=False, 
        validation_fraction=0.1, 
        beta_1=0.9, 
        beta_2=0.999, 
        epsilon=1e-8
    )

    # Initialize arrays to record metrics during training
    training_accuracies = []
    cross_val_accuracies = []
    loss_values = []

    # Number of epochs
    epochs = 100  # You can adjust this based on your preference

    # Training loop
    for epoch in range(epochs):
        # Split the dataset into training and validation sets
        x_train_epoch, x_val, y_train_epoch, y_val = train_test_split(x, y, test_size=0.2, random_state=epoch)

        # Partial fit the model for one epoch
        mlp.partial_fit(x_train_epoch, y_train_epoch, classes=np.unique(y))

        # Training accuracy
        y_train_pred = mlp.predict(x_train_epoch)
        training_accuracy = accuracy_score(y_train_epoch, y_train_pred)
        training_accuracies.append(training_accuracy)

        # Cross-validation accuracy
        y_val_pred = mlp.predict(x_val)
        cross_val_accuracy = accuracy_score(y_val, y_val_pred)
        cross_val_accuracies.append(cross_val_accuracy)

        # Loss during training
        loss_values.append(mlp.loss_)

    # Plotting
    plt.figure(figsize=(8, 6))

    # Plot Training Accuracy
    plt.plot(range(1, epochs + 1), training_accuracies, label='Training Accuracy', linestyle='-')

    # Plot Cross-validation Accuracy
    plt.plot(range(1, epochs + 1), cross_val_accuracies, label='Cross-validation Accuracy', linestyle='-')

    # Plot Loss
    plt.plot(range(1, epochs + 1), loss_values, label='Training Loss', linestyle='-')

    plt.title('Training and Validation Metrics - Backprop')
    plt.xlabel('Epochs')
    plt.ylabel('Metric Value')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'Images/{dataset}/Accuracy Curves.png')
    plt.close()

def evaluate_model(mlp, x_test, y_test):
    y_pred = mlp.predict(x_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

    # Calculate precision
    precision = precision_score(y_test, y_pred)
    print("Precision:", precision)

    # Calculate recall
    recall = recall_score(y_test, y_pred)
    print("Recall:", recall)

    # Calculate F1-score
    f1 = f1_score(y_test, y_pred)
    print("F1-score:", f1)

    # Calculate confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)