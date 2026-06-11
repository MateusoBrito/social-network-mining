import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from src.classification.trainer import ModelTrainer
from src.classification.evaluation import evaluate_model, save_results, plot_confusion_matrix

def run_classic_pipeline(X, y_encoded, splitter, model_class, param_grid, model_name, dataset_path, output_dir, class_names):
    fold_results = []

    for fold, (train_idx, test_idx) in enumerate(splitter.split(X, y_encoded)):
        print(f"\n--- Fold {fold + 1} ---")
        
        # Divisão dos dados
        X_train_raw, X_test_raw = X[train_idx], X[test_idx]
        y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_raw)
        X_test_scaled = scaler.transform(X_test_raw)
                
        # Treinamento e Otimização
        trainer = ModelTrainer(model_class, param_grid, cv=3)
        print("Otimizando parâmetros...")
        trainer.optimize_hyperparameters(X_train_scaled, y_train)
        trainer.train(X_train_scaled, y_train)
        
        # Predição e Avaliação
        y_pred = trainer.predict(X_test_scaled)
        metrics = evaluate_model(y_test, y_pred, labels=class_names)
        
        print(f"Acurácia: {metrics['accuracy']:.4f} | F1-Macro: {metrics['f1_macro']:.4f}")
        
        # Log de resultados
        fold_info = {
            'dataset': os.path.basename(dataset_path),
            'model': model_name,
            'fold': fold + 1,
            'best_params': str(trainer.best_params)
        }
        
        log_path = os.path.join(output_dir, 'experiment_log.csv')
        save_results(metrics, fold_info, log_path)

        # Salvamento de plots
        plots_path = os.path.join(output_dir, 'plots')
        cm_path = os.path.join(plots_path, f'cm_{model_name}_{fold+1}.png')
        plot_confusion_matrix(y_test, y_pred, labels=class_names, filepath=cm_path)
        
        fold_results.append(metrics['accuracy'])

    if fold_results:
        print(f"\n=== Média Final: {np.mean(fold_results):.4f} (+/- {np.std(fold_results):.4f}) ===")
    
    return fold_results 