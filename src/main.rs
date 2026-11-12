
use std::{error::Error, fs};

use linfa::prelude::*;
use linfa_trees::{DecisionTree, SplitQuality};

fn main() -> Result<(), Box<dyn Error>> {

    // 1. Load the Iris dataset
    let dataset = linfa_datasets::iris();

    println!("Number of observations: {}", dataset.nsamples());
    println!("Number of features: {}", dataset.nfeatures());

    // 2. Split into training and testing datasets
    let (train, test) = dataset.split_with_ratio(0.8);

    println!("Training samples: {}", train.nsamples());
    println!("Testing samples: {}", test.nsamples());

    // 3. Configure the decision tree
    let model = DecisionTree::params()
        .max_depth(Some(3))
        .split_quality(SplitQuality::Gini);

    // 4. Train the model
    let decision_tree = model.fit(&train)?;

    println!("\nDecision tree trained successfully!");

    // 5. Make predictions
    let predictions = decision_tree.predict(test.records());

    // 6. Evaluate the model
    let confusion_matrix = predictions.confusion_matrix(&test)?;

    let accuracy = confusion_matrix.accuracy();

    println!("\nConfusion Matrix:");
    println!("{:#?}", confusion_matrix);

    println!("\nAccuracy: {:.2}%", accuracy * 100.0);

    // 7. Examine model complexity
    println!("\nTree depth: {}", decision_tree.max_depth());
    println!("Number of leaves: {}", decision_tree.num_leaves());

    // 8. Examine feature importance
    let importance = decision_tree.feature_importance();

    let feature_names = [
        "Sepal Length",
        "Sepal Width",
        "Petal Length",
        "Petal Width",
    ];

    println!("\nFeature Importance:");

    for (index, score) in importance.iter().enumerate() {
        println!(
            "{}: {:.4}",
            feature_names[index],
            score
        );
    }

    // 9. Export the tree to TikZ
    let tikz = decision_tree.export_to_tikz().with_legend();

    fs::write(
        "decision_tree_iris.tex",
        tikz.to_string()
    )?;

    println!("\nDecision tree exported successfully!");

    Ok(())
}
